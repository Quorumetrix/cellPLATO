# pipelines.py

from initialization.config import *
from data_processing.cell_identifier import *
from data_processing.cleaning_formatting_filtering import *
from data_processing.clustering import *
from data_processing.data_wrangling import *
from data_processing.dimensionality_reduction import *
from data_processing.measurements import *
from data_processing.statistics import *
from data_processing.time_calculations import *
from data_processing.trajectory_clustering import *

from visualization.comparative_visualization import *
from visualization.plots_of_differences import *
from visualization.superplots import *
from visualization.timecourse_visualization import *

import os
import numpy as np
import pandas as pd

import datetime
from tqdm import tqdm

'''
Still missing many of the old ones from original pipelines.py
Wanted to be sure they are all up to date.
'''


def measurement_pipeline(comb_df, mixed=MIXED_SCALING):

    # Calculate the clusteredness in xy-space
    comb_df = calc_ripleys_xy(comb_df, plot=False) #'''This is still done in pixels'''

    # Calculate the cells aspect ratio
    calc_aspect_ratio(comb_df, drop=True)# if there are resulting nans

    # Clean up remaining dataframe, and calibrate the micron-dependent values
    cleaned_df = clean_comb_df(comb_df, deduplicate=False)
    # comb_df = factor_calibration(cleaned_df)

    print('Calibrating with mixed_scaling = ', MIXED_SCALING)
    comb_df = factor_calibration(cleaned_df,mixed_calibration=mixed)
    apply_unique_id(comb_df)

    if(SELF_STANDARDIZE):
        print('Self-standardizing factors: ',FACTORS_TO_STANDARDIZE)
        comb_df = standardize_factors_per_cell(comb_df,FACTORS_TO_STANDARDIZE)


    if(FACTOR_TIMEAVERAGE):

        # Which factors should we calculate acorss the cells time window.
        factors_to_timeaverage = DR_FACTORS
        print('Time-averaging factors: ', factors_to_timeaverage)
        comb_df, add_fac_list = t_window_metrics(std_comb_df,factor_list=factors_to_timeaverage)

    # Reset index since aspect calc may have dropped rows.
    comb_df.reset_index(inplace=True, drop=True)

    return comb_df

def dr_pipeline(df, dr_factors=DR_FACTORS, dr_input='factors', tsne_perp=TSNE_PERP,umap_nn=UMAP_NN,min_dist=UMAP_MIN_DIST):

    '''
    An updated dimensionality reduction that performs PCA, tSNE and UMAP,
    and returns the combined dataframe.
    Function intended to be useful as a single call with default values (from config)
    A single call with user-defined input parameters (i.e. following a sweep or optimization)
    OR as a part of a parameter sweep.

    Input:
        df
        dr_factors: list of factors to use when extracting the data matrix
                    default: DR_FACTORS constant from the config.
        dr_input: string indicating what to use as input to the tSNE and UMAP functions.
                default: 'factors' standardized factors directly. Alternatively 'PCs' will use PCA output.

        tsne_perp: default = TSNE_PERP
        umap_nn:default = UMAP_NN
        min_dist:default = UMAP_MIN_DIST


    '''

    print('Running dr_pipeline...')
    print('tSNE perplexity = ',tsne_perp)
    print('UMAP nearest neighbors = ', umap_nn, ' min distance = ',min_dist)

    # Prepare data for dimensionality reduction by extracting the factors of interest from the DR_FACTORS list.
    x = get_data_matrix(df,dr_factors)

    # Principal component analysis
    pca_df, _, _ = do_pca(x)

    if dr_input == 'factors':

        x_ = StandardScaler().fit_transform(x)
        print('Using standardized factors for dimensionality reduction, matrix shape: ', x_.shape)

    elif dr_input == 'PCs':

        x_ = pca_df.values
        print('Using Principal Components for dimensionality reduction, matrix shape: ', x_.shape)

    # openTSNE using default vaues
    '''
    This should be replaced by a version of tSNE that allows us to set
    the perplexity value right here, or use the default.
    Not just the default as currently.

    Otherwsie need to use other function test_tsne()
    '''
    tsne_x, flag = do_open_tsne(x_,perplexity=tsne_perp)

    # # openTSNE using default vaues (Previously...)
    # tsne_x, flag = do_open_tsne(x)

    # Format tSNE results into a dataframe
    tsne_df = pd.DataFrame(data = tsne_x, columns = ['tSNE1', 'tSNE2'])
    # tsne_df['used_existing'] = flag


    # Use standrard scaler upstream of tSNE.
    umap_x = do_umap(x_, n_neighbors=umap_nn, min_dist=min_dist)
    umap_df = pd.DataFrame(data = umap_x, columns = ['UMAP1', 'UMAP2'])

    # Create Dimension-reduced dataframe by adding PCA and tSNE columns.
    dr_df = pd.concat([df,pca_df,tsne_df, umap_df], axis=1)

    assert list(df.index) == list(dr_df.index), 'dr_df should be the same length as input dataframe. Check indexing of input dataframe.'

    return dr_df




def trajectory_clustering_pipeline(dr_df, traj_factor=CLUSTER_BY, zeroed=False, dist_metric='hausdorff', filename_out='trajectory_clustering_test'):

    cell_df_list = get_cell_df_list(dr_df,length_threshold=10)


    traj_list = get_trajectories(cell_df_list,traj_factor=traj_factor, interp_pts=20, zeroed=zeroed)

    '''
    A few alternative processing steps that aren't currently in use,
    kept here for potentially working them back in at a later date.
    '''

    # traj_list = get_trajectories(cell_df_list,traj_factor='tSNE',
    #                              interp_pts=None, zeroed=False, method='segment')

    # traj_list = simplify_trajectories(traj_list, method='rdp', param=1)
    # traj_list = simplify_trajectories(traj_list, method='vw', param=5)

    # seg_list = get_trajectory_segments(traj_list)
    # D = trajectory_distances(seg_list, method='frechet')# 'hausdorff', 'dtw'



    D = trajectory_distances(traj_list, method=dist_metric)# 'hausdorff', 'dtw'
    eps = find_max_clusters(D)

    # Cluster the distance matrix for the trajectories list
    cluster_lst = cluster_trajectories(traj_list, D, eps, filename_out+'_'+traj_factor)


    # Add the trajectory id (label) back into the original dataframe.
    traj_clust_df = traj_clusters_2_df(dr_df, cell_df_list, cluster_lst)

    return traj_clust_df, traj_list, cluster_lst


def comparative_visualization_pipeline(df, num_factors=NUM_FACTORS, factor_pairs=FACTOR_PAIRS):


    '''
    A pipeline that produces graphics for each of the requested numerical factors
    and pairs of factors

    Input:
        df: DataFrame containing multiple conditions to compare
        num_factors: local list of numerical factors to generate plots from
            default: NUM_FACTORS defined in config
        factor_pairs: local list of factor pairs to generate plots from
            default: FACTOR_PAIRS defined in config

    '''

    # Be sure there are no NaNs before starting the visualization pipeline.
    df.dropna(inplace=True)

    assert len(df['Condition'].unique()) > 1, 'comparative_visualization_pipeline() must have >1 unique conditions in input dataframe'

    # Process a time-averaged DataFrame
    tavg_df = time_average(df)

    # Make summary calculations from time-averaged dataframe
    #Per condition:
    avg_df = average_per_condition(tavg_df)

    # Per replicate
    repavg_df = average_per_condition(tavg_df, avg_per_rep=True)

    '''
    A hypothesis-testing, p-value calculation function would go here,
    and run its own loop over factors, generating a single .txt file

    ** May be better WITHIN average_per_condition() **
    '''


    # Create comparative plots for each of the numerical factors
    for factor in tqdm(num_factors):
    # for factor in num_factors:

        '''
        Summary comparison of means between conditions and replicates

        Note:
            - Displaying error bars for SDEV or SEM will be more complicated, will involve creating other dataframes
            or adding another nested set of measurements to the existing avg_df or repavg_df, in the functions where
            they are being generated, average_per_condition().

        For now, the comparative_bar() plots in comparative_visualization.py exists as a convenience function and placeholder
        for something potentially more elaborate in the future.

        '''
        print('Processing factor: ', factor)

        print('Processing statistics...')
        stats_table(tavg_df, factor)


        if DRAW_BARPLOTS:
            print('Exporting Comparative bar charts... ')
            cond_stats = average_per_condition(tavg_df, avg_per_rep=False)
            comparative_bar(cond_stats, x='Condition', y=factor, to_plot='avg',title='_per_condition_')
            comparative_bar(cond_stats, x='Condition', y=factor, to_plot='n',title='_per_condition_')
            rep_stats = average_per_condition(tavg_df, avg_per_rep=True)
            comparative_bar(rep_stats, x='Replicate_ID', y=factor, to_plot='avg', title='_per_replicate_')
            comparative_bar(rep_stats, x='Replicate_ID', y=factor, to_plot='n', title='_per_replicate_')


        if DRAW_SUPERPLOTS:
            print('Exporting static Superplots...')

            # Time-averaged superplots
            superplots_plotly(tavg_df, factor, t='timeaverage')
            # superplots(tavg_df,factor , t='timeaverage')

        if DRAW_DIFFPLOTS:
            print('Exporting static Plots of Differences')
           # Time-averaged plots-of-differences
            plots_of_differences_plotly(tavg_df, factor=factor, ctl_label=CTL_LABEL)
            plots_of_differences_sns(tavg_df, factor=factor, ctl_label=CTL_LABEL)

        if DRAW_TIMEPLOTS:
            print('Exporting static Timeplots')
            print('Time superplots..')
            time_superplot(df, factor)


    if DRAW_MARGSCAT:
        print('Exporting static Marginal scatterplots')
        print('Processing factor pairs: ')

        # for pair in tqdm(factor_pairs):
        for pair in factor_pairs:

            print('Currently generating scatter, hex, contour plots for pair: ', pair)

            '''
            Consider refactoring the section below into a more concise function.
            '''

            if(AXES_LIMITS == 'min-max'):
                # Calculate bounds of entire set:
                x_min = np.min(tavg_df[pair[0]])
                x_max = np.max(tavg_df[pair[0]])
                y_min = np.min(tavg_df[pair[1]])
                y_max = np.max(tavg_df[pair[1]])

            elif(AXES_LIMITS == '2-sigma'):
                # Set the axes limits custom (3 sigma)
                x_min = np.mean(tavg_df[pair[0]]) - 2 * np.std(tavg_df[pair[0]])
                x_max = np.mean(tavg_df[pair[0]]) + 2 * np.std(tavg_df[pair[0]])
                y_min = np.mean(tavg_df[pair[1]]) - 2 * np.std(tavg_df[pair[1]])
                y_max = np.mean(tavg_df[pair[1]]) + 2 * np.std(tavg_df[pair[1]])

            elif(AXES_LIMITS == '3-sigma'):
                # Set the axes limits custom (3 sigma)
                x_min = np.mean(tavg_df[pair[0]]) - 3 * np.std(tavg_df[pair[0]])
                x_max = np.mean(tavg_df[pair[0]]) + 3 * np.std(tavg_df[pair[0]])
                y_min = np.mean(tavg_df[pair[1]]) - 3 * np.std(tavg_df[pair[1]])
                y_max = np.mean(tavg_df[pair[1]]) + 3 * np.std(tavg_df[pair[1]])


            bounds = x_min, x_max,y_min, y_max
            print(AXES_LIMITS, ' bounds: ', x_min, x_max, y_min, y_max)

            # Make the combined versions of all plots first.
            marginal_xy(tavg_df, pair, plot_type = 'scatter', renderer='seaborn') # All conditions shown.
            marginal_xy(tavg_df, pair, plot_type = 'hex', renderer='seaborn') # All conditions shown.
            marginal_xy(tavg_df, pair, plot_type = 'contour', renderer='seaborn') # All conditions shown.


            # Generate separate plots for each condition.
            for cond in tavg_df['Condition'].unique():
                cond_sub_df = tavg_df[tavg_df['Condition'] == cond]
                if USE_SHORTLABELS:
                    this_label = cond_sub_df['Condition_shortlabel'].unique()[0]
                else:
                    this_label = cond

                marginal_xy(cond_sub_df, pair, plot_type = 'contour', renderer='seaborn', bounds=bounds, supp_label=this_label)
                marginal_xy(cond_sub_df, pair, plot_type = 'hex', renderer='seaborn', bounds=bounds,supp_label=this_label)
                marginal_xy(cond_sub_df, pair, plot_type = 'scatter', renderer='seaborn', bounds=bounds, supp_label=this_label)


def cluster_analysis_pipeline(df,cluster_by, eps=None):#=CLUSTER_BY):

    '''
    A <pipeline> function that groups together elements of the workflow that involve clustering on tSNE, PCA or XY data,
    generating a labelled DataFrame (lab_dr_df), and generating plots that compare subgroup behaviors.


    Input:
        df: DataFrame - containing dimension reduced columns. (tSNE1, tSNE2, PC1...)
        cluster_by: 'tSNE', 'PCA' or 'xy'
        ***Also allow overriding of clustering values EPS and min_samples??**
            defaults to CLUSTER_BY global defined in the config.

    Returns:
        lab_dr_df: DataFrame containing an extra column, 'label' for each cells group ID at a given timepoint,
        where a value of -1 is for uncategorized cells.

    '''

    print('Clustering by: ' + cluster_by)
    print('Plot output folder :',CLUST_DIR)

    if cluster_by == 'xy':
        x_name = 'x'
        y_name = 'y'
        curr_save_path = CLUST_XY_DIR # Keep in root along with cluster_counts
        print('DBScan clustering by x,y position...')

    elif cluster_by == 'pca':
        x_name = 'PC1'
        y_name = 'PC2'
        curr_save_path = CLUST_PCA_DIR
        print('DBScan clustering by principal components...')

    elif cluster_by == 'tsne':
        x_name = 'tSNE1'
        y_name = 'tSNE2'
        curr_save_path = CLUST_TSNE_DIR
        print('DBScan clustering by tSNE...')

    elif cluster_by == 'umap':
        x_name = 'tSNE1'
        y_name = 'tSNE2'
        curr_save_path = CLUST_TSNE_DIR
        print('DBScan clustering by tSNE...')

    # Allow override of the EPS from config in dbscan_clustering
    if eps is not None:
        # Cluster the input dataframe by the specified factors
        lab_dr_df = dbscan_clustering(df,cluster_by=cluster_by, eps=eps,
                                      plot=True, save_path=curr_save_path)
    else:
        # Cluster the input dataframe by the specified factors
        lab_dr_df = hdbscan_clustering(df,cluster_by=cluster_by,
                                      plot=True, save_path=curr_save_path)

    lab_count_df = get_label_counts(lab_dr_df, per_rep=True)


    # # Plot the counts per subgroup in a swarm plot
    # fig = px.strip(lab_count_df, x="label", y="count", color="Condition")
    #
    # if STATIC_PLOTS:
    #     fig.write_image(curr_save_path+'cluster_counts_' + cluster_by + '.png')
    #
    # if PLOTS_IN_BROWSER:
    #     fig.show()

    for factor in NUM_FACTORS:

        plots_of_differences_plotly(lab_dr_df, factor, ctl_label=-1, cust_txt=cluster_by+'_clustered_', save_path=curr_save_path)


    # For each label, create a subfolder and export superplots (seaborn)
    for label in lab_dr_df['label'].unique():

        print('label: ', label)
        curr_label_path = os.path.join(curr_save_path,'label_'+str(label)+'/')
        print('curr_label_path: ', curr_label_path)
        if not os.path.exists(curr_label_path):
             os.makedirs(curr_label_path)

        # Get the sub_df for this label
        this_lab_df = lab_dr_df[lab_dr_df['label'] == label]

        # print('Reminder, cannot time-average labelled datasets, producing plots without time-averaging')

        for factor in NUM_FACTORS:

            if(DRAW_DIFFPLOTS):
                plots_of_differences_plotly(this_lab_df, factor=factor,
                                            ctl_label=CTL_LABEL, save_path=curr_label_path)

    return lab_dr_df



def cluster_switching_pipeline(lab_dr_df):

    assert 'label' in lab_dr_df.columns, 'dataframe passed to cluster_switching_pipeline() must have cluster <labels>'

    # Count the cluster changes
    sum_labels, tptlabel_dr_df = count_cluster_changes(lab_dr_df)

    time_superplot(tptlabel_dr_df, 'n_changes',t_window=None)

    clust_sum_df = cluster_purity(lab_dr_df)

    trajclust_sum_df = cluster_purity(lab_dr_df, cluster_label='traj_id')

    purity_plots(lab_dr_df, clust_sum_df,lab_dr_df,trajclust_sum_df)

    clust_sum_t_df = cluster_composition_timecourse(lab_dr_df)
    cluster_timeplot(clust_sum_t_df)

    # Count the number of cells that fall into each cluster - show on per condition and replicate basis.
    lab_count_df = get_label_counts(lab_dr_df, per_rep=True)

    # # Plot the counts per subgroup in a swarm plot
    # fig = px.strip(lab_count_df, x="label", y="count", color="Condition")
    # fig.show()

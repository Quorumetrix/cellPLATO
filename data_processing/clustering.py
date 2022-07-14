#clustering.py

from initialization.config import *
from visualization.cluster_visualization import *

import os
import numpy as np
import pandas as pd

from tqdm import tqdm

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS
import hdbscan

'''
K. Chaudhuri and S. Dasgupta. “Rates of convergence for the cluster tree.”
In Advances in Neural Information Processing Systems, 2010.
'''

def dbscan_clustering(df_in, eps=EPS, min_samples=MIN_SAMPLES,cluster_by='tsne', plot=False, save_path=CLUST_DIR):

    print('dbscan_clustering() with eps = ', eps)
    # Determine how to cluster
    x_name,y_name = ' ', ' '

    if cluster_by == 'xy':
        x_name = 'x'
        y_name = 'y'
        # save_path = CLUST_XY_DIR # Keep in root along with cluster_counts
        print('DBScan clustering by x,y position...')

    elif (cluster_by == 'pca' or cluster_by == 'PCA'):
        x_name = 'PC1'
        y_name = 'PC2'
        # save_path = CLUST_PCA_DIR
        print('DBScan clustering by principal components...')

    elif (cluster_by == 'tsne' or cluster_by == 'tSNE'):
        x_name = 'tSNE1'
        y_name = 'tSNE2'
        # save_path = CLUST_TSNE_DIR
        print('DBScan clustering by tSNE...')

    elif (cluster_by == 'umap' or cluster_by == 'UMAP'):
        x_name = 'UMAP1'
        y_name = 'UMAP2'
        # save_path = CLUST_TSNE_DIR
        print('DBScan clustering by UMAP...')


    #DBScan
    sub_set = df_in[[x_name, y_name]] # self.df
    X = StandardScaler().fit_transform(sub_set)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Assemble a dataframe from the results
    lab_df = pd.DataFrame(data = labels, columns = ['label'])
    lab_dr_df = pd.concat([df_in,lab_df], axis=1)

    if(plot):
        import seaborn as sns
        import matplotlib.pyplot as plt

        ''' Eventually this plotting function should probably be in another script.'''
        scatter_fig = sns.jointplot(data=lab_dr_df[lab_dr_df['label']!=-1],x=x_name, y=y_name, hue="label",
                          kind='scatter',
                          palette=CLUSTER_CMAP,
                          joint_kws={'alpha': 0.4,'s': 5}, height=10, legend=False)
        if STATIC_PLOTS:
            # plt.savefig(CLUST_PARAMS_DIR+cluster_by+'_sweep_eps_'+str(eps)+'.png', dpi=300)
            plt.savefig(save_path+'cluster_scatter_'+cluster_by+'.png')
        if PLOTS_IN_BROWSER:
            plt.show()


    print('Clustering generated: '+str(len(lab_dr_df['label'].unique())) + ' subgoups.')

    return lab_dr_df


def hdbscan_clustering(df_in, min_cluster_size=20,cluster_by='tsne', plot=False, save_path=CLUST_DIR):

    print('hdbscan_clustering() with min_cluster_size = ', min_cluster_size)
    # Determine how to cluster
    x_name,y_name = ' ', ' '

    if cluster_by == 'xy':
        x_name = 'x'
        y_name = 'y'
        # save_path = CLUST_XY_DIR # Keep in root along with cluster_counts
        print('DBScan clustering by x,y position...')

    elif (cluster_by == 'pca' or cluster_by == 'PCA'):
        x_name = 'PC1'
        y_name = 'PC2'
        # save_path = CLUST_PCA_DIR
        print('DBScan clustering by principal components...')

    elif (cluster_by == 'tsne' or cluster_by == 'tSNE'):
        x_name = 'tSNE1'
        y_name = 'tSNE2'
        # save_path = CLUST_TSNE_DIR
        print('DBScan clustering by tSNE...')

    elif (cluster_by == 'umap' or cluster_by == 'UMAP'):
        x_name = 'UMAP1'
        y_name = 'UMAP2'
        # save_path = CLUST_TSNE_DIR
        print('DBScan clustering by UMAP...')


    #hDBScan
    sub_set = df_in[[x_name, y_name]] # self.df
    X = StandardScaler().fit_transform(sub_set)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    labels = clusterer.fit_predict(X)

    # Assemble a dataframe from the results
    lab_df = pd.DataFrame(data = labels, columns = ['label'])
    lab_dr_df = pd.concat([df_in,lab_df], axis=1)

    if(plot):
        import seaborn as sns
        import matplotlib.pyplot as plt

        ''' Eventually this plotting function should probably be in another script.'''
        scatter_fig = sns.jointplot(data=lab_dr_df[lab_dr_df['label']!=-1],x=x_name, y=y_name, hue="label",
                          kind='scatter',
                          palette=CLUSTER_CMAP,
                          joint_kws={'alpha': 0.4,'s': 5}, height=10, legend=False)
        if STATIC_PLOTS:
            # plt.savefig(CLUST_PARAMS_DIR+cluster_by+'_sweep_eps_'+str(eps)+'.png', dpi=300)
            plt.savefig(save_path+'cluster_scatter_'+cluster_by+'.png')
        if PLOTS_IN_BROWSER:
            plt.show()


    # print('HDBscan Clustering generated: '+str(len(lab_dr_df['label'].unique())) + ' subgoups.')

    return lab_dr_df


def optics_clustering(df_in, min_samples=MIN_SAMPLES,cluster_by='tsne', plot=False, save_path=CLUST_DIR):


    # Determine how to cluster
    x_name,y_name = ' ', ' '

    if cluster_by == 'xy':
        x_name = 'x'
        y_name = 'y'
        # save_path = CLUST_XY_DIR # Keep in root along with cluster_counts
        print('OPTICS clustering by x,y position...')

    elif (cluster_by == 'pca' or cluster_by == 'PCA'):
        x_name = 'PC1'
        y_name = 'PC2'
        # save_path = CLUST_PCA_DIR
        print('OPTICS clustering by principal components...')

    elif (cluster_by == 'tsne' or cluster_by == 'tSNE'):
        x_name = 'tSNE1'
        y_name = 'tSNE2'
        # save_path = CLUST_TSNE_DIR
        print('OPTICS clustering by tSNE...')

    elif (cluster_by == 'umap' or cluster_by == 'UMAP'):
        x_name = 'UMAP1'
        y_name = 'UMAP2'
        # save_path = CLUST_TSNE_DIR
        print('OPTICS clustering by UMAP...')


    # Optics
    sub_set = df_in[[x_name, y_name]] # self.df
    X = StandardScaler().fit_transform(sub_set)
    clustering = OPTICS(min_samples=min_samples).fit(X)
    core_samples_mask = np.zeros_like(clustering.labels_, dtype=bool)
    labels = clustering.labels_


    # Assemble a dataframe from the results
    lab_df = pd.DataFrame(data = labels, columns = ['label'])
    lab_dr_df = pd.concat([df_in,lab_df], axis=1)

    if(plot):
        import seaborn as sns
        import matplotlib.pyplot as plt

        ''' Eventually this plotting function should probably be in another script.'''
        scatter_fig = sns.jointplot(data=lab_dr_df[lab_dr_df['label']!=-1],x=x_name, y=y_name, hue="label",
                          kind='scatter',
                          palette=CLUSTER_CMAP,
                          joint_kws={'alpha': 0.4,'s': 5}, height=10, legend=False)
        if STATIC_PLOTS:
            plt.savefig(save_path+'cluster_scatter_'+cluster_by+'.png')
        if PLOTS_IN_BROWSER:
            plt.show()

    return lab_dr_df


def cluster_purity(lab_dr_df, cluster_label='label'):

    '''
    Calculate purity of input dataframe clusters with respect to the experimental condition.

    Input:
        lab_dr_df: pd.DataFrame containing cluster ids in the 'label' column.

    '''

    assert cluster_label in lab_dr_df.columns, 'Dataframe must contain cluster labels'
    assert 'Condition_shortlabel' in lab_dr_df.columns, 'For now, assuming shortlabels in use.'

    cond_list = lab_dr_df['Condition_shortlabel'].unique()


    # Create a new dataframe to hold the cluster summary info.
    clust_sum_df = pd.DataFrame()

    clusters = list(set(lab_dr_df[cluster_label].dropna())) # DropNA to also handle trajectory_ids where some are NaN

    for cluster_id in clusters[:-1]: # Skip last one that is noise (-1)


        clust_sum_df.at[cluster_id,'cluster_id'] = cluster_id

        # Get the dataframe for this cluster.
        clust_sub_df = lab_dr_df[lab_dr_df[cluster_label] == cluster_id]


        for cond in cond_list:


            cond_clust_sub_df = clust_sub_df[clust_sub_df['Condition_shortlabel'] == cond]

            # Count the number of timepoints for this condition in the cluster
            clust_sum_df.at[cluster_id,cond+'_ntpts'] = len(cond_clust_sub_df)
            clust_sum_df.at[cluster_id,cond+'_ntpts_%'] = len(cond_clust_sub_df) / len(clust_sub_df) * 100

            # Count the number of unique cells for this condition in the cluster
            clust_sum_df.at[cluster_id,cond+'_ncells'] = len(cond_clust_sub_df['uniq_id'].unique())
            clust_sum_df.at[cluster_id,cond+'_ncells_%'] = len(cond_clust_sub_df['uniq_id'].unique()) / len(clust_sub_df['uniq_id'].unique()) * 100

    return clust_sum_df







def count_cluster_changes(df):

    '''
    Count the number of changes to the cluster ID
    '''

    assert 'label' in df.columns, 'count_cluster_changes() must be provided with label-containing dataframe.'

    lab_list = []
    lab_list_tpt = []

    for rep in df['Replicate_ID'].unique():

        for cell_id in df[df['Replicate_ID'] == rep]['particle'].unique():

            cell_df = df[(df['Replicate_ID']==rep) &
                            (df['particle']==cell_id)]

            # If you've already attributed a unique id, then use it.
            if 'uniq_id' in cell_df.columns:
                assert len(cell_df['uniq_id'].unique()) == 1
                uniq_id = cell_df['uniq_id'].unique()[0]
            else:
                uniq_id = cell_id

            # Calculate at the per-cell level
            s = cell_df['label']
            label_changes = (np.diff(s)!=0).sum() # Count the number of times it changes
            n_labels = len(s.unique()) # Count the number of times it changes
            lab_list.append((cell_df['Condition'].unique()[0],rep, uniq_id, label_changes,n_labels))

            # The extra step of doing these calculations cumulatively, per timepoint.
            for t in cell_df['frame'].unique():

                # Get the dataframe including all timepoints upto and including t
                cumulative_df = cell_df[(cell_df['frame']<=t)]

                # Count the label changes and total numbers.
                s = cumulative_df['label']
                label_changes = (np.diff(s)!=0).sum() # Count the number of times it changes
                n_labels = len(s.unique()) # Count the number of times it changes
                lab_list_tpt.append((cell_df['Condition'].unique()[0],rep, uniq_id, t, label_changes,n_labels))

    # Turn the lists to dataframes
    lab_list_tpt_df = pd.DataFrame(lab_list_tpt, columns=['Condition','Replicate', 'Cell_ID','frame', 'n_changes', 'n_labels'])
    lab_list_df = pd.DataFrame(lab_list, columns=['Condition','Replicate', 'Cell_ID', 'n_changes', 'n_labels'])

    assert len(lab_list_tpt_df.index) == len(df.index), 'Label count list doesnt add up'

    # Test that this lines up with the original dataframe
    for i in range(len(lab_list_tpt_df)):

        cond_1 = df.iloc[i]['Condition']
        cond_2 = lab_list_tpt_df.iloc[i]['Condition']

        rep_1 = df.iloc[i]['Replicate_ID']
        rep_2 = lab_list_tpt_df.iloc[i]['Replicate']

        # id_1 = df.iloc[i]['particle']
        # id_2 = lab_list_tpt_df.iloc[i]['Cell_ID']

        assert cond_1 == cond_2
        assert rep_1 == rep_2
        # assert id_1 == id_2

    # Insert the timepoint label counts back into the input dataframe
    new_df = pd.concat([df,lab_list_tpt_df[['n_changes','n_labels']]], axis=1)

    return lab_list_df, new_df


def dbscan(df_in, x_name, y_name, eps, cust_label = 'label'):

    '''
    Pretty sure this is a clustering accessory function
    that is now obeselete
    '''
    print('dbscan with eps=', eps)
    # DBScan
    sub_set = df_in[[x_name, y_name]] # self.df
    X = StandardScaler().fit_transform(sub_set)
    db = DBSCAN(eps=eps, min_samples=MIN_SAMPLES).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    print(np.unique(labels))
    # Assemble a dataframe from the results
    lab_df = pd.DataFrame(data = labels, columns = [cust_label])

    return lab_df





def get_label_counts(df, per_rep=False):


    '''
    Takes a DBscan labelled dataframe as input, and computes the number of cells that has each of the labels,
    separated by condition or replicate.

    input:
        df: dataframe that must contain a 'label' column (i.e. output from  dbscan_clustering())

        per_rep: Boolean, if True - each row in the summary dataframe corresponds to an individual replicate,
            otherwise defaults to one row per conditon.

    returns:
        label_counts_df: dataframe with the columns: Condition, Replicate_ID, label, count


    Note: Output easily visualized by plotly express stripplot.
        fig = px.strip(lab_count_df, x="label", y="count", color="Condition")

    '''

    assert 'label' in df.columns, 'No label in dataframe'
    labels = df['label'].unique() #IMportantly computed for the whole set.
    label_counts_df = pd.DataFrame(columns=['Condition', 'Replicate_ID', 'label', 'count'])

    # Per condition OR per replicate
    cond_list = df['Condition'].unique()
    rep_list = df['Replicate_ID'].unique()

    i=0

    if(per_rep):


        for this_rep in rep_list:

            this_rep_df = df[df['Replicate_ID'] == this_rep]

            assert len(this_rep_df['Condition'].unique()) == 1, 'Condition not unique in rep_df'

            # Count how many rows for each label
            for label in labels:

                 # Keep this dataframe being made for when we want to look at distributions
                this_lab_df = this_rep_df[this_rep_df['label'] == label]

                label_counts_df.loc[i] = [this_rep_df['Condition'].unique()[0], this_rep, label, len(this_lab_df.index) ]
                i+=1

    else:

        for cond in cond_list:

            this_cond_df = df[df['Condition'] == cond]


            # Count how many rows for each label
            for label in labels:

                 # Keep this dataframe being made for when we want to look at distributions
                this_lab_df = this_cond_df[this_cond_df['label'] == label]

                label_counts_df.loc[i] = [cond, 'NA', label, len(this_lab_df.index) ]
                i+=1

    return label_counts_df


def sweep_dbscan(df,eps_vals,cluster_by):

    '''
    Convenience function to parameter sweep values of eps using dbscan
    while showing a progress bar.

        Input:
        perp_range: tuple (start, end, number)
    '''

    # Parameter sweep values of eps
    for eps in tqdm(eps_vals):
        _ = cluster_vis_sns(df,float(eps), cluster_by)

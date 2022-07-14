#cluster_visualization.py

from initialization.config import *
from visualization.low_dimension_visualization import *

import numpy as np
import pandas as pd
import os


# matplotlib imports
import matplotlib.pyplot as plt
from matplotlib import cm

plt.rcParams.update({
    "figure.facecolor":  (1.0, 1.0, 1.0, 1.0),
    "axes.facecolor":    (1.0, 1.0, 1.0, 1.0),
    "savefig.facecolor": (1.0, 1.0, 1.0, 1.),
    "figure.figsize":    (10,10),
    "font.size": 12
})

import sklearn.preprocessing
import sklearn.pipeline
import scipy.spatial

# DBscan functon for parameter sweeping Eps
def cluster_vis_sns(df_in,eps, cluster_by,min_samples=10):

    '''
    Visualize labels from DBScan in a scatterplot.

    Intended to work with positional or dimension-reduced input data.

    Input:
        df_in: dataframe containing only the columns to be clustered and plotted
        eps: value that influences the clustering.
        cluster_by: str: 'pos', 'tsne', 'pca'
    '''

    if cluster_by == 'pos':
        x_name = 'x'
        y_name = 'y'
    elif cluster_by == 'pca':
        x_name = 'PC1'
        y_name = 'PC2'
    elif cluster_by == 'tsne':
        x_name = 'tSNE1'
        y_name = 'tSNE2'


    #DBScan
    X = StandardScaler().fit_transform(df_in.values)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Assemble a dataframe from the results
    lab_df = pd.DataFrame(data = labels, columns = ['label'])
    dr_label_df = pd.concat([df_in,lab_df], axis=1)

    g = sns.jointplot(data=dr_label_df[dr_label_df['label']!=-1],x=x_name, y=y_name, hue="label",
                      kind='scatter',
                   palette=palette,
                      joint_kws={'alpha': 0.4,'s': 5}, height=10, legend=False)
    if STATIC_PLOTS:
        plt.savefig(CLUST_PARAMS_DIR+cluster_by+'_sweep_eps_'+str(eps)+'.png', dpi=300)

    if PLOTS_IN_BROWSER:
        plt.show()

    plt.clf() # Because it will be run in a loop.

    return plt


def calculate_hull(
        X,
        scale=1.0,
        padding="scale",
        n_interpolate=100,
        interpolation="quadratic_periodic",
        return_hull_points=False):

    """
    Calculates a "smooth" hull around given points in `X`.
    The different settings have different drawbacks but the given defaults work reasonably well.
    Parameters
    ----------
    X : np.ndarray
        2d-array with 2 columns and `n` rows
    scale : float, optional
        padding strength, by default 1.1
    padding : str, optional
        padding mode, by default "scale"
    n_interpolate : int, optional
        number of interpolation points, by default 100
    interpolation : str or callable(ix,iy,x), optional
        interpolation mode, by default "quadratic_periodic"

    From: https://stackoverflow.com/questions/17553035/draw-a-smooth-polygon-around-data-points-in-a-scatter-plot-in-matplotlib

    """

    if padding == "scale":

        # scaling based padding
        scaler = sklearn.pipeline.make_pipeline(
            sklearn.preprocessing.StandardScaler(with_std=False),
            sklearn.preprocessing.MinMaxScaler(feature_range=(-1,1)))
        points_scaled = scaler.fit_transform(X) * scale
        hull_scaled = scipy.spatial.ConvexHull(points_scaled, incremental=True)
        hull_points_scaled = points_scaled[hull_scaled.vertices]
        hull_points = scaler.inverse_transform(hull_points_scaled)
        hull_points = np.concatenate([hull_points, hull_points[:1]])

    elif padding == "extend" or isinstance(padding, (float, int)):
        # extension based padding
        # TODO: remove?
        if padding == "extend":
            add = (scale - 1) * np.max([
                X[:,0].max() - X[:,0].min(),
                X[:,1].max() - X[:,1].min()])
        else:
            add = padding
        points_added = np.concatenate([
            X + [0,add],
            X - [0,add],
            X + [add, 0],
            X - [add, 0]])
        hull = scipy.spatial.ConvexHull(points_added)
        hull_points = points_added[hull.vertices]
        hull_points = np.concatenate([hull_points, hull_points[:1]])
    else:
        raise ValueError(f"Unknown padding mode: {padding}")

    # number of interpolated points
    nt = np.linspace(0, 1, n_interpolate)

    x, y = hull_points[:,0], hull_points[:,1]

    # ensures the same spacing of points between all hull points
    t = np.zeros(x.shape)
    t[1:] = np.sqrt((x[1:] - x[:-1])**2 + (y[1:] - y[:-1])**2)
    t = np.cumsum(t)
    t /= t[-1]

    # interpolation types
    if interpolation is None or interpolation == "linear":
        x2 = scipy.interpolate.interp1d(t, x, kind="linear")(nt)
        y2 = scipy.interpolate.interp1d(t, y, kind="linear")(nt)
    elif interpolation == "quadratic":
        x2 = scipy.interpolate.interp1d(t, x, kind="quadratic")(nt)
        y2 = scipy.interpolate.interp1d(t, y, kind="quadratic")(nt)

    elif interpolation == "quadratic_periodic":
        x2 = scipy.interpolate.splev(nt, scipy.interpolate.splrep(t, x, per=True, k=4))
        y2 = scipy.interpolate.splev(nt, scipy.interpolate.splrep(t, y, per=True, k=4))

    elif interpolation == "cubic":
        x2 = scipy.interpolate.CubicSpline(t, x, bc_type="periodic")(nt)
        y2 = scipy.interpolate.CubicSpline(t, y, bc_type="periodic")(nt)
    else:
        x2 = interpolation(t, x, nt)
        y2 = interpolation(t, y, nt)

    X_hull = np.concatenate([x2.reshape(-1,1), y2.reshape(-1,1)], axis=1)
    if return_hull_points:
        return X_hull, hull_points
    else:
        return X_hull





def draw_cluster_hulls(df_in, cluster_by=CLUSTER_BY, min_pts=5, color_by='cluster',cluster_label='label',ax=None,draw_pts=False,save_path=CLUST_DIR, legend=False):

    df = df_in.copy()

    if cluster_by == 'xy':
        x_name = 'x'
        y_name = 'y'

    elif cluster_by == 'pca':
        x_name = 'PC1'
        y_name = 'PC2'

    elif (cluster_by == 'tsne' or cluster_by == 'tSNE'):

        x_name = 'tSNE1'
        y_name = 'tSNE2'

    elif (cluster_by == 'umap' or cluster_by == 'UMAP'):

        x_name = 'UMAP1'
        y_name = 'UMAP2'

    labels = list(set(df[cluster_label].unique()))

    # conditions = df['Cond_label'].unique()
    # colors = np.asarray(sns.color_palette(PALETTE, n_colors=len(conditions)))
    # print(conditions, colors)

    catcol = 'Cond_label'
    categories = np.unique(df[catcol])
    colors = np.linspace(0, 1, len(categories))
    colordict = dict(zip(categories, colors))
    df["Color"] = df[catcol].apply(lambda x: colordict[x])

    # Define a list of cluster colors to be consistent across the module

    cluster_colors = []
    cmap = cm.get_cmap(CLUSTER_CMAP, len(labels))
    for i in range(cmap.N):
        cluster_colors.append(cmap(i))


    # If no axis is supplied, then createa simple fig, ax and default to drawing the points.
    if ax is None:

        fig, ax = plt.subplots()
        ax.set_xlabel(x_name)
        ax.set_ylabel(y_name)

        all_pts = df[[x_name, y_name]].values
        draw_pts = True
        ax.scatter(x=all_pts[:,0], y=all_pts[:,1], s=0.1, c='grey')

    if(color_by=='condition' and draw_pts):
        # Draw the scatter points for the current cluster
        scatter = ax.scatter(x=df[x_name], y=df[y_name], s=0.3, c=df['Color'],cmap=CONDITION_CMAP)

        ''' Note: for some reason we cant get the conditions labels correctly on the legend.'''
        legend1 = ax.legend(*scatter.legend_elements(),#labels=list(df['Condition_shortlabel'].unique()),
                            loc="upper right", title="Condition")
        ax.add_artist(legend1)

    elif(color_by=='PCs' and draw_pts):
        pc_colors = colormap_pcs(df, cmap='rgb')
        # pc_colors = np.asarray(df[['PC1','PC2','PC3']])
        # scaler = MinMaxScaler()
        # scaler.fit(pc_colors)
        # pc_colors= scaler.transform(pc_colors)

        # Color code the scatter points by their respective PC 1-3 values
        ax.scatter(x=df[x_name], y=df[y_name], s=0.5, c=pc_colors)

    for i_lab,curr_label in enumerate(labels[:-2]):


        curr_lab_df = df[df[cluster_label] == curr_label]
        curr_lab_pts = curr_lab_df[[x_name, y_name]].values

        if curr_lab_pts.shape[0] > min_pts:

            x=curr_lab_pts[:,0]
            y=curr_lab_pts[:,1]

            if(color_by=='cluster' and draw_pts):
                '''
                Having this color_by block within the cluster loop makes sure that the
                points within the cluster have the same default color order as the cluster hulls
                drawn below in the same loop.

                Ideally it would be possible to apply a custom colormap to the clusters, apply them to
                the scatter plots, and use them in other plots to make the link.
                '''

                # Draw the scatter points for the current cluster
                ax.scatter(x=x, y=y, s=0.3, color=cluster_colors[i_lab])

            # Catch cases where we can't draw the hull because the interpolation fails.
            try:
                X_hull  = calculate_hull(
                    curr_lab_pts,
                    scale=1.0,
                    padding="scale",
                    n_interpolate=100,
                    interpolation="quadratic_periodic")

                ax.plot(X_hull[:,0], X_hull[:,1],c=cluster_colors[i_lab], label=curr_label)
                if(legend):
                    ax.legend()
            except:
                print('Failed to draw cluster, failed to draw cluster: ',curr_label, ' with shape: ', curr_lab_pts.shape)


    if STATIC_PLOTS:
        plt.savefig(save_path+'clusterhull_scatter_'+cluster_by+'.png')
    if PLOTS_IN_BROWSER:
        plt.show()


    return ax




def purity_plots(lab_dr_df, clust_sum_df,traj_clust_df,trajclust_sum_df,cluster_by=CLUSTER_BY):


    if(cluster_by == 'tsne' or cluster_by == 'tSNE'):

        x_name = 'tSNE1'
        y_name = 'tSNE2'


    elif cluster_by == 'umap':

        x_name = 'UMAP1'
        y_name = 'UMAP2'




    # Create a Subplot figure that shows the effect of clustering between conditions

    fig, ((ax1, ax2),(ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=[20,20])

    fig.suptitle("Low dimensional cluster analysis and purity", fontsize='x-large')

    #
    # First subplot: Scatter of lowD space with cluster outlines
    #

    ax1.set_xlabel(x_name)
    ax1.set_ylabel(y_name)
    ax1.set_title('Low-dimensional scatter with cluster outlines', fontsize=18)
    # ax1.scatter(x=lab_dr_df['tSNE1'], y=lab_dr_df['tSNE2'], c='grey', s=0.1)
    draw_cluster_hulls(lab_dr_df,cluster_by=cluster_by, color_by='condition',ax=ax1, draw_pts=True,legend=True)

    colors=[]
    cmap = cm.get_cmap(CONDITION_CMAP, len(lab_dr_df['Condition_shortlabel'].unique()))
    for i in range(cmap.N):
        colors.append(cmap(i))

    # Define a custom colormap for the clusters (To match what's done in draw_cluster_hulls())
    cluster_colors = []
    labels = list(set(lab_dr_df['label'].unique()))
    cmap = cm.get_cmap(CLUSTER_CMAP, len(labels))
    for i in range(cmap.N):
        cluster_colors.append(cmap(i))



    #
    # Second subplot: stacked bar plots of cluster purity
    #
    ax2.set_title('Low-dimension cluster purity', fontsize=18)

    for i, cond in enumerate(lab_dr_df['Condition_shortlabel'].unique()):
        '''
        Assumes that the conditions are already in the correct order in the dataframe.
        '''
        if(i==0):
            ax2.bar(clust_sum_df['cluster_id'], clust_sum_df[cond+'_ncells_%'], label=cond,color=colors[i])
            prev_bottom = clust_sum_df[cond+'_ncells_%']
        else:
            ax2.bar(clust_sum_df['cluster_id'], clust_sum_df[cond+'_ncells_%'], bottom=prev_bottom, label=cond,color=colors[i])

    ax2.set_xticks(clust_sum_df['cluster_id'])
    ax2.set_ylabel('% of cells per cluster')
    ax2.legend()


    for ticklabel, tickcolor in zip(ax2.get_xticklabels(), cluster_colors):
        ticklabel.set_color(tickcolor)





    #
    # Thirds subplot: Trajectories through lowD space
    #

    # Define a custom colormap for the trajectory clusters (To match what's done in draw_cluster_hulls())
    cluster_colors = []
    labels = list(set(traj_clust_df['traj_id'].unique()))
    cmap = cm.get_cmap(CLUSTER_CMAP, len(labels))
    for i in range(cmap.N):
        cluster_colors.append(cmap(i))

    ax3.set_title('Cell trajectory clustering through low-dimensional space', fontsize=18)
    ax3.scatter(x=traj_clust_df['UMAP1'], y=traj_clust_df['UMAP2'], s=0.5,alpha=0.5,
            c='gray')
    for i, traj in enumerate(labels[:-2]): # Same as when drawing contours
        traj_sub_df = traj_clust_df[traj_clust_df['traj_id'] == traj]
#         display(traj_sub_df)
        # Draw lines and scatters individually for each label
        ax3.plot(traj_sub_df[x_name], traj_sub_df[y_name], alpha=1, c=cluster_colors[i],linewidth=0.1)
        ax3.scatter(x=traj_sub_df[x_name], y=traj_sub_df[y_name], s=0.8,alpha=0.5,
            color=cluster_colors[i])

    '''Cannot colormap lines explicitly with matplotlib.
        If you need to the lines colormapped, they would have to be iterated and drawn with discrete colors.
    '''
#     ax3.plot(traj_clust_df['UMAP1'], traj_clust_df['UMAP2'], alpha=0.5, c='gray',linewidth=0.1)




    ax3.set_xlabel(x_name)
    ax3.set_ylabel(y_name)
    # Alt: Colormap by cluster, using custom map. This will include the unlabeled ones
#     ax3.scatter(x=traj_clust_df['UMAP1'], y=traj_clust_df['UMAP2'], s=1,alpha=0.9,
#                 c=traj_clust_df['traj_id'], cmap= CLUSTER_CMAP)

    #
    # ALT: Using datashader for 3rd plot.
    #
    # traj_clust_df = trajectory_clustering_pipeline(std_dr_df, traj_factor='umap', dist_metric='hausdorff', filename_out='std_dr_df_traj_')
    # datashader_lines(traj_clust_df, 'UMAP1', 'UMAP2',color_by='traj_id', categorical=True)
    # ax3.set_facecolor('#000000')

    # cvs = ds.Canvas()#,x_range=x_range, y_range=y_range)
    # artist = dsshow(cvs.line(traj_clust_df,'UMAP1', 'UMAP2', agg=ds.count_cat('Cat')),ax=ax3)

    # Datashader for 3rd panel (points only working, not lines)
    # ax3.set_facecolor('#000000')
    # traj_clust_df['Cat'] = traj_clust_df['Condition'].astype('category')
    # artist = dsshow(
    #     traj_clust_df,
    #     ds.Point('UMAP1', 'UMAP2'),
    #     ds.count_cat('Cat'),
    # #     norm='log',
    #     ax=ax3
    # )
    #


    draw_cluster_hulls(traj_clust_df,cluster_by=cluster_by, cluster_label='traj_id', ax=ax3, color_by='cluster',legend=True)

    #
    # Fourth subplot: Trajectories cluster purity
    #

    ax4.set_title('Cell trajectory cluster purity', fontsize=18)

    for i, cond in enumerate(lab_dr_df['Condition_shortlabel'].unique()):
        '''
        Assumes that the conditions are already in the correct order in the dataframe.
        '''
        if(i==0):
            ax4.bar(trajclust_sum_df['cluster_id'], trajclust_sum_df[cond+'_ncells_%'], label=cond,color=colors[i])
            prev_bottom = trajclust_sum_df[cond+'_ncells_%']
        else:
            ax4.bar(trajclust_sum_df['cluster_id'], trajclust_sum_df[cond+'_ncells_%'], bottom=prev_bottom, label=cond,color=colors[i])

    ax4.set_xticks(trajclust_sum_df['cluster_id'])
    ax4.set_ylabel('Number of cells per trajectory cluster')
    ax4.legend()

    for ticklabel, tickcolor in zip(ax4.get_xticklabels(), cluster_colors):
        ticklabel.set_color(tickcolor)

    return fig

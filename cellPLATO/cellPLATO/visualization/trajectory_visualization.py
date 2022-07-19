from initialization.config import *
from initialization.initialization import *

from data_processing.shape_calculations import *

from visualization.cluster_visualization import *
from visualization.low_dimension_visualization import colormap_pcs

import numpy as np
import pandas as pd
import os
import imageio

import matplotlib.pyplot as plt
from matplotlib import cm

plt.rcParams.update({
    "figure.facecolor":  (1.0, 1.0, 1.0, 1.0),
    "axes.facecolor":    (1.0, 1.0, 1.0, 1.0),
    "savefig.facecolor": (1.0, 1.0, 1.0, 1.),
    "figure.figsize":    (10,10),
    "font.size": 12
})

def plot_cell_metrics(cell_df, i_step):
    '''
    For a selected cell, at a selected step of its trajectory, make a plot

    '''
    # Using the same sample cell trajectory, print out some measurements
    fig, (ax) = plt.subplots(1,1)
    ax.title.set_text('Cell contour: xy space')
    ax.plot(cell_df['x_pix']*2,cell_df['y_pix']*2,'-o',markersize=3,c='black')


    contour_list = get_cell_contours(cell_df)

    # Draw all contours faintly as image BG
    for i,contour in enumerate(contour_list):

        rgb = np.random.rand(3,)
        this_colour = rgb#'red' # Eventually calculate color along colormap
        contour_arr = np.asarray(contour).T


        if not np.isnan(np.sum(contour_arr)): # If the contour is not nan

            x = cell_df['x_pix'].values[i]# - window / 2
            y = cell_df['y_pix'].values[i]# - window / 2

            # Cell contour relative to x,y positions
            '''Want to double check that the x,y positions not mirrored from the contour function'''
            ax.plot(x+contour_arr[:,0],y+contour_arr[:,1],'-o',markersize=1,c='gray', alpha=0.2)

    # Draw this contour, highlighted.
    contour = contour_list[i_step]
    contour_arr = np.asarray(contour).T
    x = cell_df['x_pix'].values[i_step]# - window / 2
    y = cell_df['y_pix'].values[i_step]# - window / 2

    # Cell contour relative to x,y positions
    '''Want to double check that the x,y positions not mirrored from the contour function'''
    if not np.isnan(np.sum(contour_arr)):
        ax.plot(x+contour_arr[:,0],y+contour_arr[:,1],'-o',markersize=1,c='tab:orange',linewidth=5, alpha=1)

        # Current segment
        if i_step > 0:
            x_seg = cell_df['x_pix'].values[i_step-1:i_step+1]# - window / 2
            y_seg = cell_df['y_pix'].values[i_step-1:i_step+1]# - window / 2
            # ax.plot(x+contour_arr[:,0],y+contour_arr[:,1],'-o',markersize=1,c='red', alpha=1)
            ax.plot(x_seg*2,y_seg*2,'-o',markersize=10,c='tab:blue', linewidth=4)

    text_x = x*2 # Still a magic number, replace with calibration value ??
    text_y = y*2 # Still a magic number, replace with calibration value ??

    for n, fac in enumerate(MIG_DISPLAY_FACTORS):

        ax.text(text_x + 10,text_y +2 +  n, fac +': '+ format(cell_df.iloc[i_step][fac], '.1f'),
                color='tab:blue', fontsize=10)

    for n, fac in enumerate(SHAPE_DISPLAY_FACTORS):

        ax.text(text_x + 10, text_y - 2 - n, fac +': '+ format(cell_df.iloc[i_step][fac], '.1f'),
                color='tab:orange', fontsize=10)

    return fig



def plot_cell_trajectories(cell_df, dr_df, dr_method='tSNE',contour_scale=1/2,cmap_by=None):#'area'):

    '''
    Compare cell trajectories through physical (xy) and Dimension Reduced (dr) space.

    Input:
        cell_df: DataFrame containing the cell to be plotted.
        dr_df: dimension reduced dataframe (for plot context, grayed out - full data set.)
        dr_method: tSNE, PCA, umap
        contour_scale: factor to scale the contours when plotting them in DR space
        cmap_by: factor to colormap the tSNE scatter. Default: area.
                Set to 'label' if lab_dr_df supplied as input

TO TEST: Does this handle multiple ones?? Maybe it should...
INCOMPLETE: Still need to colormap the trajectory segments AND contours by time.




    '''

    if(dr_method == 'tsne' or dr_method == 'tSNE'):

            x_lab = 'tSNE1'
            y_lab = 'tSNE2'

    elif dr_method == 'PCA':

            x_lab = 'PC1'
            y_lab = 'PC2'

    elif dr_method == 'umap':

            x_lab = 'UMAP1'
            y_lab = 'UMAP2'

    contour_list = get_cell_contours(cell_df)
    assert len(contour_list) == len(cell_df.index), 'Contours doesnt match input data'

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10,10))

    ax1.title.set_text('Cell trajectory: xy space')
    ax2.title.set_text('Cell contour: xy space')
    ax3.title.set_text('Cell trajectory: ' + dr_method + ' space')
    ax4.title.set_text('Cell contour:' + dr_method + ' space')

    ax1.set_aspect('equal')
    ax1.set_adjustable("datalim")
    ax2.set_aspect('equal')
    ax2.set_adjustable("datalim")

    # Plots that don't need to be redrawn for each contour
    ax1.scatter(x=dr_df['x'],y=dr_df['y'],c=dr_df['rip_L'], alpha=0.5, s=10) # 'gray'
    # ax2.scatter(x=dr_df['x'],y=dr_df['y'],c=dr_df['rip_L'], alpha=0.5, s=10) #'gray'

    '''Consider scaling the scatter window to a function of cell diameters.'''
    scat_wind = 100

    ax1.set_xlim(cell_df['x_pix'].values[0] - scat_wind/2, cell_df['x_pix'].values[0] + scat_wind/2)
    ax1.set_ylim(cell_df['y_pix'].values[0] - scat_wind/2, cell_df['y_pix'].values[0] + scat_wind/2)

    ax2.plot(cell_df['x_pix']*2,cell_df['y_pix']*2,'-o',markersize=3,c='gray')
    '''
    Why does 2* work for scaling???
    '''
    # Calculate rgb values of colormap according to PCs 1-3.
    pc_colors = colormap_pcs(dr_df, cmap='rgb')
    # pc_colors = np.asarray(dr_df[['PC1','PC2','PC3']])
    # scaler = MinMaxScaler()
    # scaler.fit(pc_colors)
    # pc_colors= scaler.transform(pc_colors)

    # ax3.scatter(x=dr_df[x_lab],y=dr_df[y_lab],c=dr_df[cmap_by], alpha=0.1, s=2)
    ax3.scatter(x=dr_df[x_lab],y=dr_df[y_lab],c=pc_colors, alpha=0.1, s=2)

    if 'label' in dr_df.columns:

        print('Label included, drawing cluster hulls.')
        draw_cluster_hulls(dr_df,cluster_by=dr_method,ax=ax3)

    ax3.plot(cell_df[x_lab],cell_df[y_lab],'-o',markersize=3,c='red')

    ax4.scatter(x=dr_df[x_lab],y=dr_df[y_lab],c='gray', alpha=0.3, s=1)

    assert len(cell_df['x']) == len(contour_list)

    # Colormap the contours
    PALETTE = 'flare'
    colors = np.asarray(sns.color_palette(PALETTE, n_colors=len(contour_list)))
    cm_data = np.asarray(colors)

    for i,contour in enumerate(contour_list):

        rgb=cm_data[i,:]

        this_colour = rgb#'red' # Eventually calculate color along colormap

        if i > 0:
            # segment of the trajectory
            this_seg = cell_df[['x_pix','y_pix']].values[i-1:i+1]
            ax1.plot(this_seg[:,0],this_seg[:,1],'-o',markersize=3,c=this_colour)

            # segment of the DR trajectory
            dr_seg = cell_df[[x_lab,y_lab]].values[i-1:i+1]
            ax3.plot(dr_seg[:,0],dr_seg[:,1],'-o',markersize=3,c=this_colour)

        contour_arr = np.asarray(contour).T


        x = cell_df['x_pix'].values[i]# - window / 2
        y = cell_df['y_pix'].values[i]# - window / 2

        # Cel contour change relative to centroid (less relevant)
        if not np.isnan(np.sum(contour_arr)):
            ax2.plot(x+contour_arr[:,0],y+contour_arr[:,1],'-o',markersize=1,c=this_colour)

        # Need to cell contours to be centered on the cells position within the image
        x_dr = cell_df[x_lab].values[i] - x# - window / 2
        y_dr = cell_df[y_lab].values[i] - y# - window / 2

        # Cell contour relative to tSNE positions
        if not np.isnan(np.sum(contour_arr)):
            ax4.plot(x_dr+contour_arr[:,0],y_dr+contour_arr[:,1],'-o',markersize=1,c=this_colour)


def visualize_cell_t_window(cell_df, dr_df, t, t_window, contour_list=None, dr_method='tSNE', cid='test', contour_scale=1/5,cmap_by = 'area'):

    '''

    NOTE; THis function still uses a magic factor of 2 when drawing the trajectories relative to the contours.
    So, I wouldn't trust the absolute axis values for the cell location,

    TO DO: find unique identifier for this cell.


    To better understand the measurements being included for each cell at each timepoint,.
    A plot to represent the cell shape through space and time, that is representative
    of the measures included in the time window.

    Namely, the path centered at that point.
    The cell contour at that timepoint, but also faint representations of the othrer timepoints
    that are being averaged across.

    '''

    df = cell_df.copy()

    if(dr_method == 'tsne' or dr_method == 'tSNE'):

            x_lab = 'tSNE1'
            y_lab = 'tSNE2'

    elif dr_method == 'PCA':

            x_lab = 'PC1'
            y_lab = 'PC2'

    elif dr_method == 'umap':

            x_lab = 'UMAP1'
            y_lab = 'UMAP2'

    fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(10,5))

    # ax1: xy space
    ax1.title.set_text('Cell contour + trajectory in time window: xy space' )
    ax2.title.set_text('Cell contour + trajectory: ' + dr_method + ' space')

    ax1.set_aspect('equal')
    ax1.set_adjustable("datalim")

    scale_factor = 2

    scat_wind = int(60 / MICRONS_PER_PIXEL)#100
    min_x, max_x = (np.min(cell_df['x_pix']),np.max(cell_df['x_pix']))
    min_y, max_y = (np.min(cell_df['y_pix']),np.max(cell_df['y_pix']))

    ax1.set_xlim(min_x* scale_factor-scat_wind/2, max_x* scale_factor+scat_wind/2)
    ax1.set_ylim(min_y* scale_factor-scat_wind/2, max_y* scale_factor+scat_wind/2)

    # Add select metrics as text.
    text_x = min_x * scale_factor-scat_wind/1.6 # Still a magic number, replace with calibration value ??
    text_y = np.mean(cell_df['y_pix']) * scale_factor#+scat_wind/2# Still a magic number, replace with calibration value ??


    # ax2: Low-dimensional space
    ax2.set_xlim(np.min(dr_df[x_lab])-abs(np.min(dr_df[x_lab])/2),
                 np.max(dr_df[x_lab])+ abs(np.max(dr_df[x_lab])/2))
    ax2.set_ylim(np.min(dr_df[y_lab])-abs(np.min(dr_df[y_lab])/2),
                 np.max(dr_df[y_lab]) + abs(np.max(dr_df[y_lab])/2))

    ax2.set_aspect('equal')
    # ax2.set_adjustable("datalim")

    if contour_list is None: # Allow animate_t_window() to pass the contours instead of reloading
        contour_list = get_cell_contours(cell_df)

    assert len(contour_list) == len(cell_df.index), 'Contours doesnt match input data'

    frames = cell_df['frame']
    frame_i = int(t * (len(cell_df['frame'])-1))
    frame = cell_df.iloc[frame_i]['frame']

    identifier = str(cid) +'_'+ dr_method +'_t_'+str(t)
    fig.suptitle('Cell '+str(cell_df['uniq_id'].values[0])+' behaviour at ' +
            str(format(frame * SAMPLING_INTERVAL, '.2f')) + ' mins', fontsize=12)
    # Time-windowed dataframe
    if t_window is not None:

        # get a subset of the dataframe across the range of frames
        cell_t_df = cell_df[(cell_df['frame']>=frame - t_window/2) &
                      (cell_df['frame']<frame + t_window/2)]
    else:

        cell_t_df = cell_df.copy()

    # Colormap the contours (to the time window)
    PALETTE = 'flare'
    colors = np.asarray(sns.color_palette(PALETTE, n_colors=t_window))
    cm_data = np.asarray(colors)

    # Get these contours and the track.
    '''Should be able to index the contour list generated above using the frame_i'''

    this_contour = contour_list[frame_i]
    t_window_contours = contour_list[int(frame_i-t_window/2):int(frame_i+t_window/2)]

    # Plot the spatial trajectory.
    ax1.plot(cell_t_df['x_pix']*scale_factor,cell_t_df['y_pix']*scale_factor,'-o',markersize=3,c='gray')


    # Plot the dimension reduction plot
    # ax2.scatter(x=dr_df[x_lab],y=dr_df[y_lab],c=dr_df[cmap_by], alpha=0.1, s=2)
    pc_colors = colormap_pcs(dr_df, cmap='rgb')
    ax2.scatter(x=dr_df[x_lab],y=dr_df[y_lab],c=pc_colors, alpha=0.3, s=1)
    ax2.plot(cell_t_df[x_lab],cell_t_df[y_lab],'-o',markersize=2,c='black')

    # Plot the contours (All indexing relative to time_window.)
    for i,contour in enumerate(t_window_contours):

        rgb=cm_data[i,:]

        this_colour = rgb#'red' # Eventually calculate color along colormap

        contour_arr = np.asarray(contour).T# * contour_scale

        # Get the x,y pixel position of the cell at this frame
        x = cell_t_df['x_pix'].values[i]# - window / 2
        y = cell_t_df['y_pix'].values[i]# - window / 2

        # Need to cell contours to be centered on the cells position within the image
        x_dr = cell_t_df[x_lab].values[i] - x# - window / 2
        y_dr = cell_t_df[y_lab].values[i] - y# - window / 2


#         # Translate and scale the contour. (NOT YET WORKING CORRECTLY)
#         rel_contour_arr = contour_arr.copy()
#         rel_contour_arr[:,0] = contour_arr[:,0] - x_dr
#         rel_contour_arr[:,1] = contour_arr[:,1] - y_dr
#         rel_contour_arr = rel_contour_arr * contour_scale

        twind_traj = cell_t_df[['x_pix','y_pix']].values


        # Cell contour change relative to centroid (less relevant)
        if not np.isnan(np.sum(contour_arr)):

            if(i == int(t_window/ 2)): # If center of the t_window, this frame

                # XY-space visualization

                ax1.plot(x+contour_arr[:,0],y+contour_arr[:,1],'-o',markersize=5,linewidth=5,c=this_colour, alpha=1)
                this_seg = cell_t_df[['x_pix','y_pix']].values[i-1:i+1]
                ax1.plot(this_seg[:,0]*scale_factor,this_seg[:,1]*scale_factor,'-o',markersize=5,linewidth=5,c=this_colour)

#                 # Add select metrics as text (moving with the cell)
#                 text_x = x*scale_factor # Still a magic number, replace with calibration value ??
#                 text_y = y*scale_factor # Still a magic number, replace with calibration value ??

                for n, fac in enumerate(MIG_DISPLAY_FACTORS):

                    ax1.text(text_x + 15,text_y +5 *  (n+1), fac +': '+ format(cell_t_df.iloc[i][fac], '.1f'),
                            color='tab:blue', fontsize=10)

                for n, fac in enumerate(SHAPE_DISPLAY_FACTORS):

                    ax1.text(text_x + 15, text_y - 5 * (n+1), fac +': '+ format(cell_t_df.iloc[i][fac], '.1f'),
                            color='tab:orange', fontsize=10)


                # Low-D space visualization

                # Draw a brief trajectory through DR space
                ax2.scatter(x=cell_t_df[x_lab].values[i],y=cell_t_df[y_lab].values[i],color=this_colour, alpha=1, s=50)
                # Also draw this contour directly onto the DR plot.
                ax2.plot(x_dr+contour_arr[:,0],y_dr+contour_arr[:,1],'-o',markersize=1,c=this_colour)
                # Draw the trajectory segment for the time window included in the current data

                ax2.plot((x_dr+twind_traj[:,0]),
                         (y_dr+twind_traj[:,1]),'-o',markersize=3,c=this_colour,linewidth=1)
                ax2.plot((x_dr+this_seg[:,0]),
                         (y_dr+this_seg[:,1]),'-o',markersize=3,c=this_colour,linewidth=3)


            else:
                ax1.plot(x+contour_arr[:,0],y+contour_arr[:,1],'-o',markersize=1,c=this_colour, alpha=0.5)

    return fig

def plot_traj_cluster_avg(traj_list, cluster_list, label):
    '''
    Plots given trajectories with a color that is specific for every trajectory's own cluster index.
    Outlier trajectories which are specified with -1 in `cluster_lst` are plotted dashed with black color
    '''

    color_lst = plt.rcParams['axes.prop_cycle'].by_key()['color']
    color_lst.extend(['firebrick', 'olive', 'indigo', 'khaki', 'teal', 'saddlebrown',
                 'skyblue', 'coral', 'darkorange', 'lime', 'darkorchid', 'dimgray'])

    cluster_count = np.max(cluster_list) + 1

    for cluster_id in np.unique(cluster_list):


        clust_inds = np.argwhere(cluster_list==cluster_id)
        clust_inds = np.squeeze(clust_inds).astype(int)

        this_traj_list = []

        for ind in clust_inds:
            this_traj = traj_list[ind]
            this_traj_list.append(this_traj)

#         # Plot the trajectories without any averging(if not the same length)
#         for cluster_traj in this_traj_list:
#             plt.plot(cluster_traj[:,0], cluster_traj[:,1], c=color_lst[cluster_id % len(color_lst)], alpha=0.1)

        # Find the average along the trajectory or segment (must be same length)

        cluster_traj = np.asarray(this_traj_list)
        clust_avg_traj = np.mean(cluster_traj, axis=0)

        if cluster_id == -1:
            # Means it it a noisy trajectory, paint it black
            plt.plot(cluster_traj[:, :,0], cluster_traj[:, :,1], c='k', linestyle='dashed', alpha=0.01)

        else:
            plt.plot(cluster_traj[:, :,0], cluster_traj[:, :,1], c=color_lst[cluster_id % len(color_lst)], alpha=0.1)
            plt.plot(clust_avg_traj[:, 0], clust_avg_traj[:, 1], c=color_lst[cluster_id % len(color_lst)], alpha=1,mec='black')



    if STATIC_PLOTS:

        plt.savefig(CLUST_DIR+'average_trajectories.png', dpi=300)


def trajectory_cluster_vis(traj_clust_df,traj_factor, scatter=False):

    if (traj_factor=='tSNE' or traj_factor=='tsne'):
        if scatter:
            traj_clust_df.plot.scatter(x='tSNE1',y='tSNE2',c='traj_id',colormap='viridis')
        x_label = 'tSNE1'
        y_label = 'tSNE2'

    elif (traj_factor=='UMAP' or traj_factor=='umap'):
        if scatter:
            traj_clust_df.plot.scatter(x='UMAP1',y='UMAP2',c='traj_id',colormap='viridis')
        x_label = 'UMAP1'
        y_label = 'UMAP2'

    # Define a custom colormap for the trajectory clusters (To match what's done in draw_cluster_hulls())
    cluster_colors = []
    labels = list(set(traj_clust_df['traj_id'].unique()))
    cmap = cm.get_cmap(CLUSTER_CMAP, len(labels))
    for i in range(cmap.N):
        cluster_colors.append(cmap(i))

    fig, ax = plt.subplots()
    ax.set_title('Cell trajectory clustering through low-dimensional space', fontsize=18)
    ax.scatter(x=traj_clust_df[x_label], y=traj_clust_df[y_label], s=0.5,alpha=0.5,
            c='gray')

    # labels = list(set(traj_clust_df['traj_id'].unique())) # Set helps return a unioque ordeed list.

    for i, traj in enumerate(labels[:-1]): # Same as when drawing contours
        traj_sub_df = traj_clust_df[traj_clust_df['traj_id'] == traj]

        # Draw lines and scatters individually for each label
        ax.plot(traj_sub_df[x_label], traj_sub_df[y_label], alpha=0.2, c=cluster_colors[i],linewidth=0.5)
        ax.scatter(x=traj_sub_df[x_label], y=traj_sub_df[y_label], s=0.8,alpha=0.5,
            color=cluster_colors[i])

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    draw_cluster_hulls(traj_clust_df,cluster_by=traj_factor, cluster_label='traj_id', ax=ax, color_by='cluster',legend=True)

    if STATIC_PLOTS:

        plt.savefig(CLUST_DIR + 'trajectory_clusters.png', format='png', dpi=600)

    # return fig

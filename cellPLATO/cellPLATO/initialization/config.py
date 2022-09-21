#config.py

'''
Experiment-specific constants
'''


INPUT_FMT = 'btrack' # 'usiigaci'#btrack
MICRONS_PER_PIXEL = 0.537
TRACK_FILENAME = '.h5'
Z_SCALE = 1.00
CALIBRATED_POS = False


##################
# Small test set of Btracker data
#################

DATA_PATH = 'D://Michael_Shannon/MartinezMaster/cellplato_analysis_20x_30mins_10secperframe_CD18_Fc_PLL_only/'
CTL_LABEL = 'Condition_WT_PLL_20x'
CONDITIONS_TO_INCLUDE = ['Condition_WT_PLL_20x', 'Condition_WT_FcCtrl_20x','Condition_WT_antiCD18_20x',
'Condition_KO_PLL_20x', 'Condition_KO_FcCtrl_20x','Condition_KO_antiCD18_20x']

CONDITION_SHORTLABELS = ['WT PLL', 'WT FcCtrl', 'WT antiCD18',
'KO PLL', 'KO FcCtrl', 'KO antiCD18']

USE_SHORTLABELS = True
PERFORM_RIPLEYS = True
DATASET_SHORTNAME = 'MartinezMaster_paperdata20x_30mins_10secondstimepoints_CD18_Fc_PLL_only_9_13_2022'
# ATASET_SHORTNAME = 'CellPlatoFigure_20x100x_July20_mod1'
# ATASET_SHORTNAME = 'CellPlatoFigure_20x100x_July20_mod2'
# ATASET_SHORTNAME = 'CellPlatoFigure_20x100x_July20_mod3'

SAMPLING_INTERVAL = 10/60 # time between frames in minutes
IMAGE_HEIGHT = 1024 # pixels
IMAGE_WIDTH = 1024 # pixels
OVERWRITE = False # Overwrite the pre-processed data.

USE_INPUT_REGIONPROPS = True
CALCULATE_REGIONPROPS = False


MICRONS_PER_PIXEL_LIST = [0.537,0.537,0.537,0.537,0.537,0.537]
MICRONS_PER_PIXEL = MICRONS_PER_PIXEL_LIST[0] # Default value
SAMPLING_INTERVAL_LIST= [10/60,10/60,10/60,10/60,10/60,10/60]#[1,1, 1/60,1/60]
SAMPLING_INTERVAL = SAMPLING_INTERVAL_LIST[0] # Default value

# Timecourse analysis parameters
FRAME_START = 0 # Start frame for analysis
FRAME_END = 180 # End frame for analysis

MIXED_SCALING = False # Not used yet, for futureproofing
SELF_STANDARDIZE = False
FACTOR_TIMEAVERAGE = False

# '''
# Ultimate, high temporal resolution dataset
# '''
#
# DATA_PATH = 'Z://Collaboration_data/Mace_Lab/ultimate_high_temp_res'
# CTL_LABEL = 'Condition_ICAM_nochemokine_day2_cp_masks'
# CONDITIONS_TO_INCLUDE = ['untreated_day1', 'untreated_day2','cytoD40uM_day1','cytoD40uM_day2']
# CONDITION_SHORTLABELS = ['ctl_day1', 'ctl_day2', 'cytoD_day1', 'cytoD_day2']
# USE_SHORTLABELS = True
#
# DATASET_SHORTNAME = 'ultimate_high_temp_res'
# USE_INPUT_REGIONPROPS = True
#
# # SAMPLING_INTERVAL = 1/60 # time between frames
# IMAGE_HEIGHT = 1024 # pixels
# IMAGE_WIDTH = 1024 # pixels
# EPS=0.06
# USE_INPUT_REGIONPROPS = True
# # MICRONS_PER_PIXEL_LIST = [0.537,0.537, 0.537, 0.0.537]
# MICRONS_PER_PIXEL = 0.537
# SAMPLING_INTERVAL_LIST= [1/60,1/60, 1/60,1/60]#[1,1, 1/60,1/60]
# SAMPLING_INTERVAL = SAMPLING_INTERVAL_LIST[0] # Default value
#
# CALCULATE_REGIONPROPS = False
# OVERWRITE = False # Overwrite the pre-processed data.
#
# MIXED_SCALING = False # Not used yet, for futureproofing
# SELF_STANDARDIZE = False
# FACTOR_TIMEAVERAGE = False

############
# # Mixed 20x - 100x (fixed temporal resolution, changew in spatial resolution)
############
# DATA_PATH = 'Z://Collaboration_data/Mace_Lab/20x_100x/20x_100x_high_temp_res'
# CTL_LABEL = 'CAMWT_20x'
# CONDITIONS_TO_INCLUDE = ['CAMWT_20x', 'CAMKO_20x','NK92WT_100x', 'NK92KO_100x']
# CONDITION_SHORTLABELS = ['WT_20x', 'KO_20x','WT_100x', 'KO_100x']
# USE_SHORTLABELS = True
#
# DATASET_SHORTNAME = '20x_100x_hightr'
# USE_INPUT_REGIONPROPS = True
#
# # These ones happen to be consistent now, but that could easily change...
# IMAGE_HEIGHT = 1024 # pixels
# IMAGE_WIDTH = 1024 # pixels
#
# # Params to change between 20x and 100x
# MICRONS_PER_PIXEL = 0.107 # Known to be only half true ?????????
# MICRONS_PER_PIXEL_LIST = [0.537,0.537, 0.107, 0.107]
# SAMPLING_INTERVAL_LIST= [1/60,1/60, 1/60,1/60]#[1,1, 1/60,1/60]
# SAMPLING_INTERVAL = SAMPLING_INTERVAL_LIST[0] # Default value
#
# CALCULATE_REGIONPROPS = False
# OVERWRITE = True # Overwrite the pre-processed data.
#
# MIXED_SCALING = True
# SELF_STANDARDIZE = False
# FACTOR_TIMEAVERAGE = False






'''
Non-experiment specific constants
'''
MIG_T_WIND = 6 # in frames
# MIG_T_WIND = ?? * SAMPLING_INTERVAL ''' To convert into seconds'''

MIN_CELLS_PER_TPT = 1 # used in: average_per_timepoint()

OUTPUT_PATH = 'MartinezMaster_paperdata20x_30mins_10secondstimepoints_CD18_Fc_PLL_only_9_13_2022_OUTPUT/'

CLUSTER_CMAP = 'tab20'
CONDITION_CMAP = 'viridis'
CLUSTER_BY = 'umap' # TEMP - already in config

PALETTE = 'colorblind'
PX_COLORS = 'px.colors.qualitative.Safe' # Choose between discrete colors from https://plotly.com/python/discrete-color/
ARCHIVE_CONFIG = True
STATIC_PLOTS = True
PLOTS_IN_BROWSER = False

ANIMATE_TRAJECTORIES = True
DEBUG = False

#tSNE parameters and embedding:
TSNE_PERP = 185#230 # Perplexity
TSNE_R_S = 11 # Random seed
USE_SAVED_EMBEDDING = False#True
EMBEDDING_FILENAME = 'saved_embedding.npy'
TRAINX_FILENAME = 'saved_x_train.npy'

# UMAP parameters:
UMAP_NN = 10 # Nearest-neighbors
UMAP_MIN_DIST = 0.5

# DBScan
MIN_SAMPLES = 10
EPS = 0.06

# Factors to display on the animateed plots
MIG_DISPLAY_FACTORS=['speed', 'euclidean_dist']
SHAPE_DISPLAY_FACTORS = ['area','solidity']

STAT_TEST = 'st.ttest_ind'

# Plot display Parameters
PLOT_TEXT_SIZE = 16
DIFF_PLOT_TYPE = 'violin' # 'swarm', 'violin', 'box'


# Measurement constants
ARREST_THRESHOLD = 2 * SAMPLING_INTERVAL# microns - distance per step for arrest coefficient. Default 2 microns
RIP_R = 25 # Radius to search when calculating Ripleys K

# Factor to standardize to themselves over time (to look at self-relative instead of absolute values.)
FACTORS_TO_STANDARDIZE = ['area',
                          'bbox_area',
                          'equivalent_diameter',
                          'filled_area',
                          'major_axis_length',
                          'minor_axis_length',
                          'perimeter']


# Add fine control over certain plots axis limits, allows: '3-sigma','min-max', '2-sigma'
# Currently only implemented in marginal_xy contour plots.
AXES_LIMITS = '2-sigma' #'min-max' #'2-sigma'

FACTORS_TO_CONVERT = ['area', 'bbox_area', 'equivalent_diameter', 'extent', 'filled_area',
       'major_axis_length', 'minor_axis_length', 'perimeter']






# Cell migration factors calculated in migration_calcs()
MIG_FACTORS = ['euclidean_dist',     # Valid?
                'cumulative_length', # Valid?
                'speed',
                'orientedness', # name changed from orientation
                'directedness',
                'turn_angle',
                'endpoint_dir_ratio',
                'dir_autocorr',
                'outreach_ratio',
                'MSD',                # Valid?
                'max_dist',           # Valid?
                'glob_turn_deg',
                'arrest_coefficient']

# Region property factors to be extracted from the cell contours
# This list must match with props from regionprops
REGIONPROPS_LIST = ['area',
                    'bbox_area',
                    'eccentricity',
                    'equivalent_diameter',
                    'extent',
                    'filled_area',
                    'major_axis_length',
                    'minor_axis_length',
                    'orientation',
                    'perimeter',
                     'solidity']



# Additional factors calculated after the migration and segmentation
# ADDITIONAL_FACTORS = ['aspect']
ADDITIONAL_FACTORS = ['aspect', 'rip_p', 'rip_K', 'rip_L']

# Pre-defined pairs of factors for generating comparison plots
FACTOR_PAIRS = [['tSNE1', 'tSNE2'],
                ['area', 'speed'],
                ['directedness', 'speed'],
                ['orientedness', 'speed'],
                ['endpoint_dir_ratio', 'speed'],
                ['orientation', 'speed'],
                ['turn_angle', 'speed'], # These are identical
                ['major_axis_length', 'speed'],
                ['major_axis_length', 'minor_axis_length'],
                ['euclidean_dist','cumulative_length'],
                ['euclidean_dist','speed'],
                ['PC1', 'PC2']]

DR_FACTORS = REGIONPROPS_LIST + MIG_FACTORS + ADDITIONAL_FACTORS
# Numerical factors for plotting.
NUM_FACTORS = DR_FACTORS + ['tSNE1', 'tSNE2', 'PC1', 'PC2']

# Optionally define your data filters here.
DATA_FILTERS = {
  "area": (10, 10000), # Warning: range will change if self-normalized
  "ntpts": (12,1800)

}

T_WIND_DR_FACTORS = ['MSD',

#                      'MSD_ratio',
#                      'MSD_tmean',
                     'area',
#                      'area_ratio', # Doesn't work in DR if using self-standardized because min (0) becomes inf.
                     'area_tmean',
                     'arrest_coefficient',
#                      'arrest_coefficient_ratio',
                     'arrest_coefficient_tmean',
                     'aspect',
                     'aspect_ratio',
                     'aspect_tmean',
                     'bbox_area',
#                      'bbox_area_ratio',
                     'bbox_area_tmean',
                     'cumulative_length',
#                      'cumulative_length_ratio',
#                      'cumulative_length_tmean',
                     'dir_autocorr',
                     'dir_autocorr_ratio',
                     'dir_autocorr_tmean',
                     'directedness',
                     'directedness_ratio',
                     'directedness_tmean',
                     'eccentricity',
                     'eccentricity_ratio',
                     'eccentricity_tmean',
                     'endpoint_dir_ratio',
                     'endpoint_dir_ratio_ratio',
                     'endpoint_dir_ratio_tmean',
                     'equivalent_diameter',
                     'equivalent_diameter_ratio',
                     'equivalent_diameter_tmean',
                     'euclidean_dist',
                     'euclidean_dist_ratio',
                     'euclidean_dist_tmean',
                     'extent',
                     'extent_ratio',
                     'extent_tmean',
                     'filled_area',
#                      'filled_area_ratio', # Doesn't work in DR if using self-standardized because min (0) becomes inf.
                     'filled_area_tmean',
                     'glob_turn_deg',
#                      'glob_turn_deg_ratio',
#                      'glob_turn_deg_tmean',
                     'major_axis_length',
#                      'major_axis_length_ratio',
#                      'major_axis_length_tmean',
                     'max_dist',
                     'max_dist_ratio',
                     'max_dist_tmean',
                     'minor_axis_length',
#                      'minor_axis_length_ratio',
#                      'minor_axis_length_tmean',
                     'orientation',
                     'orientation_ratio',
                     'orientation_tmean',
                     'orientedness',
                     'orientedness_ratio',
                     'orientedness_tmean',
                     'outreach_ratio',
                     'outreach_ratio_ratio',
                     'outreach_ratio_tmean',
                     'perimeter',
                     'perimeter_ratio',
                     'perimeter_tmean',
                     'rip_K',
#                      'rip_K_ratio',
#                      'rip_K_tmean',
                     'rip_L',
#                      'rip_L_ratio',
#                      'rip_L_tmean',
                     'rip_p',
#                      'rip_p_ratio',
#                      'rip_p_tmean',
                     'solidity',
                     'solidity_ratio',
                     'solidity_tmean',
                     'speed',
                     'speed_ratio',
                     'speed_tmean',
                     'turn_angle',
                     'turn_angle_ratio',
                     'turn_angle_tmean']



# Booleans to draw or not specific plots.
DRAW_SUPERPLOTS = True
DRAW_DIFFPLOTS = True
DRAW_MARGSCAT = True
DRAW_TIMEPLOTS = True
DRAW_BARPLOTS = True

# Booleans for Analysis components:
'''(Only run pipelines if true)'''
DIMENSION_REDUCTION = True
PARAM_SWEEP = True
CLUSTERING = True

CLUSTER_TSNE = True
CLUSTER_PCA = True
CLUSTER_XY = True

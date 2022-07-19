from initialization.config import *
from initialization.initialization import *

import os
import numpy as np
import pandas as pd

import fnmatch
import h5py
import scipy.stats

#Previously:
# from data_processing import *
# from segmentations import *



def load_data(cond,exp,cond_label, rep_label):#, paths):
    '''
    cond_label and rep_label can be either numbers or names,
    consistent with the superplots function.
    '''

    assert DATA_PATH is not None, 'Error loading DATA_PATH'

    # load_path = data_path + cond + '/' + exp + '/' + track_filename
    load_path = os.path.join(DATA_PATH, cond, exp,
                             TRACK_FILENAME)

    assert os.path.exists(load_path), 'File not present: ' + load_path

    df = pd.read_csv(load_path)
    df.insert(0, 'Condition', cond)
    df.insert(1, 'Replicate_ID', exp)
    df.insert(0, 'Cond_label', cond_label)
    df.insert(1, 'Rep_label', rep_label)

    return df

def populate_experiment_list(fmt=INPUT_FMT,save=True): #'usiigaci'

    '''
    Auto-generate a list of experiments from the folder organization.
    Detects presence of TRACK_FILENAME to determine which to include.

    Serves as a sanity check to ensure that the folders listed in constants:
    CTL_LABEL
    CONDITIONS_TO_INCLUDE
        are in the folder.

    Input: fmt: string indiciating input format. Default: 'usiigaci' (for compatibiltiy)
            accepts: 'usiigaci', 'btrack'
    '''

    exp_list = []
    if(fmt == 'usiigaci'):

        for cond_dir in os.listdir(DATA_PATH):
            f = os.path.join(DATA_PATH, cond_dir)
            if os.path.isdir(f):
                for sub_dir in os.listdir(os.path.join(DATA_PATH, cond_dir)):
                    if(os.path.exists(os.path.join(DATA_PATH, cond_dir,sub_dir, TRACK_FILENAME))):
                        exp_list.append((cond_dir, sub_dir))

        exp_list_df = pd.DataFrame(exp_list, columns=['Condition', 'Experiment'])


        # Until this point, an experiment list is populated entirely from the folder structure.

        # Test that the correct conditions are listed in the config
        assert CTL_LABEL in exp_list_df['Condition'].unique(), 'CTL_LABEL not in exp_list'
        for cond in CONDITIONS_TO_INCLUDE:
            assert cond in exp_list_df['Condition'].unique(), cond+' not in exp_list'

        # Filter the condition list such that only those in CONDITIONS_TO_INCLUDE are included
        included_exps = exp_list_df.Condition.isin(CONDITIONS_TO_INCLUDE)
        exp_list_df = exp_list_df[included_exps]

        # Reorder the experiment list according to the order of CONDITIONS_TO_INCLUDE
        exp_list_df['Condition'] = pd.Categorical(exp_list_df['Condition'], CONDITIONS_TO_INCLUDE)
        exp_list_df.sort_values(by='Condition', inplace=True, ascending=True)

    if(fmt == 'btrack'):

        exp_list_df = pd.DataFrame()

        for cond_dir in os.listdir(DATA_PATH):
            f = os.path.join(DATA_PATH, cond_dir)
            contents = os.listdir(f)
            pattern = '*' + TRACK_FILENAME#"*.h5"

            for entry in contents:
                if fnmatch.fnmatch(entry, pattern):

                    exp_name = entry[:-3] # remove the '.h5' extension from the string.
                    exp_list.append((cond_dir, exp_name))

        exp_list_df = pd.DataFrame(exp_list, columns=['Condition', 'Experiment'])

    # A test to be sure the same Experiment name is not used twice.
    for rep in exp_list_df['Experiment'].unique():
        rep_df = exp_list_df[exp_list_df['Experiment'] == rep]
        assert len(rep_df['Condition'].unique()) == 1, 'Experiment name not unique to condition'

    if USE_SHORTLABELS:
        exp_list_df['Replicate_ID'] = exp_list_df['Experiment']
        exp_list_df = add_shortlabels(exp_list_df)

    if save:
        exp_list_df.to_csv(DATA_OUTPUT+'exp_list' + '.csv') #+ TIMESTAMP (already in TIMESTAMP-named folder)

    return exp_list_df


def get_experiments(path):

    '''
    Get experiments from an external csv file, indicated by path
    Input:
        path: file path where the experiment list is located (must be in same Folder
        as the condition subfolders.)
    '''

    print(' Need to replace exp_list in get_experiments() with the self-populated list.')
    exp_list = pd.read_csv(path, sep="\t|,") # Allow tabs or spaces as seperators
    print('Experiment list:')
    print(exp_list)

    assert isinstance(exp_list, pd.DataFrame), 'exp_list is not a dataframe'
    assert exp_list.columns[0] == 'Condition', 'Error loading csv headers'
    assert exp_list.columns[1] == 'Experiment', 'Error loading csv headers'

    # Verify that each experiment folder exists and contains the tracking file
    for i, row in exp_list.iterrows():

        this_path = os.path.join(DATA_PATH,
                                 row['Condition'],
                                 row['Experiment'])

        this_file = os.path.join(this_path, TRACK_FILENAME)

        assert os.path.exists(this_path), 'Folder not present: ' + this_path
        assert os.path.exists(this_path), 'File not present: ' + this_file

    return exp_list


def combine_dataframes(exp_list, fmt=INPUT_FMT, dedup_columns=True): #, paths):

    '''
    Load results from multiple experiments together into a single DataFrame
    To be used in subsequent processing steps of the pipeline.

    Input:
        exp_list: DataFrame containing Experiment and Condition for each replicate.
        fmt: string: indicating data source. Accepts: 'usiigaci' and 'btrack'
            default: 'usiigaci' (for backward compatibility)
    '''
    combined_df = pd.DataFrame()

    if(fmt=='usiigaci'):

        uniq_cond =list(pd.unique(exp_list['Condition']))

        for i, cond in enumerate(uniq_cond):
            #Return sub_df for each condition, containing replicates
            sub_df = pd.DataFrame()
            sub_df = exp_list.loc[(exp_list['Condition'] == cond )]

            # Get the list of replicates for each condition.
            rep_list = list(sub_df['Experiment'])


            # For each replicate in the list
            for j, rep in enumerate(rep_list):
                loaded_df =  load_data(cond,rep, cond,j)#, paths)

                '''
                Apply pixel micron corrections immediately to the loaded dataframe
                '''

                # Test that the position values are reasonable, considering the image dimensions
                if(CALIBRATED_POS):

                    print('CALIBRATED_POS == ' ,str(CALIBRATED_POS), ', Input positions in microns.')

                    loaded_df['x_um'] = loaded_df['x']
                    loaded_df['y_um'] = loaded_df['y']
                    loaded_df['x_pix'] = loaded_df['x'] / MICRONS_PER_PIXEL
                    loaded_df['y_pix'] = loaded_df['y'] / MICRONS_PER_PIXEL


                else:
                    print('CALIBRATED_POS == ' ,str(CALIBRATED_POS), ', Input positions in pixels.')

                    loaded_df['x_um'] = loaded_df['x'] * MICRONS_PER_PIXEL
                    loaded_df['y_um'] = loaded_df['y'] * MICRONS_PER_PIXEL
                    loaded_df['x_pix'] = loaded_df['x']
                    loaded_df['y_pix'] = loaded_df['y']

                assert np.max(loaded_df['x_pix']) <= IMAGE_WIDTH, 'Position not within image coordinates, max x: ' + str(np.max(combined_df['x_pix']))
                assert np.max(loaded_df['y_pix']) <= IMAGE_HEIGHT, 'Position not within image coordinates, max y: '+ str(np.max(combined_df['y_pix']))

                # Also ensure that the values do fill up the image.
                # assert np.max(loaded_df['x_pix']) >= IMAGE_WIDTH * 0.90, 'Position seems off'
                # assert np.max(loaded_df['y_pix']) >= IMAGE_HEIGHT * 0.90, 'Position seems off'

                print('max x_pix: ', str(np.max(loaded_df['x_pix'])), ', image width: ', IMAGE_WIDTH)
                print('max y_pix: ', str(np.max(loaded_df['y_pix'])), ', image height: ', IMAGE_HEIGHT)
                print('max x_um: ', str(np.max(loaded_df['x_um'])), ', MICRONS_PER_PIXEL: ', MICRONS_PER_PIXEL)
                print('max y_um: ', str(np.max(loaded_df['y_um'])), ', MICRONS_PER_PIXEL: ', MICRONS_PER_PIXEL)

                '''
                Insert step here to remove RegionProps from the Usiigaci imported DataFrame
                overlap = list(set(loaded_df.columns).intersection(REGIONPROPS_LIST))
                print(' Removing regionprops columns from usiigaci-imported dataframe:', overlap)
                loaded_df.drop(columns=overlap, inplace=True)
                ** Not implemented yet: wait to see whether we include the regionprops from btrack directly
                '''


                # Additional data saving and loading shortcuts.
                # To save processing time, instead of recalculating segmentation and migration calculations.

                calcs_path = os.path.join(DATA_PATH, cond, rep,
                                 'seg_mig_calcs.csv')

                if(os.path.exists(calcs_path) and not OVERWRITE):
                    # If the file exists, load it
                    print('Loading existing file: ' + cond + ', '+rep + '.csv')
                    mig_df = pd.read_csv(calcs_path)

                else:
                    print('WARNING: upstream calibratin method not yet tested for Usiitracker')
                    if(CALIBRATED_POS):

                        print('CALIBRATED_POS == ' ,str(CALIBRATED_POS), ', Input positions in microns.')

                        loaded_df['x_um'] = loaded_df['x']
                        loaded_df['y_um'] = loaded_df['y']
                        loaded_df['x_pix'] = loaded_df['x'] / MICRONS_PER_PIXEL
                        loaded_df['y_pix'] = loaded_df['y'] / MICRONS_PER_PIXEL


                    else:
                        print('CALIBRATED_POS == ' ,str(CALIBRATED_POS), ', Input positions in pixels.')

                        loaded_df['x_um'] = loaded_df['x'] * MICRONS_PER_PIXEL
                        loaded_df['y_um'] = loaded_df['y'] * MICRONS_PER_PIXEL
                        loaded_df['x_pix'] = loaded_df['x']
                        loaded_df['y_pix'] = loaded_df['y']


                    # Run the migration calcs and seg_df functions
                    print(calcs_path +' doesnt already exist, processing input data:')
                    # Segmentations and region properties
                    seg_df = batch_shape_measurements(loaded_df,n_samples ='all')#'all') # 10000 for btracker. This is just a coding test. Its not a valid way to select out a sample of the data, because it does not sample equally from each replicate/timepoint.
                    seg_df = clean_comb_df(seg_df,deduplicate=dedup_columns)
                    if dedup_columns is False:
                        print('Warning: deduplicate=False on clean_comb_df(), there may be duplicate regionprops columns. Only intended use for debugging/testing.')
                    # Process the migration calculations and clean resulting dataframe
                    mig_df = migration_calcs(seg_df)
                    mig_df.dropna(inplace=True)
                    mig_df.reset_index(inplace=True, drop=True)

                    # Save the file
                    mig_df.to_csv(calcs_path)
                    print('Saving file: ' + calcs_path)

                combined_df =  pd.concat([combined_df,mig_df])

        combined_df.reset_index(inplace=True, drop=True)

        print('Generated combined_df in new way, still needs to be integrated into original function, and adapted for 3D data')


        # Testing the validity of the combined DataFrame
        assert len(combined_df.index) > 0, 'Error combining dataframe, no data'

        assert np.array_equal(pd.unique(combined_df['Cond_label']),
                             pd.unique(exp_list['Condition'])),'Condition lists not equal'

        assert len(pd.unique(combined_df['Replicate_ID'])) == len(pd.unique(exp_list['Experiment'])), ' Number of experiments not matching between exp_list and comb_df'

        assert np.array_equal(pd.unique(combined_df['Replicate_ID']),
                             pd.unique(exp_list['Experiment'])),'Condition lists not equal'


    elif(fmt=='btrack'):

        cond_list = exp_list['Condition']
        exp_list = exp_list['Experiment']

        for i, cond in enumerate(cond_list):

            calcs_path = os.path.join(DATA_PATH, cond,  exp_list[i],
                             'seg_mig_calcs.csv')

            if(os.path.exists(calcs_path) and not OVERWRITE):
                # If the file exists, load it
                print('Loading existing file: ' + cond + ', '+ exp_list[i] + '.csv')
                mig_df = pd.read_csv(calcs_path)

            else:


                if not os.path.exists(os.path.join(DATA_PATH, cond,  exp_list[i])):
                    os.makedirs(os.path.join(DATA_PATH, cond,  exp_list[i]))

                # Only load the h5 file if its not alraedy been processed.
                this_file = os.path.join(DATA_PATH,cond,exp_list[i]) + TRACK_FILENAME
                f = h5py.File(this_file)
                assert 'segmentation' in f.keys(), 'segmentation not found'
                print('h5 file contents: ',f.keys())
                file_contents = btrack_unpack(this_file)

                h5_df = h5_to_df(file_contents)
                h5_df['Condition'] = cond
                h5_df['Experiment'] = exp_list[i]


                print(h5_df.columns)
                #
                # New part to do the segmentation and migration calcs per replicate
                #
                # Make sure it has Replicate_ID column
                h5_df = clean_comb_df(h5_df)

                # Calibration must be done BEFORE the processing steps:
                if(CALIBRATED_POS):

                    print('CALIBRATED_POS == ' ,str(CALIBRATED_POS), ', Input positions in microns.')

                    h5_df['x_um'] = h5_df['x']
                    h5_df['y_um'] = h5_df['y']
                    h5_df['x_pix'] = h5_df['x'] / MICRONS_PER_PIXEL
                    h5_df['y_pix'] = h5_df['y'] / MICRONS_PER_PIXEL


                else:
                    print('CALIBRATED_POS == ' ,str(CALIBRATED_POS), ', Input positions in pixels.')

                    h5_df['x_um'] = h5_df['x'] * MICRONS_PER_PIXEL
                    h5_df['y_um'] = h5_df['y'] * MICRONS_PER_PIXEL
                    h5_df['x_pix'] = h5_df['x']
                    h5_df['y_pix'] = h5_df['y']

                print(h5_df.columns)
                # Run the migration calcs and seg_df functions
                print(calcs_path +' doesnt already exist, processing input data:')

                if(CALCULATE_REGIONPROPS):
                    # Segmentations and region properties
                    seg_df = batch_shape_measurements(h5_df,n_samples ='all')#'all') # 10000 for btracker. This is just a coding test. Its not a valid way to select out a sample of the data, because it does not sample equally from each replicate/timepoint.
                    # seg_df = clean_comb_df(seg_df)
                    seg_df = clean_comb_df(seg_df,deduplicate=dedup_columns)
                    if dedup_columns is False:
                        print('Warning: deduplicate=False on clean_comb_df(), there may be duplicate regionprops columns. Only intended use for debugging/testing.')

                    mig_df = migration_calcs(seg_df)

                else: # Assume that h5 file already contains regionprops

                    mig_df = migration_calcs(h5_df)

                mig_df.dropna(inplace=True)
                mig_df.reset_index(inplace=True, drop=True)

                # Save the file
                mig_df.to_csv(calcs_path)
                print('Saving file: ' + calcs_path)

            combined_df =  pd.concat([combined_df,mig_df])
            combined_df.reset_index(inplace=True, drop=True)

    else:

        print('Unrecognized format')



    # Assertions only for usiigaci-tracked data, as btracker can have values beyond this range
    if(fmt=='usiigaci'):
        assert np.max(combined_df['x_pix']) <= IMAGE_WIDTH, 'Position not within image coordinates, max x: ' + str(np.max(combined_df['x_pix']))
        assert np.max(combined_df['y_pix']) <= IMAGE_HEIGHT, 'Position not within image coordinates, max y: '+ str(np.max(combined_df['y_pix']))

    print('max x_pix: ', str(np.max(combined_df['x_pix'])), ', image width: ', IMAGE_WIDTH)
    print('max y_pix: ', str(np.max(combined_df['y_pix'])), ', image height: ', IMAGE_HEIGHT)
    print('max x_um: ', str(np.max(combined_df['x_um'])), ', MICRONS_PER_PIXEL: ', MICRONS_PER_PIXEL)
    print('max y_um: ', str(np.max(combined_df['y_um'])), ', MICRONS_PER_PIXEL: ', MICRONS_PER_PIXEL)

    # Clean and calibrate the dataframe
    if(USE_SHORTLABELS):
        combined_df = add_shortlabels(combined_df)

    # Renamed the old index value for returning to the raw input downstream.
    combined_df.rename(columns={'Unnamed: 0': 'rep_row_ind'}, inplace=True)
    return combined_df


def h5_to_df(h5_data,min_pts=0):

    '''
    Data wrangling operations to take the extracted h5 data and transform to the same format as the comb_df used with usiigaci tracking data.

    Input:
        h5_data: dict outputed by btrack_unpack with keys:
            ['coords', 'labels', 'omap', 'lbepr', 'dummies', 'fates', 'tmap', 'ttracks', 'segmentation']
        min_pts: int: minimum number of points to consider keeping the track

    Output:

        h5_df: DataFrame containing h5 data in format consistent with Usiigaci
    '''

    tmap = h5_data['tmap']
    coords = h5_data['coords']
    # dummies = h5_data['dummies']
    ttracks = h5_data['ttracks']

    if  'dummies' in h5_data.keys():

        dummies = h5_data['dummies']

    if  'regionprops' in h5_data.keys():#if h5_data['regionprops'] is not None:

        print('h5_data passed t h5_to_df() contains regionprops, adding to df.')
        print(h5_data['regionprops'].columns)
        props_arr = np.asarray(h5_data['regionprops'])
        print('props_arr: ',props_arr.shape)
        print('coords: ',coords.shape)

        # The modified version of the h5 to dataframe function that also processes the regionprops data
        coord_list = []
        regionprops_list = []
        for i in range(tmap.shape[0]):

            this_vect = ttracks[tmap[i,0]:tmap[i,1]]
            assert len(this_vect) > 0, 'Zero length element of ttracks found.'

            if(len(this_vect) > min_pts):

                coord_slice = coords[this_vect,:]
                props_slice = props_arr[this_vect,:]
                negs = this_vect[this_vect < 0]
                neg_ids = np.where(this_vect < 0)

                if  'dummies' in h5_data.keys() and len(negs) > 0:
                    dummy_slice = dummies[abs(negs)-1,:]

                    # Add the dummy slice to the coord_slice
                    coord_slice[neg_ids,:] = dummy_slice

                coord_slice = np.c_[coord_slice, np.full(coord_slice.shape[0],i)] # Add index as column
#                 props_slice = np.c_[props_slice, np.full(props_slice.shape[0],i)] # Add index as c

            coord_list.append(coord_slice)
            regionprops_list.append(props_slice)

        coord_array = np.vstack(coord_list)
        props_array = np.vstack(regionprops_list)

        h5_pos_df = pd.DataFrame(data=coord_array,columns=['frame', 'x', 'y', 'z', '_', 'particle'])
        h5_props_df = pd.DataFrame(data=props_array,columns=h5_data['regionprops'].columns)

        # Concatenate the resulting dataframes
        h5_df = pd.concat([h5_pos_df, h5_props_df],axis=1)

        # Sort the resulting combined dataframe
        h5_df.sort_values(['frame', 'particle'], ascending=[True, True], inplace=True)


    else:
        # The way the function used to work, before the Btracker Regionprops were included.
        coord_list = []
        for i in range(tmap.shape[0]):

            this_vect = ttracks[tmap[i,0]:tmap[i,1]]
            assert len(this_vect) > 0, 'Zero length element of ttracks found.'

            if(len(this_vect) > min_pts):

                coord_slice = coords[this_vect,:]
                negs = this_vect[this_vect < 0]
                neg_ids = np.where(this_vect < 0)

                if  'dummies' in h5_data.keys() and len(negs) > 0:
                    dummy_slice = dummies[abs(negs)-1,:]

                    # Add the dummy slice to the coord_slice
                    coord_slice[neg_ids,:] = dummy_slice

                coord_slice = np.c_[coord_slice, np.full(coord_slice.shape[0],i)] # Add index as column


            coord_list.append(coord_slice)

        coord_array = np.vstack(coord_list)

        h5_df = pd.DataFrame(data=coord_array,columns=['frame', 'x', 'y', 'z', '_', 'particle'])
        h5_df.sort_values(['frame', 'particle'], ascending=[True, True], inplace=True)

    return h5_df

def csv_summary(df, label='csv_summary', supp_headers=None, cond_header='Condition', rep_header='Replicate_ID', plots=False):

    '''
    Export a csv summary of the provided dataframe.
    Intended to be useful at multiple steps of the processing pipeline
    to compare intput and output.

    Input:
        df: DataFrame to be summarized across experimental conditions and replicates.
        label: str, name for the saved csv file.
        supp_headers: list of str, additional column headers to be included in the summary.
            - For example, ['tSNE1', 'tSNE2'] if summarizing a low-dimension dataframe
        cond_header: str, label fo the condition column header in the input dataframe
        rep_header: str, label fo the repicate column header in the input dataframe

    NOTE:
        headers_to_avg copied from 'select_factors' in data_processing.py,
        consider defining them centrally in the config file
    '''
    headers_to_avg = DR_FACTORS

    if supp_headers is not None:

        headers_to_avg = headers_to_avg + supp_headers
        print(headers_to_avg)

    # Unique list of the conditions included.
    uniq_cond = list(pd.unique(df[cond_header]))

    # New dataframe to store the summary.
    sum_df = pd.DataFrame()#columns=[cond_header, rep_header, 'N'] + headers_to_avg)
    mean_df = pd.DataFrame()
    median_df = pd.DataFrame()
    stdev_df = pd.DataFrame()
    sem_df = pd.DataFrame()

    i_exp = 0 # Unique index for each experiment (unique between multiple conditions)

    for cond in uniq_cond:

        # DataFrame containing the matching condition
        sub_df = df[df[cond_header] == cond]

        # Unique list of replicates for this condition
        uniq_rep = list(pd.unique(sub_df[rep_header]))

        for rep in uniq_rep:

            subsub_df = sub_df[sub_df[rep_header] == rep]
            this_n = len(pd.unique(subsub_df['particle'])) # Alternatively use the id directly

            # Add to dataframe(s).
            sum_df.loc[i_exp, cond_header] = cond
            sum_df.loc[i_exp, rep_header] = rep
            sum_df.loc[i_exp, 'N'] = this_n

            mean_df.loc[i_exp, cond_header] = cond
            mean_df.loc[i_exp, rep_header] = rep
            mean_df.loc[i_exp, 'N'] = this_n

            median_df.loc[i_exp, cond_header] = cond
            median_df.loc[i_exp, rep_header] = rep
            median_df.loc[i_exp, 'N'] = this_n

            stdev_df.loc[i_exp, cond_header] = cond
            stdev_df.loc[i_exp, rep_header] = rep
            stdev_df.loc[i_exp, 'N'] = this_n

            sem_df.loc[i_exp, cond_header] = cond
            sem_df.loc[i_exp, rep_header] = rep
            sem_df.loc[i_exp, 'N'] = this_n

            for header in headers_to_avg:

                this_mean = np.mean(subsub_df[header])
                this_median = np.median(subsub_df[header])
                this_stdev = np.std(subsub_df[header])
                this_sem = scipy.stats.sem(subsub_df[header])

                sum_df.at[i_exp, header] = '' # Init cell with empty string
                sum_df = sum_df.astype({header: str, 'N': int})

                mean_df.at[i_exp, header] = this_mean
                median_df.at[i_exp, header] = this_median
                stdev_df.at[i_exp, header] = this_stdev
                sem_df.at[i_exp, header] = this_sem

            i_exp += 1


    mean_df.to_csv(DATA_OUTPUT+label+'_mean.csv')
    median_df.to_csv(DATA_OUTPUT+label+'_median.csv')
    stdev_df.to_csv(DATA_OUTPUT+label+'_stdev.csv')
    sem_df.to_csv(DATA_OUTPUT+label+'_sem.csv')

    with pd.ExcelWriter(DATA_OUTPUT+label+'.xlsx') as writer:

        mean_df.to_excel(writer,sheet_name='Mean')
        median_df.to_excel(writer,sheet_name='Median')
        stdev_df.to_excel(writer,sheet_name='StDev')
        sem_df.to_excel(writer,sheet_name='SEM')


def add_shortlabels(df_in):

    '''
    If shortlabels are used, add the shortlabels to the dataframe.
    '''
    assert USE_SHORTLABELS is True, 'This should only be used if USE_SHORTLABELS is True...'

    df_sl = df_in.copy()

    full_condition_list = list(df_in['Condition'])
    condition_shortlabels = []

    # Create a shortlabel per replicate
    rep_shortlabel_list = []

    for this_cond_label in full_condition_list:

        this_cond_ind = CONDITIONS_TO_INCLUDE.index(this_cond_label)
        this_shortlabel = CONDITION_SHORTLABELS[this_cond_ind]
        condition_shortlabels.append(this_shortlabel)

    df_sl['Condition_shortlabel'] = condition_shortlabels

    for cond in df_in['Condition'].unique():

        cond_sub_df = df_in[df_in['Condition'] == cond]
        for i, rep in enumerate(cond_sub_df['Replicate_ID'].unique()):

            rep_sub_df = df_in[df_in['Replicate_ID'] == rep]

            this_cond_ind = CONDITIONS_TO_INCLUDE.index(cond)
            this_cond_shortlabel = CONDITION_SHORTLABELS[this_cond_ind]

            this_rep_shortlabel = this_cond_shortlabel + '_' + str(i)

            # Add the list for each of these replicates, into the subdf
            rep_sub_df['Replicate_shortlabel'] = this_rep_shortlabel
            # Add the subdf into a list to add back into the main df
            rep_shortlabel_list.append(rep_sub_df['Replicate_shortlabel'])

    # Flatten the list of lists and add it as a colun to the dataframe
    flat_list = [item for sublist in rep_shortlabel_list for item in sublist]

    df_sl['Replicate_shortlabel'] = flat_list

    return df_sl




'''
Basic format of the HDF file is: (FROM BAYESIAN TRACKER DOCS)
        segmentation/
            images          - (J x (d) x h x w) uint16 segmentation
        objects/
            obj_type_1/
                coords      - (I x 5) [t, x, y, z, object_type]
                labels      - (I x D) [label, (softmax scores ...)]
                map         - (J x 2) [start_index, end_index] -> coords array
                properties  - (I x D) [property-0, ..., property-D]
            ...
        tracks/
            obj_type_1/
                tracks      - (I x 1) [index into coords]
                dummies     - similar to coords, but for dummy objects
                map         - (K x 2) [start_index, end_index] -> tracks array
                LBEPRG      - (K x 6) [L, B, E, P, R, G]
                fates       - (K x n) [fate_from_tracker, ...future_expansion]

'''

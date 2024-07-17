import sys
import re
import os
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from matplotlib.animation import FuncAnimation
from scipy import stats
import random
from itertools import count

print(sys.path)

# Define paths
proj_dir = pathlib.Path(pathlib.Path.home(), 'bacmman', 'bacmman_results', '2022-05-26_long_channels')
csv_file = proj_dir / '2022-05-26_long_channels_1_produced_2024-04-16.csv'
selections_file = proj_dir / '2022-05-26_long_channels_Selections_produced_2024-04-13.csv'
results_dir = proj_dir / "results_2024-07-17"

# Make a directory to store subset dataframes
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Define the data types for specific columns
dtype_spec = {10: 'float64', 12: 'float64'}

# Load data 
data = pd.read_csv(csv_file, sep=';', dtype=dtype_spec, low_memory=False)

# Convert Frame to Time
data['Time']=data['Frame']*10/60
data['NextDivisionTime']=data['NextDivisionFrame']*10/60
data['PreviousDivisionTime']=data['PreviousDivisionFrame']*10/60

# Time parameters

# Define experiment phases
has_exp_phase = True
has_grad_phase = True
has_stat_phase = True
has_wakeup_phase = True
has_drug_phase = True
has_recov_phase = True

# Initialize dictionary
phases_duration = {}

# Define phases duration in experiment

if has_exp_phase:
    phases_duration['exp'] = 3
    exp_phase_start = 0
    print(exp_phase_start)
    exp_phase_end = int(exp_phase_start + (phases_duration['exp'] * 60/ 10) - 1)
    print(exp_phase_end)

if has_grad_phase:
    phases_duration['grad'] = 2.5
    grad_phase_start = int(exp_phase_end +1)
    print(grad_phase_start)
    grad_phase_end = int(grad_phase_start + (phases_duration['grad']* 60/ 10) -1)
    print(grad_phase_end)

if has_stat_phase:
    phases_duration['stat'] = 10
    stat_phase_start = int(grad_phase_end +1)
    print(stat_phase_start)
    stat_phase_end = int(stat_phase_start + (phases_duration['stat']* 60/ 10) -1)
    print(stat_phase_end)
    
if has_wakeup_phase:
    phases_duration['wakeup'] = 1 # CHANGE ACCORDINGLY
    wakeup_phase_start = int(stat_phase_end + 1)
    print(wakeup_phase_start)
    wakeup_phase_end = int(wakeup_phase_start + (phases_duration['wakeup']* 60/ 10) -1)
    print(wakeup_phase_end)

if has_drug_phase:
    phases_duration['drug'] = 3
    if not has_wakeup_phase:
        drug_phase_start = int(stat_phase_end + 1)
        print(drug_phase_start)
        drug_phase_end = int(drug_phase_start + (phases_duration['drug']* 60/ 10) - 1)
        print(drug_phase_end)
    else: 
        drug_phase_start = int(wakeup_phase_end + 1)
        print(drug_phase_start)
        drug_phase_end = int(drug_phase_start + (phases_duration['drug']* 60/ 10) - 1)
        print(drug_phase_end)

if has_recov_phase:
    phases_duration['recov'] = 24
    if has_drug_phase:
        recov_phase_start = int(drug_phase_end + 1)
        print(recov_phase_start)
        recov_phase_end = data['Frame'].max() # int(recov_phase_start + (phases_duration['recov']* 60/ 10) -1)
        print(recov_phase_end)
    else:
        recov_phase_start = int(stat_phase_end + 1)
        print(recov_phase_start)
        recov_phase_end = data['Frame'].max() # int(recov_phase_start + (phases_duration['recov']* 60/ 10) -1)
        print(recov_phase_end)
    

# Calculate cumulative sum of phase durations
cumulative_sum = 0
phases_switches = {}

for phase_name, duration in phases_duration.items():
    if duration is not None:  # Only accumulate if duration is defined
        cumulative_sum += duration
    phases_switches[phase_name] = cumulative_sum

# Print cumulative durations
for phase_name, phase_end in phases_switches.items():
    print(f"{phase_name} end time = {phase_end}")


# Define experiment phases and their durations (in hours)
phase_config = {
    'exp': {'exists': True, 'duration': 3},
    'grad': {'exists': True, 'duration': 2.5},
    'stat': {'exists': True, 'duration': 10},
    'wakeup': {'exists': False, 'duration': 1},
    'drug': {'exists': True, 'duration': 3},
    'recov': {'exists': True, 'duration': 24}
}

# Initialize dictionaries for phase durations and start times
phases_duration = {}
phases_start = {}

# Initialize current start time
current_start = 0

# Iterate through each phase in phase_config
for phase_name, config in phase_config.items():
    if config['exists']:
        duration = config['duration']
        phases_duration[phase_name] = duration
        phases_start[phase_name] = current_start
        
        # Calculate phase end time
        phase_end = int(current_start + (duration * 60 / 10) - 1)
        print(f"{phase_name}_phase_start:", current_start)
        print(f"{phase_name}_phase_end:", phase_end)
        
        # Update current start time for the next phase
        current_start = phase_end + 1

# Calculate cumulative sum of phase durations
cumulative_sum = 0
phases_switches = {}

for phase_name, duration in phases_duration.items():
    if duration is not None:  # Only accumulate if duration is defined
        cumulative_sum += duration
    phases_switches[phase_name] = cumulative_sum

# Print cumulative durations
for phase_name, phase_end in phases_switches.items():
    print(f"{phase_name} time of switch = {phase_end}")

exp_end = phases_switches.get('exp')
grad_end = phases_switches.get('grad')
stat_end = phases_switches.get('stat')
drug_end = phases_switches.get('drug')
recov_end = phases_switches.get('recov')
if has_wakeup_phase:
    wakeup_end = phases_switches.get('wakeup')
else:
    wakeup_end = None

def add_info(df):
    
    ChIdx = [int(re.split("\-",ind)[1]) for ind in df['Indices']]
    df['ChannelIdx'] = ChIdx
    
    ChPos = [int(re.split("\-",ind)[2]) for ind in df['Indices']]
    df['CellPosChannel'] = ChPos 
    
    #combine PositionIdx-ChannelIdx-BacteriaLineage into single string and add string lin_id_str property
    df['lin_id_str_long'] = df['PositionIdx'].map(str) + '-' + df['ChannelIdx'].map(str) + '-' + df['BacteriaLineage'].map(str)
    
    #combine PositionIdx-ChannelIdx-BacteriaLineage into single string and add string lin_id_str property
    df['lin_id_str'] = df['Position'].map(str) + '-' + df['Indices'].map(str)

    #combine PositionIdx-ChannelIdx-BacteriaLineage into single string and add string lin_id_str property
    df['lin_id_str_Prev'] = np.where(~df['Prev'].isna(), df['Position'].astype(str) + '-' + df['Prev'].astype(str), np.nan)
    
    #combine Position and ChannelIdx to group persister cells selections 
    df['lin_id_str2'] = df['PositionIdx'].map(str) + '-' + df['ChannelIdx'].map(str)  
    
    #convert the ID to a unique integer
    df['id_cell'] = df.groupby(['lin_id_str_long']).ngroup()
    
    #combine PositionIdx-ParentTrackHeadIndices into single string and add eternal cell_id_str property 
    #df['et_id_string'] = df['Position'].map(str) + '-' + df[' '].map(str)
    #df['et_id_cell']= df.groupby(['et_id_string']).ngroup()
    
    return None

# Add extra info
add_info(data)

def add_info2(df):
    
    ChIdx = [int(re.split("\-",ind)[1]) for ind in df['Indices']]
    df['ChannelIdx'] = ChIdx

    ChPos = [int(re.split("\-",ind)[2]) for ind in df['Indices']]
    df['CellPosChannel'] = ChPos 
    
    #combine PositionIdx-ChannelIdx-BacteriaLineage into single string and add string lin_id_str property
    #df['lin_id_str'] = df['PositionIdx'].map(str) + '-' + df['ChannelIdx'].map(str) + '-' + df['BacteriaLineage'].map(str)
    
    #combine PositionIdx-ChannelIdx-BacteriaLineage into single string and add string lin_id_str property
    df['lin_id_str'] = df['Position'].map(str) + '-' + df['Indices'].map(str)                           
    
    #combine Position and ChannelIdx to group persister cells selections 
    df['lin_id_str2'] = df['PositionIdx'].map(str) + '-' + df['ChannelIdx'].map(str) 
    
    #convert the ID to a unique integer
    df['id_cell'] = df.groupby(['lin_id_str']).ngroup()
    
    #combine PositionIdx-ParentTrackHeadIndices into single string and add eternal cell_id_str property 
    #df['et_id_string'] = df['Position'].map(str) + '-' + df[' '].map(str)
    #df['et_id_cell']= df.groupby(['et_id_string']).ngroup()
    
    return None


# Import bacmman selections

# Check if the selections_file exists and print the path if it does
if selections_file.exists():

    df_selections = pd.read_csv(selections_file, sep=';')
    # Convert Frame to Time
    df_selections['Time'] = df_selections['Frame']*10/60

    # Add extra info to selections 
    add_info2(df_selections)
    
    # Selections names
    selections_names = df_selections['SelectionName'].unique()
    print(selections_names)
    
    # Positions with persisters
    positions_with_persisters = np.unique(df_selections['PositionIdx'])
    print(positions_with_persisters)

    # Make dataframe with Selections
    pd.options.mode.copy_on_write = True # to avoid warnings

    df1 = data.copy() # copy csv data
    
    # Persisters
    persisters = df_selections.loc[df_selections['SelectionName']== selections_names[0]]
    df_persisters = df1[df1['lin_id_str'].isin(persisters['lin_id_str'])]
    df_persisters['cell_type'] = 'persister'

    # Susceptibles Sisters
    sisters = df_selections.loc[df_selections['SelectionName']== selections_names[1]]
    df_sisters = df1[df1['lin_id_str'].isin(sisters['lin_id_str'])]
    df_sisters['cell_type'] = 'susceptible_sister' 
    
     # Filter selections data for persisters and susceptible_sisters after drug treatment
    persisters_after_drug = df_selections.loc[(df_selections['SelectionName'] == selections_names[0]) & (df_selections['Time'] > drug_end)]
    susceptible_sisters_after_drug = df_selections.loc[(df_selections['SelectionName'] == selections_names[1]) & (df_selections['Time'] > drug_end)]
    
    # Get lin_id_str of persisters and susceptible_sisters after drug treatment
    lin_ids_to_remove = pd.concat([persisters_after_drug['lin_id_str'], susceptible_sisters_after_drug['lin_id_str']])

    # Remove data from df where lin_id_str is in lin_ids_to_remove
    df_susceptibles = df1[~df1['lin_id_str'].isin(lin_ids_to_remove)]
    df_susceptibles['cell_type'] = ''
    #print(df_susceptibles['cell_type'].unique())
   
else: 
    #print("Either df_persisters or df_sisters is not defined or None")
    df_persisters = None
    df_sisters = None


# Get number of cell divisions of selections at different phases using bacmman lineages. Short working version for multiple Selections: here applied to persisters and susceptible sisters. 

if 'df_persisters' in locals() and 'df_sisters' in locals() and df_persisters is not None and df_sisters is not None:
    # Make copies of the merged DataFrames
    dfs = [df_persisters.copy(), df_sisters.copy()]
    
    # List to store the final concatenated DataFrames
    concatenated_dfs = []
    
    # Iterate over the list of DataFrames
    unique_lineage_id_selections =[]
    Bacterialineage_selections =[]
    for df in dfs:
        # Grouping operation
        groups = df.groupby('lin_id_str2')
        group_dfs = []
    
        for group_name, group_df in groups:
            max_frame = group_df['Frame'].max()
            max_frame_rows = group_df[group_df['Frame'] == max_frame]
            group_df['unique_lineage_id'] = max_frame_rows['lin_id_str'].values[0] # assigns the max value of lin_id_str as unique_lineage_id since we are reconstructing the lineage backwards 
            unique_lineage_id_selections.append(max_frame_rows['lin_id_str'].values[0]) # store unique_lineage_id
            Bacterialineage_selections.append(max_frame_rows['BacteriaLineage'].values[0]) # store Bacterialineage_selections 
    
            # Get number of cell divisions using lineages from bacmman (length_drops are more accurate for counting number of divisions for Selections)
            num_div_exp_phase = group_df[(group_df['Time'] >= 0) & (group_df['Time'] <= exp_end)]['PreviousDivisionFrame'].isna().sum()
            num_div_grad_phase = group_df[(group_df['Time'] > exp_end) & (group_df['Time'] <= grad_end)]['PreviousDivisionFrame'].isna().sum()
            num_div_stat_phase = group_df[(group_df['Time'] > grad_end) & (group_df['Time'] <= stat_end)]['PreviousDivisionFrame'].isna().sum()
            num_div_drug_phase = group_df[(group_df['Time'] > stat_end) & (group_df['Time'] <= drug_end)]['PreviousDivisionFrame'].isna().sum()
    
            group_df['num_div_exp_phase'] = num_div_exp_phase
            group_df['num_div_grad_phase'] = num_div_grad_phase
            group_df['num_div_stat_phase'] = num_div_stat_phase
            group_df['num_div_drug_phase'] = num_div_drug_phase
    
    
            group_dfs.append(group_df)
    
        # Concatenate DataFrames
        concatenated_df = pd.concat(group_dfs)
        concatenated_dfs.append(concatenated_df)
    
    # Concatenate both merged DataFrames
    df_selections_merged = pd.concat(concatenated_dfs)
    
else: 
    #print("Either df_persisters or df_sisters is not defined or None")
    df_selections_merged = None



# Working version 1: removes repeated data from treated lineages
# Define df_susceptibles if existent

if 'df_susceptibles' in locals() and df_susceptibles is not None:
    df_susceptibles = df_susceptibles #df_susceptibles #
else:
    df_susceptibles = df1 # df1  #df_without_persisters_sisters
    

# Iterate over poistions and micochannels

filtered_df = pd.DataFrame(df_susceptibles) # Important: df_susceptibles exclude lin_id_str of persisters from Frames > 111. See above
upper_limit_frame = drug_phase_end
if not has_wakeup_phase:
    lower_limit_frame = stat_phase_end
else:
    lower_limit_frame = wakeup_phase_end
    
frame_range = list(range(128))
lineage_dfs = []

# Positions data
positions =[0,1]
#positions = positions_with_persisters
#positions = random.choices(positions_with_persisters, k=2) # Test with k number of positions with persisters
#positions = filtered_df['PositionIdx'].unique() # all positions
for position in positions:
    position_df = filtered_df[filtered_df['PositionIdx'] == position]
    # print('Position Idx:', position_df['PositionIdx'].unique())

    # Microchannels data
    # microchannels = [1] # test
    microchannels = position_df['ChannelIdx'].unique()  # all microchannels
    for channel in microchannels:
        channel_df = position_df[position_df['ChannelIdx'] == channel]
        # print('Channel Idx:', channel_df['ChannelIdx'].unique())

        # Filter data during drug treatment 
        if not has_wakeup_phase:
            drug_frm_range = channel_df[(channel_df['Frame'] >= stat_phase_end) & (channel_df['Frame'] <= drug_phase_end)]
        else:
            drug_frm_range = channel_df[(channel_df['Frame'] >= wakeup_phase_end) & (channel_df['Frame'] <= drug_phase_end)]

        # Only use BacteriaLineages & corresponding lin_id_str that were treated with drug       
        treated_lineages = drug_frm_range.loc[drug_frm_range['BacteriaLineage'].notna(), ['BacteriaLineage', 'lin_id_str', 'Frame', 'lin_id_str_long']] # use 'lin_id_str_long' for checking
        
        # Loop through frames in reverse order
        for frame in range(upper_limit_frame, lower_limit_frame, -1):
            previous_frame_treated_lineages = None
            if frame < upper_limit_frame:
                previous_frame_treated_lineages = treated_lineages[treated_lineages['Frame'] == frame + 1]['BacteriaLineage'].unique()

            frame_treated_lineages = treated_lineages[treated_lineages['Frame'] == frame]

            for index, row in frame_treated_lineages.iterrows():
                treated_BacteriaLineage = row['BacteriaLineage']
                treated_lin_id_str = row['lin_id_str']

                # Skip reconstruction if treated lineage is already present in the previous frame
                if previous_frame_treated_lineages is not None and treated_BacteriaLineage in previous_frame_treated_lineages:
                    continue
                #print('treated_lin_id_str', treated_lin_id_str)
                reconstructed_lineage_back = [treated_BacteriaLineage[:i] for i in range(len(treated_BacteriaLineage), 0, -1)]

                # Filter data during drug treatment
                filtered_BacteriaLineage_df = channel_df[channel_df['BacteriaLineage'].isin(reconstructed_lineage_back)]

                # Add treated 'lin_id_str' as unique_lineage_id to all data of reconstructed lineage
                filtered_BacteriaLineage_df['unique_lineage_id'] = treated_lin_id_str

                # Count number of divisions using lineage information
                num_div_exp_phase = filtered_BacteriaLineage_df[(filtered_BacteriaLineage_df['Frame'] >= exp_phase_start) & (
                            filtered_BacteriaLineage_df['Frame'] <= exp_phase_end)]['PreviousDivisionFrame'].isna().sum()
                num_div_grad_phase = filtered_BacteriaLineage_df[(filtered_BacteriaLineage_df['Frame'] >= grad_phase_start) & (
                            filtered_BacteriaLineage_df['Frame'] <= grad_phase_end)]['PreviousDivisionFrame'].isna().sum()
                num_div_stat_phase = filtered_BacteriaLineage_df[(filtered_BacteriaLineage_df['Frame'] >= stat_phase_start) & (
                            filtered_BacteriaLineage_df['Frame'] <= stat_phase_end)]['PreviousDivisionFrame'].isna().sum()
                
                if has_wakeup_phase:
                    num_div_wakeup_phase = filtered_BacteriaLineage_df[(filtered_BacteriaLineage_df['Frame'] >= wakeup_phase_start) & (
                            filtered_BacteriaLineage_df['Frame'] <= wakeup_phase_end)]['PreviousDivisionFrame'].isna().sum()
                else:
                    continue
                    
                num_div_drug_phase = filtered_BacteriaLineage_df[(filtered_BacteriaLineage_df['Frame'] >= stat_phase_start) & (
                            filtered_BacteriaLineage_df['Frame'] <= stat_phase_end)]['PreviousDivisionFrame'].isna().sum()

                filtered_BacteriaLineage_df['num_div_exp_phase'] = num_div_exp_phase
                filtered_BacteriaLineage_df['num_div_grad_phase'] = num_div_grad_phase
                filtered_BacteriaLineage_df['num_div_stat_phase'] = num_div_stat_phase
                if has_wakeup_phase:
                    filtered_BacteriaLineage_df['num_div_wakeup_phase'] = num_div_wakeup_phase
                else:
                    continue
                filtered_BacteriaLineage_df['num_div_drug_phase'] = num_div_drug_phase

                # Checkpoint: existence of lineages
                frames_full_lineage_exists = filtered_BacteriaLineage_df["Frame"].tolist()
                missing_frames_lineage = [frame for frame in frame_range if frame not in frames_full_lineage_exists]  # find missing frames in lineage data
            
                plt.plot(filtered_BacteriaLineage_df['Frame'], filtered_BacteriaLineage_df['Spinelength'], linestyle='-')

                # Append lineage_df to lineage_dfs list
                lineage_dfs.append(filtered_BacteriaLineage_df)

# Concatenate all lineage_dfs in list and make a master_df
df_susceptibles_merged = pd.concat(lineage_dfs, ignore_index=True)
#print(df_susceptibles_merged['cell_type'].unique())


# Classify susceptibles into related & unrelated susceptibles

persisters_uli = df_persisters['lin_id_str'].unique()
grouped_susceptibles = df_susceptibles_merged.groupby('unique_lineage_id')
# Function to assign cell type
def assign_cell_type(df):
    if df['lin_id_str'].isin(persisters_uli).any():
        df['cell_type'] = "related_susceptible"
    else:
        df['cell_type'] = "unrelated_susceptible"
    return df

# Apply the function to each group and return to ungrouped DataFrame
df_susceptibles_two_classes = grouped_susceptibles.apply(assign_cell_type).reset_index(drop=True)

#check 
df_susceptibles_two_classes['cell_type'].unique()


final_df = pd.concat([df_selections_merged, df_susceptibles_two_classes], ignore_index=True)




def derivative(y, frm_interval=1, log = False):
    if y.size > 2:
        y = np.log(y) if log else y
        x = np.arange(y.size) * frm_interval
        p = np.polyfit(x, y, 1) #simply 1st order polynomial fit / regression (numpy.polyfit)
        ddt = p[0] #slope of first order fit in python p[0]
    else: 
        ddt = np.nan 
    return ddt

# Define a function that calculates all the features above

# Time window use during rolling function
window = 5

def moving_rms(y):
    return np.sqrt(np.mean(y**2))

def calculations_windows(df):

    # CELL FEATURES
    
    df['red'] = pd.to_numeric(df['MeanIntensityRFP'])
    df['green'] = pd.to_numeric(df['MeanIntensityGFP'])
    df['fluo_ratio'] =  pd.to_numeric(df['green']/df['red']) # fluorescence ratio 
    df['GrowthRateSize'] = pd.to_numeric(df['GrowthRateSize'])
    df['SizeAtBirthSize'] = pd.to_numeric(df['SizeAtBirthSize'])
    df['Size'] = pd.to_numeric(df['Size'])
    df['GrowthRateFeretMax'] = pd.to_numeric(df['GrowthRateFeretMax'])
    df['FeretMax'] = pd.to_numeric(df['FeretMax'])
    df['GrowthRateSpinelength'] = pd.to_numeric(df['GrowthRateSpinelength'])
    df['SizeAtBirthSpinelength'] = pd.to_numeric(df['SizeAtBirthSpinelength'])
    df['Spinelength'] = pd.to_numeric(df['Spinelength'])
    df['GrowthRateSpineWidth'] = pd.to_numeric(df['GrowthRateSpineWidth'])
    df['SizeAtBirthSpineWidth'] = pd.to_numeric(df['SizeAtBirthSpineWidth'])
    df['SpineWidth'] = pd.to_numeric(df['SpineWidth'])
    df['GrowthRateLocalThickness'] = pd.to_numeric(df['GrowthRateLocalThickness'])
    df['SizeAtBirthLocalThickness'] = pd.to_numeric(df['SizeAtBirthLocalThickness'])
    df['LocalThickness'] = pd.to_numeric(df['LocalThickness'])
    
    # NORMALIZATIONS 
    
    # FlUO/LENGTH normalization 
    df['red_normalized_with_length']=  pd.to_numeric(df['MeanIntensityRFP'])/pd.to_numeric(df['Spinelength'])
    df['green_normalized_with_length']=  pd.to_numeric(df['MeanIntensityGFP'])/pd.to_numeric(df['Spinelength'])
    df['fluo_ratio_normalized_with_length']= pd.to_numeric(df['green']/df['red'])/pd.to_numeric(df['Spinelength'])
    
    # FlUO/SIZE normalization
    
    df['red_normalized_with_size']=  pd.to_numeric(df['MeanIntensityRFP'])/pd.to_numeric(df['Size'])
    df['green_normalized_with_size']= pd.to_numeric(df['MeanIntensityGFP'])/pd.to_numeric(df['Size'])
    df['fluo_ratio_normalized_with_size']= pd.to_numeric(df['green']/df['red'])/pd.to_numeric(df['Size'])

    # Min-Max Normalization
    
    df['MeanIntensityGFP_normalized'] = (df['MeanIntensityGFP'] - df['MeanIntensityGFP'].min()) / (df['MeanIntensityGFP'].max() - df['MeanIntensityGFP'].min())
    df['MeanIntensityRFP_normalized'] = (df['MeanIntensityRFP'] - df['MeanIntensityRFP'].min()) / (df['MeanIntensityRFP'].max() - df['MeanIntensityRFP'].min())
    
    # Z-Score Standardization
    df['MeanIntensityGFP_standardized'] = (df['MeanIntensityGFP'] - df['MeanIntensityGFP'].mean()) / df['MeanIntensityGFP'].std()
    df['MeanIntensityRFP_standardized'] = (df['MeanIntensityRFP'] - df['MeanIntensityRFP'].mean()) / df['MeanIntensityRFP'].std()

    df['GFP_RFP_ratio_normalized'] = (df['fluo_ratio'] - df['fluo_ratio'].min()) / (df['fluo_ratio'].max() - df['fluo_ratio'].min())
    df['GFP_RFP_ratio_standardized'] = (df['fluo_ratio'] - df['fluo_ratio'].mean()) / df['fluo_ratio'].std()

    
    # ABSOLUTE DELTAS ('dx','dy' missing)
    df[['dl']] = df.groupby('unique_lineage_id')[['Spinelength']].diff(periods=-1) # keep negative values as well so we can find the length drops associated to cell division
    df['length_drop'] = np.where(df['dl'] > 0.4, 1, 0)
    df[['dfluo_ratio']] = df.groupby('unique_lineage_id')[['fluo_ratio']].diff(periods=-1) # keep negative values as well so we can find the length drops associated to cell division

    # ABSOLUTE DELTAS
    abs_deltas = np.abs(df.groupby('unique_lineage_id')[['red', 'green', 'fluo_ratio', 'MeanIntensityGFP_normalized', 'MeanIntensityRFP_normalized', 'MeanIntensityGFP_standardized', 'MeanIntensityRFP_standardized', 'GFP_RFP_ratio_normalized', 'GFP_RFP_ratio_standardized','GrowthRateSize', 'SizeAtBirthSize', 'Size', 'GrowthRateFeretMax', 'FeretMax', 'GrowthRateSpinelength', 'SizeAtBirthSpinelength', 'Spinelength',  'GrowthRateSpineWidth', 'SizeAtBirthSpineWidth', 'SpineWidth', 'GrowthRateLocalThickness', 'SizeAtBirthLocalThickness', 'LocalThickness']].diff(periods=-1))
    
    # Assigning absolute deltas to the specified columns
    df[['dred','dgreen', 'dfluo_ratio_abs', 'dMeanIntensityGFP_normalized', 'dMeanIntensityRFP_normalized', 'dMeanIntensityGFP_standardized', 'dMeanIntensityRFP_standardized', 'dGFP_RFP_ratio_normalized', 'dGFP_RFP_ratio_standardized', 'dGrowthRateSize', 'dSizeAtBirthSize', 'dSize', 'dGrowthRateFeretMax', 'dFeretMax', 'dGrowthRateSpinelength', 'dSizeAtBirthSpinelength', 'dSpinelength', 'dGrowthRateSpineWidth', 'dSizeAtBirthSpineWidth', 'dSpineWidth', 'dGrowthRateLocalThickness', 'dSizeAtBirthLocalThickness', 'dLocalThickness']] = abs_deltas
    #df[['dl_abs', 'dred','dgreen', 'dfluo_ratio_abs', 'dGrowthRateSize', 'dSizeAtBirthSize', 'dSize', 'dGrowthRateFeretMax', 'dFeretMax', 'dGrowthRateSpinelength', 'dSizeAtBirthSpinelength', 'dSpinelength', 'dGrowthRateSpineWidth', 'dSizeAtBirthSpineWidth', 'dSpineWidth', 'dGrowthRateLocalThickness', 'dSizeAtBirthLocalThickness', 'dLocalThickness']] = np.abs(df.groupby('unique_lineage_id')[['Spinelength', 'red','green', 'fluo_ratio', 'GrowthRateSize', 'SizeAtBirthSize', 'Size', 'GrowthRateFeretMax', 'FeretMax', 'GrowthRateSpinelength', 'SizeAtBirthSpinelength', 'Spinelength', 'GrowthRateSpineWidth', 'SizeAtBirthSpineWidth', 'SpineWidth', 'GrowthRateLocalThickness', 'SizeAtBirthLocalThickness', 'LocalThickness']].diff(periods=-1))

    # DISPLACEMENT IN 2D
    #df['displacement'] = np.sqrt(np.array(df['dx'],dtype=np.float64)**2 + np.array(df['dy'],dtype=np.float64)**2)
    
    # DISPLACEMENT IN 1D
    #df['displacement'] = df['dy']
    #print("DataFrame shape:", df.shape)
    #print("Absolute deltas shape:", abs_deltas.shape)
    
    # Max value over window ROLLLING ('dx','dy' missing)
    #df['max_abs_dred'] = df.groupby('unique_lineage_id')['dred'].rolling(window, min_periods=window, center=True).apply(np.max).reset_index(0,drop=True)
    #df['max_abs_dgreen'] = df.groupby('unique_lineage_id')['dgreen'].rolling(window, min_periods=window, center=True).apply(np.max).reset_index(0,drop=True)
    #df['max_abs_dfluo_ratio'] = df.groupby('unique_lineage_id')['dfluo_ratio_abs'].rolling(window, min_periods=window, center=True).apply(np.max).reset_index(0,drop=True)
    #df['max_abs_dGrowthRateSize'] = df.groupby('unique_lineage_id')['dGrowthRateSize'].rolling(window, min_periods=window, center=True).apply(np.max).reset_index(0,drop=True)
    #df['max_abs_dSizeAtBirthSize'] = df.groupby('unique_lineage_id')['dSizeAtBirthSize'].rolling(window, min_periods=window, center=True).apply(np.max).reset_index(0,drop=True)
    #df['max_abs_dSize'] = df.groupby('unique_lineage_id')['dSize'].rolling(window, min_periods=window, center=True).apply(np.max).reset_index(0,drop=True)
    #df['max_abs_dGrowthRateFeretMax'] = df.groupby('unique_lineage_id')['dGrowthRateFeretMax'].rolling(window, min_periods=window, center=True).apply(np.max).reset_index(0,drop=True)
    #df['max_abs_dSizeAtBirthSpinelength'] = df.groupby('unique_lineage_id')['dSizeAtBirthSpinelength'].rolling(window, min_periods=window, center=True).apply(np.max).reset_index(0,drop=True)
    #df['max_abs_dGrowthRateSpinelength'] = df.groupby('unique_lineage_id')['dGrowthRateSpinelength'].rolling(window, min_periods=window, center=True).apply(np.max).reset_index(0,drop=True)
    #df['max_abs_dSpinelength'] = df.groupby('unique_lineage_id')['dSpinelength'].rolling(window, min_periods=window, center=True).apply(np.max).reset_index(0,drop=True)
    #df['max_abs_dGrowthRateSpineWidth'] = df.groupby('unique_lineage_id')['dGrowthRateSpineWidth'].rolling(window, min_periods=window, center=True).apply(np.max).reset_index(0,drop=True)
    #df['max_abs_dSizeAtBirthSpineWidth'] = df.groupby('unique_lineage_id')['dSizeAtBirthSpineWidth'].rolling(window, min_periods=window, center=True).apply(np.max).reset_index(0,drop=True)
    #df['max_abs_dSpineWidth'] = df.groupby('unique_lineage_id')['dSpineWidth'].rolling(window, min_periods=window, center=True).apply(np.max).reset_index(0,drop=True)
    #df['max_abs_dGrowthRateLocalThickness'] = df.groupby('unique_lineage_id')['dGrowthRateLocalThickness'].rolling(window, min_periods=window, center=True).apply(np.max).reset_index(0,drop=True)
    #df['max_abs_dSizeAtBirthLocalThickness'] = df.groupby('unique_lineage_id')['dSizeAtBirthLocalThickness'].rolling(window, min_periods=window, center=True).apply(np.max).reset_index(0,drop=True)
    #df['max_abs_dLocalThickness'] = df.groupby('unique_lineage_id')['dLocalThickness'].rolling(window, min_periods=window, center=True).apply(np.max).reset_index(0,drop=True)

    #df['rms_displacement'] = df.groupby('unique_lineage_id')['displacement'].rolling(window, min_periods=window, center=True).apply(moving_rms).reset_index(0,drop=True)
    
    # Average value over window ROLLLING ('dx','dy' missing)
    #df['av_red'] = df.groupby('unique_lineage_id')['red'].rolling(window, min_periods=window, center=True).apply(np.mean).reset_index(0,drop=True)
    #df['av_fluo_ratio'] = df.groupby('unique_lineage_id')['fluo_ratio'].rolling(window, min_periods=window, center=True).apply(np.mean).reset_index(0,drop=True)
    #df['av_green'] = df.groupby('unique_lineage_id')['green'].rolling(window, min_periods=window, center=True).apply(np.mean).reset_index(0,drop=True)
    #df['av_GrowthRateSize'] = df.groupby('unique_lineage_id')['GrowthRateSize'].rolling(window, min_periods=window, center=True).apply(np.mean).reset_index(0,drop=True)
    #df['av_SizeAtBirthSize'] = df.groupby('unique_lineage_id')['SizeAtBirthSize'].rolling(window, min_periods=window, center=True).apply(np.mean).reset_index(0,drop=True)
    #df['av_Size'] = df.groupby('unique_lineage_id')['Size'].rolling(window, min_periods=window, center=True).apply(np.mean).reset_index(0,drop=True)
    #df['av_GrowthRateFeretMax'] = df.groupby('unique_lineage_id')['GrowthRateFeretMax'].rolling(window, min_periods=window, center=True).apply(np.mean).reset_index(0,drop=True)
    #df['av_SizeAtBirthSpinelength'] = df.groupby('unique_lineage_id')['SizeAtBirthSpinelength'].rolling(window, min_periods=window, center=True).apply(np.mean).reset_index(0,drop=True)
    #df['av_GrowthRateSpinelength'] = df.groupby('unique_lineage_id')['GrowthRateSpinelength'].rolling(window, min_periods=window, center=True).apply(np.mean).reset_index(0,drop=True)
    #df['av_Spinelength'] = df.groupby('unique_lineage_id')['Spinelength'].rolling(window, min_periods=window, center=True).apply(np.mean).reset_index(0,drop=True)
    #df['av_GrowthRateSpineWidth'] = df.groupby('unique_lineage_id')['GrowthRateSpineWidth'].rolling(window, min_periods=window, center=True).apply(np.mean).reset_index(0,drop=True)
    #df['av_SizeAtBirthSpineWidth'] = df.groupby('unique_lineage_id')['SizeAtBirthSpineWidth'].rolling(window, min_periods=window, center=True).apply(np.mean).reset_index(0,drop=True)
    #df['av_SpineWidth'] = df.groupby('unique_lineage_id')['SpineWidth'].rolling(window, min_periods=window, center=True).apply(np.mean).reset_index(0,drop=True)
    #df['av_GrowthRateLocalThickness'] = df.groupby('unique_lineage_id')['GrowthRateLocalThickness'].rolling(window, min_periods=window, center=True).apply(np.mean).reset_index(0,drop=True)
    #df['av_SizeAtBirthLocalThickness'] = df.groupby('unique_lineage_id')['SizeAtBirthLocalThickness'].rolling(window, min_periods=window, center=True).apply(np.mean).reset_index(0,drop=True)
    #df['av_LocalThickness'] = df.groupby('unique_lineage_id')['LocalThickness'].rolling(window, min_periods=window, center=True).apply(np.mean).reset_index(0,drop=True)

    #add log transforms ROLLLING
    min_value = 1E-1 # to avoid error
    #df['log_max_abs_dl'] = np.log10(df['max_abs_dl']+ min_value)
    #df['log_max_abs_df1'] = np.log10(df['max_abs_dred']+ min_value)
    #df['log_max_abs_df2'] = np.log10(df['max_abs_dgreen']+ min_value)
    #df['log_displacement'] = np.log10(df['rms_displacement']+min_value)
    
    # LOG2 TRANSFORMATIONS
    #df['log2_fluo_ratio'] = np.log2(df['fluo_ratio']+ min_value)
    #df['log2_GFP_RFP_ratio_normalized'] = np.log2(df['GFP_RFP_ratio_normalized']+ min_value)
    #df['log2_GFP_RFP_ratio_standardized'] = np.log2(df['GFP_RFP_ratio_standardized']+ min_value)
    df['log2_fluo_ratio_abs'] = np.log2(df['dfluo_ratio_abs']+ min_value)
    
    # max value (not during rolling function)
    #max_values = df.groupby('unique_lineage_id')[['red','green', 'fluo_ratio', 'GrowthRateSize', 'SizeAtBirthSize', 'Size', 'GrowthRateFeretMax', 'FeretMax', 'GrowthRateSpinelength', 'SizeAtBirthSpinelength', 'Spinelength', 'GrowthRateSpineWidth', 'SizeAtBirthSpineWidth', 'SpineWidth', 'GrowthRateLocalThickness', 'SizeAtBirthLocalThickness', 'LocalThickness', 'dl', 'dred', 'dgreen', 'dfluo_ratio', 'dGrowthRateSize', 'dSizeAtBirthSize', 'dSize', 'dGrowthRateFeretMax', 'dFeretMax', 'dGrowthRateSpinelength', 'dSizeAtBirthSpinelength', 'dSpinelength', 'dGrowthRateSpineWidth', 'dSizeAtBirthSpineWidth', 'dSpineWidth', 'dGrowthRateLocalThickness', 'dSizeAtBirthLocalThickness', 'dLocalThickness']].max()

    # Renaming columns for clarity
    #max_values.columns = ['max_red','max_green', 'max_fluo_ratio', 'max_GrowthRateSize', 'max_SizeAtBirthSize', 'max_Size', 'max_GrowthRateFeretMax', 'max_FeretMax', 'max_GrowthRateSpinelength', 'max_SizeAtBirthSpinelength', 'max_Spinelength', 'max_GrowthRateSpineWidth', 'max_SizeAtBirthSpineWidth', 'max_SpineWidth', 'max_GrowthRateLocalThickness', 'max_SizeAtBirthLocalThickness', 'max_LocalThickness', 'max_dl', 'max_dred', 'max_dgreen', 'max_dfluo_ratio', 'max_dGrowthRateSize', 'max_dSizeAtBirthSize', 'max_dSize', 'max_dGrowthRateFeretMax', 'max_dFeretMax', 'max_dGrowthRateSpinelength', 'max_dSizeAtBirthSpinelength', 'max_dSpinelength', 'max_dGrowthRateSpineWidth', 'max_dSizeAtBirthSpineWidth', 'max_dSpineWidth', 'max_dGrowthRateLocalThickness', 'max_dSizeAtBirthLocalThickness', 'max_dLocalThickness']

    # Merging the maximum values back to the original DataFrame based on 'id_cell'
    #df['rms_displacement'] = df.groupby('unique_lineage_id')['displacement'].rolling(window, min_periods=window, center=True).apply(moving_rms).reset_index(0,drop=True)
 
    # mean value (not during rolling function)
    mean_values = df.groupby('unique_lineage_id')[['red','green', 'fluo_ratio', 'GrowthRateSize', 'SizeAtBirthSize', 'Size', 'GrowthRateFeretMax', 'FeretMax', 'GrowthRateSpinelength', 'SizeAtBirthSpinelength', 'Spinelength', 'GrowthRateSpineWidth', 'SizeAtBirthSpineWidth', 'SpineWidth', 'GrowthRateLocalThickness', 'SizeAtBirthLocalThickness', 'LocalThickness', 'dl', 'dred', 'dgreen', 'dfluo_ratio', 'dGrowthRateSize', 'dSizeAtBirthSize', 'dSize', 'dGrowthRateFeretMax', 'dFeretMax', 'dGrowthRateSpinelength', 'dSizeAtBirthSpinelength', 'dSpinelength', 'dGrowthRateSpineWidth', 'dSizeAtBirthSpineWidth', 'dSpineWidth', 'dGrowthRateLocalThickness', 'dSizeAtBirthLocalThickness', 'dLocalThickness']].mean()
    mean_columns = ['mean_red','mean_green', 'mean_fluo_ratio', 'mean_GrowthRateSize', 'mean_SizeAtBirthSize', 'mean_Size', 'mean_GrowthRateFeretMax', 'mean_FeretMax', 'mean_GrowthRateSpinelength', 'mean_SizeAtBirthSpinelength', 'mean_Spinelength', 'mean_GrowthRateSpineWidth', 'mean_SizeAtBirthSpineWidth', 'mean_SpineWidth', 'mean_GrowthRateLocalThickness', 'mean_SizeAtBirthLocalThickness', 'mean_LocalThickness', 'mean_dl', 'mean_dred', 'mean_dgreen', 'mean_dfluo_ratio', 'mean_dGrowthRateSize', 'mean_dSizeAtBirthSize', 'mean_dSize', 'mean_dGrowthRateFeretMax', 'mean_dFeretMax', 'mean_dGrowthRateSpinelength', 'mean_dSizeAtBirthSpinelength', 'mean_dSpinelength', 'mean_dGrowthRateSpineWidth', 'mean_dSizeAtBirthSpineWidth', 'mean_dSpineWidth', 'mean_dGrowthRateLocalThickness', 'mean_dSizeAtBirthLocalThickness', 'mean_dLocalThickness']
    mean_values.columns = mean_columns
    
    # DERIVATIVES
    #frm_interval = 10./60 #frame interval in minutes
    #gamma = 0.0015 #protein degradation constant in 1/min
    
    #lin_mu_spine = df.groupby('unique_lineage_id')['Spinelength'].rolling(window, min_periods=3, center=True).apply(derivative, kwargs={'frm_interval':frm_interval,'log':True}).reset_index(0,drop=True)
    #df['lin_mu_spine'] = lin_mu_spine

    #lin_mu_feret = df.groupby('unique_lineage_id')['FeretMax'].rolling(window, min_periods=3, center=True).apply(derivative, kwargs={'frm_interval':frm_interval,'log':True}).reset_index(0,drop=True)
    #df['lin_mu_feret'] = lin_mu_feret
    
    #lin_ddt_gfp = df.groupby('unique_lineage_id')['MeanIntensityGFP'].rolling(window, min_periods=5, center=True).apply(derivative, kwargs={'frm_interval':frm_interval}).reset_index(0,drop=True)
    #df['lin_ddt_gfp'] = lin_ddt_gfp
    
    #lin_ddt_rfp = df.groupby('unique_lineage_id')['MeanIntensityRFP'].rolling(window, min_periods=5, center=True).apply(derivative, kwargs={'frm_interval':frm_interval}).reset_index(0,drop=True)
    #df['lin_ddt_rfp'] = lin_ddt_rfp

    #lin_ddt_timer = df.groupby('unique_lineage_id')['fluo_ratio'].rolling(window, min_periods=5, center=True).apply(derivative, kwargs={'frm_interval':frm_interval}).reset_index(0,drop=True)
    #df['lin_ddt_timer'] = lin_ddt_timer
    
    
    #df['lin_promoter_act_G'] = (gamma + df['lin_mu_spine'] ) * df['MeanIntensityGFP'] + df['lin_ddt_gfp']
    #df['lin_promoter_act_R'] = (gamma + df['lin_mu_spine'] ) * df['MeanIntensityRFP'] + df['lin_ddt_rfp']
    
    #df = df.merge(max_values, on='unique_lineage_id', how='left').merge(mean_values, on='unique_lineage_id', how='left')
    
    overlapping_columns = [col for col in df.columns if col in mean_values.columns]
    
    if overlapping_columns: 
        df = df.merge(mean_values, on='unique_lineage_id', how='left') # 
        return df
    else:
        return df


calc_final_df = calculations_windows(final_df)

# Save calc_final_df as CSV 
#file_path = proj_dir / 'calc_final_df.csv'
#calc_final_df.to_csv(file_path, index=False)  # Set index=False to exclude row indices in the CSV file


# Initialize list to hold group DataFrames
group_dfs = []

# Iterate over each group in calc_final_df
for group_name, group_df in calc_final_df.groupby('unique_lineage_id'):
    # Filter the DataFrame based on the condition on the "Frame" column
    exp_phase = group_df[(group_df['Frame'] >= exp_phase_start) & (group_df['Frame'] <= exp_phase_end)]
    grad_phase = group_df[(group_df['Frame'] >= grad_phase_start) & (group_df['Frame'] <= grad_phase_end)]
    stat_phase = group_df[(group_df['Frame'] >= stat_phase_start) & (group_df['Frame'] <= stat_phase_end)]
    drug_phase = group_df[(group_df['Frame'] >= drug_phase_start) & (group_df['Frame'] <= drug_phase_end)]
    recov_phase = group_df[(group_df['Frame'] >= recov_phase_start) & (group_df['Frame'] <= recov_phase_end)]

    # Calculate the number of 1's in the 'length_drop' column for each phase
    num_div_exp_phase_length_drop = exp_phase['length_drop'].sum()
    num_div_grad_phase_length_drop = grad_phase['length_drop'].sum()
    num_div_stat_phase_length_drop = stat_phase['length_drop'].sum()
    num_div_drug_phase_length_drop = drug_phase['length_drop'].sum()

    # Ensure the sum is zero if no divisions occurred
    num_div_exp_phase_length_drop = num_div_exp_phase_length_drop if num_div_exp_phase_length_drop > 0 else 0
    num_div_grad_phase_length_drop = num_div_grad_phase_length_drop if num_div_grad_phase_length_drop > 0 else 0
    num_div_stat_phase_length_drop = num_div_stat_phase_length_drop if num_div_stat_phase_length_drop > 0 else 0
    num_div_drug_phase_length_drop = num_div_drug_phase_length_drop if num_div_drug_phase_length_drop > 0 else 0

    # List of phase dataframes and corresponding phase names
    phases = [
        ('exp', exp_phase),
        ('grad', grad_phase),
        ('stat', stat_phase),
        ('drug', drug_phase),
        ('recov', recov_phase)
    ]
    
    # Step 1: Filter the rows where 'length_drop' is 1
    filtered_df = group_df[group_df['length_drop'] == 1]
    
    # Step 2: Further filter to get 'Frame' values between 45 and 93
    frame_values = filtered_df[(filtered_df['Frame'] >= stat_phase_start) & (filtered_df['Frame'] <= drug_phase_start)]
    
    # Step 3: Sort the frame values
    sorted_frames = frame_values['Frame'].sort_values()
    
    # Step 4: Find the next immediate value and calculate the difference
    time_first_div_after_stat = None
    if not sorted_frames.empty: 
        last_div_stat = sorted_frames.iloc[-1]
        print('frame last_div_stat', last_div_stat)
    
        # Find the next immediate value after last_value_in_range
        first_div_after_stat = filtered_df[filtered_df['Frame'] > last_div_stat]['Frame'].min()
        
        if pd.notna(first_div_after_stat):
            spinelength_at_last_div_stat = filtered_df[filtered_df['Frame']==last_div_stat]['Spinelength']
            spinelength_first_div_after_stat = filtered_df[filtered_df['Frame']==first_div_after_stat]['Spinelength']
            spinelength_at_drug_phase_start = filtered_df[filtered_df['Frame']==drug_phase_start]['Spinelength']
            difference = first_div_after_stat - last_div_stat # Number of frames in between last division in stationary phase and first division after stat phase
            cell_cycle_progression_time_percent = ((drug_phase_start - last_div_stat) * 100) / difference  # Cell cycle progression (percentage) when drug hits
            #time_first_div_after_stat = first_div_after_stat * 10/60
            difference_length =  spinelength_first_div_after_stat - spinelength_at_last_div_stat
            cell_cycle_progression_length_percent = ((spinelength_at_drug_phase_start - spinelength_at_last_div_stat) * 100)/difference_length
            
            time_first_div_after_stat = first_div_after_stat * 10 / 60
        else:
            difference = None
            difference_length = None
            cell_cycle_progression_time_percent = None
            cell_cycle_progression_length_percent= None
            first_div_after_stat = None
    else:
        last_div_stat = None
        first_div_after_stat = None
        difference = None
        difference_length = None
        cell_cycle_progression_time_percent = None
        cell_cycle_progression_length_percent= None
    
    # Initialize columns in group_df to store results
    group_df['time_first_div_after_stat'] = time_first_div_after_stat
    group_df['cell_cycle_progression_time_percent'] = cell_cycle_progression_time_percent
    group_df['cell_cycle_progression_length_percent'] = cell_cycle_progression_length_percent

    for phase_name, _ in phases:
        group_df[f'last_length_drop_{phase_name}_time'] = None
        group_df[f'first_length_drop_{phase_name}_time'] = None
    
    # Dictionary to hold the last length drop times of each phase
    last_length_drop_times = {}
    
    # Loop through each phase dataframe
    for phase_name, phase_df in phases: 
        # Filter rows where 'length_drop' is equal to 1
        length_drop_phase = phase_df[phase_df['length_drop'] == 1]
        
        # Check if there are any rows meeting the condition
        if not length_drop_phase.empty:
            # Get the maximum (last) and minimum (first) values from the "Time" column
            last_length_drop_time = length_drop_phase['Time'].max()
            first_length_drop_time = length_drop_phase['Time'].min()
            
            # Store the results in the group_df dataframe
            group_df[f'last_length_drop_{phase_name}_time'] = last_length_drop_time
            group_df[f'first_length_drop_{phase_name}_time'] = first_length_drop_time
            
            # Store the last length drop time in the dictionary
            last_length_drop_times[phase_name] = last_length_drop_time
            
            # Calculate the difference with the previous phase's last_length_drop_time
            if phase_name != 'exp':  # No previous phase for exp
                prev_phase_name = phases[phases.index((phase_name, phase_df)) - 1][0]
                if prev_phase_name in last_length_drop_times:
                    lag_time = first_length_drop_time - last_length_drop_times[prev_phase_name]
                    group_df[f'lag_{phase_name}'] = lag_time
                    group_df[f'adaptability_{phase_name}'] = 1 / lag_time
        else:
            print(f"No rows with 'length_drop' equal to 1 found in {phase_name} phase.")
    
    # Issue should be in following code:
    group_df['num_div_exp_phase_length_drop'] = num_div_exp_phase_length_drop
    group_df['num_div_grad_phase_length_drop'] = num_div_grad_phase_length_drop
    group_df['num_div_stat_phase_length_drop'] = num_div_stat_phase_length_drop
    group_df['num_div_drug_phase_length_drop'] = num_div_drug_phase_length_drop
    
    # Append the modified group DataFrame to the list
    group_dfs.append(group_df)

# Concatenate all DataFrames in group_dfs into a single DataFrame
calc_final_df_right_count_count_division = pd.concat(group_dfs, ignore_index=True)


# Save calc_final_df as CSV 
file_path = results_dir / 'df_final_produced_on_2024-07-17b.csv'
calc_final_df_right_count_count_division.to_csv(file_path, index=False)  # Set index=False to exclude row indices in the CSV file


# LAG & ADAPTABILITY

# Function to add number of unique 'unique_lineage_id' for each 'cell_type'
def add_unique_count(data, ax, variable):
    cell_types = data['cell_type'].unique()
    for cell_type in cell_types:
        unique_count = data[data['cell_type'] == cell_type]['unique_lineage_id'].nunique()
        x_pos = list(cell_types).index(cell_type)
        ax.text(x_pos, data[variable].max(), f'n={unique_count}', 
                horizontalalignment='center', size='medium', color='black', weight='semibold')

# Columns to plot
columns_to_plot = [
    'last_length_drop_exp_time',
    'first_length_drop_exp_time',
    'last_length_drop_grad_time',
    'first_length_drop_grad_time',
    'last_length_drop_stat_time',
    'first_length_drop_stat_time',
    'last_length_drop_drug_time',
    'first_length_drop_drug_time',
    'last_length_drop_recov_time',
    'first_length_drop_recov_time',
    'lag_grad',
    'adaptability_grad',
    'lag_stat',
    'adaptability_stat',
    'lag_drug',
    'adaptability_drug',
    'lag_recov',
    'adaptability_recov'
]

# Ensure 'cell_type' and 'unique_lineage_id' are included for counting
columns_to_plot = ['cell_type', 'unique_lineage_id'] + columns_to_plot
df_to_plot = calc_final_df_right_count_count_division[columns_to_plot]

# Create the figure with more subplots (5 rows by 4 columns)
fig, axes = plt.subplots(nrows=5, ncols=4, figsize=(20, 25))
axes = axes.flatten()

# Plot each variable
for idx, column in enumerate(columns_to_plot[2:]):  # Skip 'cell_type' and 'unique_lineage_id' for plotting
    ax = axes[idx]
    sns.boxplot(x='cell_type', y=column, data=df_to_plot, ax=ax)
    ax.set_title(column)
    ax.set_xlabel('')
    ax.set_ylabel('')

    # Add unique lineage ID count information
    add_unique_count(df_to_plot, ax, column)

plt.tight_layout()
plt.savefig(results_dir / 'boxplot_lag_adaptability_phases.svg', format='svg', dpi=200, bbox_inches='tight', pad_inches=0.2)




# Filter only 'persister' and 'non-persister' from the 'cell_type' column
only_per_and_nonper_df = calc_final_df_right_count_count_division[
    calc_final_df_right_count_count_division['cell_type'].isin(['related_susceptible', 'unrelated_susceptible',
       'susceptible_sister', 'persister'])]

# Calculate number of unique lineage_id for each cell_type
unique_lineage_counts = only_per_and_nonper_df.groupby('cell_type')['unique_lineage_id'].nunique()

# Create a single boxplot for 'last_length_drop_frame' across cell types
plt.figure(figsize=(6, 6))

# Define boxplot properties
boxprops = dict(linewidth=2)
medianprops = dict(color='red', linewidth=2)

# Generate data for boxplot (last_length_drop_frame)
data = [only_per_and_nonper_df[only_per_and_nonper_df['cell_type'] == cell_type]['last_length_drop_stat_time'].dropna()
        for cell_type in unique_lineage_counts.index]

# Create boxplot and capture the artists for customization
bp = plt.boxplot(data, labels=unique_lineage_counts.index, boxprops=boxprops, medianprops=medianprops)

# Annotate each box with the count of unique lineage_id values
for i, box in enumerate(bp['boxes']):
    cell_type = unique_lineage_counts.index[i]
    num_unique_lineage_ids = unique_lineage_counts[cell_type]
    plt.annotate(f'n = {num_unique_lineage_ids}', xy=(box.get_xdata().mean(), box.get_ydata().max()), 
                 xytext=(10, 10), textcoords='offset points', ha='center', fontsize=14)

# Customize plot
plt.title('Time of last cell division before drug tratment ', fontsize=14)
plt.xlabel('Cell Type', fontsize=18)
plt.ylabel('Time (hrs)', fontsize=18)
plt.xticks(fontsize=16)  # Adjusted fontsize for x-axis labels
plt.yticks(fontsize=16)  # Adjusted fontsize for y-axis labels

# Save the plot as an SVG file
plt.tight_layout()
plt.savefig(results_dir/'boxplot_last_length_drop_time.svg', format='svg', dpi=200, bbox_inches='tight', pad_inches=0.2)


# Drop rows with NaN values in the relevant columns
cleaned_data = calc_final_df_right_count_count_division.dropna(subset=['last_length_drop_stat_time', 'time_first_div_after_stat'])

# Create scatter plots with line of best fit and correlation coefficient
g = sns.lmplot(data=cleaned_data, x='last_length_drop_stat_time', y='time_first_div_after_stat',
               col='cell_type', hue='cell_type', height=5)

# Add titles to each plot
g.set_titles("{col_name}")

# Set axis labels
g.set_axis_labels("Last division during stat.", "First division after stat.")

# Calculate and display Pearson correlation coefficient on each plot
for ax, (_, subdata) in zip(g.axes.flat, cleaned_data.groupby('cell_type')):
    corr_coef = stats.pearsonr(subdata['last_length_drop_stat_time'], subdata['time_first_div_after_stat'])[0]
    ax.annotate(f"Pearson correlation = {corr_coef:.2f}", xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12, color='blue')

plt.tight_layout()
plt.savefig(results_dir/'plt_firstlast_division_afterduring_stat.svg', format='svg', dpi=200, bbox_inches='tight', pad_inches=0.2)


# Filter only 'persister' and 'non-persister' from the 'cell_type' column
only_per_and_nonper_df = calc_final_df_right_count_count_division[
    calc_final_df_right_count_count_division['cell_type'].isin(['related_susceptible', 'unrelated_susceptible',
       'susceptible_sister', 'persister'])]

# Calculate number of unique lineage_id for each cell_type
unique_lineage_counts = only_per_and_nonper_df.groupby('cell_type')['unique_lineage_id'].nunique()

# Create a single boxplot for 'last_length_drop_frame' across cell types
plt.figure(figsize=(6, 6))

# Define boxplot properties
boxprops = dict(linewidth=2)
medianprops = dict(color='red', linewidth=2)

# Generate data for boxplot (last_length_drop_frame)
data = [only_per_and_nonper_df[only_per_and_nonper_df['cell_type'] == cell_type]['cell_cycle_progression_time_percent'].dropna()
        for cell_type in unique_lineage_counts.index]

# Create boxplot and capture the artists for customization
bp = plt.boxplot(data, labels=unique_lineage_counts.index, boxprops=boxprops, medianprops=medianprops)

# Annotate each box with the count of unique lineage_id values
for i, box in enumerate(bp['boxes']):
    cell_type = unique_lineage_counts.index[i]
    num_unique_lineage_ids = unique_lineage_counts[cell_type]
    plt.annotate(f'n = {num_unique_lineage_ids}', xy=(box.get_xdata().mean(), box.get_ydata().max()), 
                 xytext=(10, 10), textcoords='offset points', ha='center', fontsize=14)

# Customize plot
plt.title('Cell cycle progression at drug treatment (time)', fontsize=14)
plt.xlabel('Cell Type', fontsize=18)
plt.ylabel('Progress to division (%)', fontsize=18)
plt.xticks(fontsize=16)  # Adjusted fontsize for x-axis labels
plt.yticks(fontsize=16)  # Adjusted fontsize for y-axis labels

# Save the plot as an SVG file
plt.tight_layout()
plt.savefig(results_dir/'boxplot_cell_cycle_progression_time_percent.svg', format='svg', dpi=200, bbox_inches='tight', pad_inches=0.2)


plt.figure(figsize=(6, 6))

# Generate data for boxplot (last_length_drop_frame)
data_cell_cycle_length = [only_per_and_nonper_df[only_per_and_nonper_df['cell_type'] == cell_type]['cell_cycle_progression_length_percent'].dropna()
        for cell_type in unique_lineage_counts.index]

# Create boxplot and capture the artists for customization
bp = plt.boxplot(data_cell_cycle_length, labels=unique_lineage_counts.index, boxprops=boxprops, medianprops=medianprops)

# Annotate each box with the count of unique lineage_id values
for i, box in enumerate(bp['boxes']):
    cell_type = unique_lineage_counts.index[i]
    num_unique_lineage_ids = unique_lineage_counts[cell_type]
    plt.annotate(f'n = {num_unique_lineage_ids}', xy=(box.get_xdata().mean(), box.get_ydata().max()), 
                 xytext=(10, 10), textcoords='offset points', ha='center', fontsize=14)

# Customize plot
plt.title('Cell cycle progression at drug treatment (length)', fontsize=14)
plt.xlabel('Cell Type', fontsize=18)
plt.ylabel('Progress to division (%)', fontsize=18)
plt.xticks(fontsize=16)  # Adjusted fontsize for x-axis labels
plt.yticks(fontsize=16)  # Adjusted fontsize for y-axis labels

# Save the plot as an SVG file
plt.tight_layout()
plt.savefig(results_dir/'boxplot_cell_cycle_progression_length_percent.svg', format='svg', dpi=200, bbox_inches='tight', pad_inches=0.2)



# Filter only 'persister' and 'non-persister' from the 'cell_type' column
only_per_and_nonper_df = calc_final_df_right_count_count_division[
    calc_final_df_right_count_count_division['cell_type'].isin(['related_susceptible', 'unrelated_susceptible',
       'susceptible_sister', 'persister'])]

# Calculate number of unique lineage_id for each cell_type
unique_lineage_counts = only_per_and_nonper_df.groupby('cell_type')['unique_lineage_id'].nunique()

# Create a single boxplot for 'last_length_drop_frame' across cell types
plt.figure(figsize=(6, 6))

# Define boxplot properties
boxprops = dict(linewidth=2)
medianprops = dict(color='red', linewidth=2)

# Generate data for boxplot (last_length_drop_frame)
data = [only_per_and_nonper_df[only_per_and_nonper_df['cell_type'] == cell_type]['time_first_div_after_stat'].dropna()
        for cell_type in unique_lineage_counts.index]

# Create boxplot and capture the artists for customization
bp = plt.boxplot(data, labels=unique_lineage_counts.index, boxprops=boxprops, medianprops=medianprops)

# Annotate each box with the count of unique lineage_id values
for i, box in enumerate(bp['boxes']):
    cell_type = unique_lineage_counts.index[i]
    num_unique_lineage_ids = unique_lineage_counts[cell_type]
    plt.annotate(f'n = {num_unique_lineage_ids}', xy=(box.get_xdata().mean(), box.get_ydata().max()), 
                 xytext=(10, 10), textcoords='offset points', ha='center', fontsize=14)

# Customize plot
plt.title('Time of first division after stationary phase', fontsize=14)
plt.xlabel('Cell Type', fontsize=18)
plt.ylabel('Time (hrs)', fontsize=18)
plt.xticks(fontsize=16)  # Adjusted fontsize for x-axis labels
plt.yticks(fontsize=16)  # Adjusted fontsize for y-axis labels

# Save the plot as an SVG file
plt.tight_layout()
plt.savefig(results_dir/'boxplot_time_first_div_after_stat.svg', format='svg', dpi=200, bbox_inches='tight', pad_inches=0.2)



# Filter only 'persister' and 'non-persister' from the 'cell_type' column
# Filter only 'persister' and 'non-persister' from the 'cell_type' column
only_per_and_nonper_df = calc_final_df_right_count_count_division[calc_final_df_right_count_count_division['cell_type'].isin(['related_susceptible', 'unrelated_susceptible', 'susceptible_sister', 'persister'])]

# Calculate number of unique lineage_id for each cell_type
unique_lineage_counts = only_per_and_nonper_df.groupby('cell_type')['unique_lineage_id'].nunique()

# Create subplots for each cell_type
fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3.5))

# Define boxplot properties
boxprops = dict(linewidth=2)
medianprops = dict(color='red', linewidth=2)

# Iterate over each phase and corresponding axis
for ax, phase, title in zip(axes, ['num_div_exp_phase_length_drop', 'num_div_grad_phase_length_drop', 
                                   'num_div_stat_phase_length_drop', 'num_div_drug_phase_length_drop'], ['Exponential', 'Gradual to Stationary', 'Stationary', 'Drug']):
    
    # Get data for boxplot
    data = [only_per_and_nonper_df[only_per_and_nonper_df['cell_type'] == cell_type][phase].dropna() for cell_type in unique_lineage_counts.index]
    
    # Create boxplot and capture the artists for customization
    bp = ax.boxplot(data, labels=unique_lineage_counts.index, boxprops=boxprops, medianprops=medianprops)
    
    # Annotate each box with the count of unique lineage_id values
    for i, box in enumerate(bp['boxes']):
        cell_type = unique_lineage_counts.index[i]
        num_unique_lineage_ids = unique_lineage_counts[cell_type]
        ax.annotate(f'n = {num_unique_lineage_ids}', xy=(box.get_xdata().mean(), box.get_ydata().max()), 
                    xytext=(10, 10), textcoords='offset points', ha='center', fontsize=10)
    
    ax.set_xlabel('Cell Type', fontsize=10)  # Adjusted x-axis label fontsize
    ax.set_ylabel('Number of Cell Divisions', fontsize=10)  # Adjusted y-axis label fontsize
    ax.set_title(title, fontsize=10)  # Adjusted title fontsize
    
    # Set custom tick labels and tick parameters for x-axis
    ax.set_xticklabels(unique_lineage_counts.index, fontsize=10, rotation=45)  # Rotate and adjust fontsize of x-axis tick labels
    ax.tick_params(axis='x', which='both', bottom=False, top=False)  # Hide x-axis ticks for better appearance
    
    # Set custom tick labels and tick parameters for y-axis
    ax.set_yticklabels(ax.get_yticks(), fontsize=10)  # Adjust fontsize of y-axis tick labels
    ax.tick_params(axis='y', which='both', left=False, right=False)  # Hide y-axis ticks for better appearance

# Set common title
#fig.suptitle('Distribution of Cell Divisions Across Phases', fontsize=18)

# Adjust layout
plt.tight_layout()
plt.savefig(results_dir/'bp_cell_div_phases.svg', format='svg', dpi=200, bbox_inches='tight', pad_inches=0.2)


# Define the experimental phase ranges

phase_ranges = {
    'exp': (0, 30),
    'grad': (30, 45),
    'stat': (45, 93),
    'drug': (93, 111)
}

# Create subplots for each experimental phase
fig, axes = plt.subplots(nrows=1, ncols=len(phase_ranges), figsize=(15, 5))

# Iterate over each phase and plot the boxplot
for i, (phase_name, (start_frame, end_frame)) in enumerate(phase_ranges.items()):
    phase_data = calc_final_df_right_count_count_division[(calc_final_df_right_count_count_division['Frame'] >= start_frame) & (calc_final_df_right_count_count_division['Frame'] <= end_frame)]
    
    # Create boxplot for 'GR' grouped by 'cell_type'
    sns.boxplot(x='cell_type', y='GrowthRateSpinelength', data=phase_data, ax=axes[i])
    axes[i].set_title(f'Phase: {phase_name}')
    axes[i].set_xlabel('Cell Type')
    axes[i].set_ylabel('GrowthRateSpinelength')

# Adjust layout and display the plots
plt.tight_layout()
plt.savefig(results_dir/'plt_growthrate_length.svg', format='svg', dpi=200, bbox_inches='tight', pad_inches=0.2)


# Filter only 'persister' and 'non-persister' from the 'cell_type' column at Frame 93
only_per_and_nonper_atfr93_df = calc_final_df_right_count_count_division[
    (calc_final_df_right_count_count_division['cell_type'].isin(['related_susceptible', 'unrelated_susceptible',
       'susceptible_sister', 'persister'])) & 
    (calc_final_df_right_count_count_division['Frame'] == 93)
]

# Calculate number of unique lineage_id for each cell_type at Frame 93
unique_lineage_counts_fr93 = only_per_and_nonper_atfr93_df.groupby('cell_type')['unique_lineage_id'].nunique()

# Create subplots for each cell_type
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 6))

# Define boxplot properties
boxprops = dict(linewidth=2)
medianprops = dict(color='red', linewidth=2)

# Create boxplots for each cell_type at Frame 93
for ax, phase, title in zip(axes, ['RelativeCoordY', 'BoundsYMin', 'BoundsYMax'], ['RelativeCoordY', 'BoundsYMin', 'BoundsYMax']):
    # Get data for boxplot
    data = [only_per_and_nonper_atfr93_df[only_per_and_nonper_atfr93_df['cell_type'] == cell_type][phase].dropna() for cell_type in unique_lineage_counts_fr93.index]
    
    # Create boxplot and capture the artists for customization
    bp = ax.boxplot(data, labels=unique_lineage_counts_fr93.index, boxprops=boxprops, medianprops=medianprops)
    
    # Annotate each box with the count of unique lineage_id values at Frame 93
    for i, box in enumerate(bp['boxes']):
        cell_type = unique_lineage_counts_fr93.index[i]
        num_unique_lineage_ids = unique_lineage_counts_fr93[cell_type]
        ax.annotate(f'n = {num_unique_lineage_ids}', xy=(box.get_xdata().mean(), box.get_ydata().max()), 
                    xytext=(10, 10), textcoords='offset points', ha='center', fontsize=10)
    
    ax.set_xlabel('Cell Type', fontsize=12)
    ax.set_ylabel('Coordinates', fontsize=12)
    ax.set_title(title)  

    # Invert y-axis direction
    ax.invert_yaxis()
    
    # Calculate midpoint of y-axis
    y_mid = (ax.get_ylim()[0] + ax.get_ylim()[1]) / 2
    
    # Draw blue dotted line at the midpoint
    ax.axhline(y_mid, color='blue', linestyle='--')

# Set common title
fig.suptitle('Location of cells along the trap right before drug', fontsize=14)

# Adjust layout
plt.tight_layout()
plt.savefig(results_dir/'bp_cells_y_coor_frm93.svg', format='svg', dpi=200, bbox_inches='tight', pad_inches=0.2)



# This version is doing exactly what pablo wants
df_stat_phase = calc_final_df_right_count_count_division.copy()
# Filter the DataFrame based on the range of 'Frame' values
df_stat_phase = df_stat_phase[(df_stat_phase['Frame'] >= 30) & (df_stat_phase['Frame'] <= 93)]
# Filter the DataFrame based on the initial range of 'frames'
lower_limit_frame = 30
upper_limit_frame = 93

path = results_dir

# Make a directory to store subset dataframes
if not os.path.exists("df_subsets_2022-05-26_produced_2024-07-17b"):
    os.makedirs("df_subsets_2022-05-26_produced_2024-07-17b")

# Define the frame interval and initial window size
step_size =  1
min_window_size = 6
max_window_size = upper_limit_frame - lower_limit_frame # Maximum value for 'frames' in your filter

for window_size in range(min_window_size, max_window_size, step_size):
    frame_range = [frms for frms in range(upper_limit_frame, upper_limit_frame - window_size, -step_size)]
    print(frame_range)
    frms_min= min(frame_range)
    #print(frms_min)
    frms_max= max(frame_range)
    #print(frms_max)
    
    df_subset = df_stat_phase[df_stat_phase['Frame'].isin(frame_range)]
    #print([df_subset['frames']])

    result = df_subset.groupby('cell_type')['unique_lineage_id'].nunique().reset_index(name='unique_id_count')
    # Display the result
    print(result)
    
    df_subset= calculations_windows(df_subset)
    
    # Count the number of NaN values in the 'PreviousDivisionFrame' column for each group
    num_div_window_counts = df_subset.groupby('unique_lineage_id')['length_drop'].sum()
    
    # Merge the unique counts back into the subset DataFrame
    df_subset = df_subset.merge(num_div_window_counts, on='unique_lineage_id', suffixes=('', '_div_window_counts'))

    # Count the number of length drops (>10) for each 'id_cell'
    #num_len_drops = df_subset[df_subset['dl'] > 10].groupby('id_cell').size().rename('num_len_drops')
    #print(num_len_drops)

    result2 = df_subset.groupby('cell_type')['unique_lineage_id'].nunique().reset_index(name='unique_lineage_id_window_count')
    # Display the result
    print(result2)
    

    # Define the filename for the CSV file
    file_name = f"df_subsets_2022-05-26_produced_2024-07-17b/df_subset_{frms_min}_to_{frms_min + window_size - 1}.csv"

    # Save the df_subset as a CSV file in the directory
    #df_subset.to_csv(file_name, index=False)
    #df_subset.to_excel(file_name, index=False)
    df_subset.to_csv(results_dir / file_name, index=False) #save file

    # Print a message to confirm the file has been saved
    print(f"Saved {file_name}")
    

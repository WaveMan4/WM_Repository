import copy
import csv
import pandas as pd
import sklearn
import numpy as np
import matplotlib

def broLog_to_df(filename):
    """
    Summary: converts Zeek/Bro log to a Pandas DataFrame. Takes input log filename and column headers, returns dataframe with labeled columns
    """
    data = []
    col_names = []
    row_count = 0
    with open(filename) as tsvfile: # read in TSV
            reader = csv.reader(tsvfile, delimiter='\t')
            for row in reader:
                row_count += 1
                if (row_count == 7):
                    col_names = row
                if (row_count >= 9): # data in conn.log.labeled starts at row 9
                    data.append(row)

    data.pop() #remove the last row because it doesn't contain data
    col_names.pop(0) # remove first item in col_names because it's not a column name

    df = pd.DataFrame(data) # convert data currently in list of lists format to a pandas dataframe
    df.columns = col_names # add column names to the dataframe

    # important: sort dataframe by timestamp so we can calculate interpacket spacing later
    df['ts'] = df['ts'].astype(float) # convert timestamp col to float type
    df = df.sort_values(by='ts', ascending=True) # sort by timestamp column

    return(df) # return the sorted dataframe

def preprocess_df(df):
    """
    Summary: Utility/helper function for cleaning/pre-processing of bro/zeek log dataframe
    """
    # 1. Check for any duplicate entries in the data
    print("Checking for duplicates in log: ")
    df["is_duplicate"] = df.duplicated() # adds a new column to the dataset to track duplication
    print(f"#total data points in log= {len(df)}")
    print(f"#duplicated data points in log= {len(df[df['is_duplicate']==True])}")

    index_to_drop = df[df['is_duplicate']==True].index # Drop the duplicate rows using index
    df.drop(index_to_drop, inplace=True)

    df.drop(columns='is_duplicate', inplace=True) # Remove the duplicate marker column

    # 2. Check for any missing values/NaNs in the data
    # print("--------------------------------")
    # print("Missing values in log (True indicates missing value): ")
    # print(df.isnull().any())

    # 3. Check type for each column
    # print("--------------------------------")
    # print("Data types in log: ")
    # print(df.dtypes)

def encode_onehot(_df, f):
    """
    Summary: One-hot encodes the input categorical/nominal feature within the input dataframe.
    Returns updated dataframe with feature one-hot encoded.
    """
    _df2 = pd.get_dummies(_df[f], prefix='', prefix_sep='').max(level=0, axis=1).add_prefix(f+' - ')
    df3 = pd.concat([_df, _df2], axis=1)
    df3 = df3.drop([f], axis=1)
    return df3

def updateDF(idx, orig, dest, _df):

    results = _df.loc[(_df['id.orig_h'] == orig) & (_df['id.resp_h'] == dest)]

    if results.shape[0] < 2:
        return _df

    idx = 0
    prev_row = results.iloc[idx]

    for index, row in results.iterrows():
        if prev_row.equals(row) == False:     #Check that current row is not previous row
            #if (row['duration'] != '-'):
            interpacket_gap_time = row['ts'] - prev_row['ts'] #Retrieve ip_gap_time
            results.at[idx, 'interpacket_gap_time'] = interpacket_gap_time #add ip_gap_time to results DF
            prev_row = copy.deepcopy(row) #Assign new previous copy
            idx = index                   #Assign index of new previous copy
            #else:
            #    results.at[idx, 'interpacket_gap_time'] = 0

    _df.update(results)#overwrite _df with results rows

    return _df

if __name__ == "__main__":

    connLogLabeled_DF = broLog_to_df('C:\Users\Gilles\PycharmProjects\Capstone\IoTScenarios\CTU-IoT-Malware-Capture-49-1\bro\conn.log.labeled')

    preprocess_df(connLogLabeled_DF)


    #################################################################################
    ##### Create new features from existing features and append to dataframe ########
    #################################################################################
    # Note: From the above step, we see there are 4 labels (multi-class) within this dataset: benign, malicious C&C, malicious DDoS, malicious PartOfAHorizontalPortScan
    # We can create an additional column with binary labels (i.e. 'benign' vs 'malicious') to have the flexibility to consider this as a binary classification problem as well
    connLogLabeled_DF.insert(0, 'binary_label', True) # add new column to the end of the dataset for the binary label (True = malicious, False = benign)

    for index, row in connLogLabeled_DF.iterrows(): # iterate through dataset and populate binary label as malicious or benign
        current_mclassLabel = row['tunnel_parents']
        if ('Benign' in current_mclassLabel or 'benign' in current_mclassLabel):
            connLogLabeled_DF.at[index, 'binary_label'] = False

    print("Adding packet_spacing...")
    # Create inter-packet spacing feature
    if 'packet_spacing' not in connLogLabeled_DF:
        connLogLabeled_DF.insert(10, 'packet_spacing', 0) # add new column to the end of the dataset for spacing

    current_row = connLogLabeled_DF.at[0,'ts'] # initialize current row value

    for index, row in connLogLabeled_DF.iterrows(): # iterate through dataset and calculate and populate interpacket spacing value
        prev_row = copy.deepcopy(current_row)
        current_row = row['ts']
        time_delta = current_row - prev_row
        connLogLabeled_DF.at[index, 'packet_spacing'] = time_delta

    print("Adding interpacket_gap_time...")
    #Create average bytes for each transaction, based on current row
    connLogLabeled_DF.insert(11, 'interpacket_gap_time', 0)
    stored_pairs = []

    #Iterate through dataset
    for index, row in connLogLabeled_DF.iterrows():
        print("Current Row: ", index)
        if (connLogLabeled_DF_1.shape[0] - index) > 1 and stored_pairs.count([row['id.orig_h'], row['id.resp_h']]) < 1:
            connLogLabeled_DF_1 = updateDF(index, row['id.orig_h'], row['id.resp_h'], connLogLabeled_DF_1)
            stored_pairs.append([row['id.orig_h'], row['id.resp_h']])

    ################################################################################
    ##### One-hot encode the categorical (i.e. non-numerical) features  ############
    ################################################################################

    # One-hot encode the nominal/categorical columns we'll be using as features
    connLogLabeled_DF_1 = encode_onehot(connLogLabeled_DF_1, 'proto') # one-hot encode 'proto' feature
    cols = []
    for f in list(connLogLabeled_DF_1.columns.values):
        if 'proto' in f:
            cols += [f]

    connLogLabeled_DF_1 = encode_onehot(connLogLabeled_DF_1, 'history') # one-hot encode 'd.orig_h' feature
    cols = []
    for f in list(connLogLabeled_DF_1.columns.values):
        if 'history' in f:
            cols += [f]

    connLogLabeled_DF_1 = encode_onehot(connLogLabeled_DF_1, 'conn_state') # one-hot encode 'd.orig_h' feature
    cols = []
    for f in list(connLogLabeled_DF_1.columns.values):
        if 'conn_state' in f:
            cols += [f]

    # ----- Set the features the model will use by dropping the ones we don't want to use -----
    features_to_drop = ['uid', 'id.orig_h', 'id.resp_h', 'service', 'duration', 'orig_bytes', 'resp_bytes', 'local_orig', 'local_resp', 'tunnel_parents']

    # ----- Features to Keep
    # 'id.orig_p', 'id.resp_p', 'missed_bytes', 'orig_pkts', 'resp_pkts', 'resp_ip_bytes'

    #Features to Drop
    for feature in features_to_drop:
        if (feature in connLogLabeled_DF_1.columns):
            connLogLabeled_DF_1 = connLogLabeled_DF_1.drop(feature, axis=1)

    #connLogLabeled_DF.to_csv (r'C:\Users\Gilles\PycharmProjects\Capstone\connLogDerived1_testoutput.csv', index = False, header=True)

    connLogLabeled_DF_1 = connLogLabeled_DF_1.apply(pd.to_numeric) # ensure all features being used are numeric

    # sanity check
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    #################################################################################
    ##### Create new features from existing features and append to dataframe ########
    #################################################################################
    # Note: From the above step, we see there are 4 labels (multi-class) within this dataset: benign, malicious C&C, malicious DDoS, malicious PartOfAHorizontalPortScan
    # We can create an additional column with binary labels (i.e. 'benign' vs 'malicious') to have the flexibility to consider this as a binary classification problem as well
    connLogLabeled_DF_2.insert(0, 'binary_label', True) # add new column to the end of the dataset for the binary label (True = malicious, False = benign)

    for index, row in connLogLabeled_DF_2.iterrows(): # iterate through dataset and populate binary label as malicious or benign
        current_mclassLabel = row['tunnel_parents']
        if ('Benign' in current_mclassLabel or 'benign' in current_mclassLabel):
            connLogLabeled_DF_2.at[index, 'binary_label'] = False

    print("Adding packet_spacing...")
    # Create inter-packet spacing feature
    if 'packet_spacing' not in connLogLabeled_DF_2:
        connLogLabeled_DF_2.insert(10, 'packet_spacing', 0) # add new column to the end of the dataset for spacing

    current_row = connLogLabeled_DF_2.at[0,'ts'] # initialize current row value

    for index, row in connLogLabeled_DF_2.iterrows(): # iterate through dataset and calculate and populate interpacket spacing value
        prev_row = copy.deepcopy(current_row)
        current_row = row['ts']
        time_delta = current_row - prev_row
        connLogLabeled_DF_2.at[index, 'packet_spacing'] = time_delta

    print("Adding interpacket_gap_time...")
    #Create average bytes for each transaction, based on current row
    connLogLabeled_DF_2.insert(11, 'interpacket_gap_time', 0)
    stored_pairs = []

    #Iterate through dataset
    for index, row in connLogLabeled_DF_2.iterrows():
        print("Current Row: ", index)
        if (connLogLabeled_DF_2.shape[0] - index) > 1 and stored_pairs.count([row['id.orig_h'], row['id.resp_h']]) < 1:
            connLogLabeled_DF_2 = updateDF(index, row['id.orig_h'], row['id.resp_h'], connLogLabeled_DF_2)
            stored_pairs.append([row['id.orig_h'], row['id.resp_h']])

    ################################################################################
    ##### One-hot encode the categorical (i.e. non-numerical) features  ############
    ################################################################################

    # One-hot encode the nominal/categorical columns we'll be using as features
    connLogLabeled_DF_2 = encode_onehot(connLogLabeled_DF_2, 'proto') # one-hot encode 'proto' feature
    cols = []
    for f in list(connLogLabeled_DF_2.columns.values):
        if 'proto' in f:
            cols += [f]

    connLogLabeled_DF_2 = encode_onehot(connLogLabeled_DF_2, 'history') # one-hot encode 'd.orig_h' feature
    cols = []
    for f in list(connLogLabeled_DF_2.columns.values):
        if 'history' in f:
            cols += [f]

    connLogLabeled_DF_2 = encode_onehot(connLogLabeled_DF_2, 'conn_state') # one-hot encode 'd.orig_h' feature
    cols = []
    for f in list(connLogLabeled_DF_2.columns.values):
        if 'conn_state' in f:
            cols += [f]

    # ----- Set the features the model will use by dropping the ones we don't want to use -----
    features_to_drop = ['uid', 'id.orig_h', 'id.resp_h', 'service', 'duration', 'orig_bytes', 'resp_bytes', 'local_orig', 'local_resp', 'tunnel_parents']

    # ----- Features to Keep
    # 'id.orig_p', 'id.resp_p', 'missed_bytes', 'orig_pkts', 'resp_pkts', 'resp_ip_bytes'

    #Features to Drop
    for feature in features_to_drop:
        if (feature in connLogLabeled_DF_2.columns):
            connLogLabeled_DF_2 = connLogLabeled_DF_2.drop(feature, axis=1)

    #connLogLabeled_DF.to_csv (r'C:\Users\Gilles\PycharmProjects\Capstone\connLogDerived1_testoutput.csv', index = False, header=True)

    connLogLabeled_DF_2 = connLogLabeled_DF_2.apply(pd.to_numeric) # ensure all features being used are numeric

    # sanity check
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    connLogLabeled_DF_1.to_csv(r'C:\Users\Gilles\PycharmProjects\Capstone\HUE_Alexa_test-1_log.csv', index = False, header=True)
    connLogLabeled_DF_2.to_csv(r'C:\Users\Gilles\PycharmProjects\Capstone\HUE_Alexa_test-2-Filter_2_log.csv', index = False, header=True)

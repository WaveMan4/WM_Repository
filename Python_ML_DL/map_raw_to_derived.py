# ----- Importing libraries -----
import copy
import csv
import pandas as pd
import sklearn
import numpy as np
import matplotlib
#matplotlib.use('TkAgg') # Note: comment out this line if you're not using MacOS
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier

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
            if (row['duration'] != '-'):
                interpacket_gap_time = row['ts'] - prev_row['ts'] #Retrieve ip_gap_time
                results.at[idx, 'interpacket_gap_time'] = interpacket_gap_time #add ip_gap_time to results DF
                prev_row = copy.deepcopy(row) #Assign new previous copy
                idx = index                   #Assign index of new previous copy
            else:
                results.at[idx, 'interpacket_gap_time'] = 0

    _df.update(results)#overwrite _df with results rows

    return _df

def retrieve_derived_features(derivedDS, orighost, origport, resphost, respport):
    derived_row = derivedDS.loc[(derivedDS['id.orig_h'] == orighost)
                                & (derivedDS['id.resp_h'] == resphost)
                                & (derivedDS['id.orig_p'] == int(origport))
                                & (derivedDS['id.resp_p'] == int(respport)) ]
    return derived_row

if __name__ == "__main__":

    # ----- Convert conn log zeek/bro file to Pandas dataframe (df) -----
    # Note: you will have to change the path here depending on where you've saved the files...
    connLogLabeled_DF = pd.read_csv('exp4_dataset.csv')
    DS_3_1 = pd.read_csv('IoT-Malware-Capture-3-1.csv') #input derived file
    DS_8_1 = pd.read_csv('IoT-Malware-Capture-8-1.csv') #input derived file
    DS_20_1 = pd.read_csv('IoT-Malware-Capture-20-1.csv') #input derived file
    DS_34_1 = pd.read_csv('IoT-Malware-Capture-34-1.csv') #input derived file
    DS_42_1 = pd.read_csv('IoT-Malware-Capture-42-1.csv') #input derived file
    frames = [DS_3_1, DS_8_1, DS_20_1, DS_34_1, DS_42_1]
    derived_DS = pd.concat(frames)

    # ----- Preprocess and clean both data sets -----
    print('preprocessing connLogLabeled_DF...')
    preprocess_df(connLogLabeled_DF)
    print('preprocessing derived_DS...')
    preprocess_df(derived_DS)

    # ----- Output current datafrom to csv file (conn_log_dataframe.csv)
    #connLogLabeled_DF.to_csv (r'C:\Users\Gilles\PycharmProjects\Capstone\conn_log_dataframe.csv', index = False, header=True)

    #Next, add columns for the binary label, the packet spacing, and the interpacket-gap-time
    #connLogLabeled_DF.insert(0, 'binary_label', True) # add new column to the end of the dataset for the binary label (True = malicious, False = benign)

    #for index, row in connLogLabeled_DF.iterrows(): # iterate through dataset and populate binary label as malicious or benign
    #    current_mclassLabel = row['tunnel_parents   label   detailed-label']
    #    if ('Benign' in current_mclassLabel or 'benign' in current_mclassLabel):
    #        connLogLabeled_DF.at[index, 'binary_label'] = False

    #################################################################################
    ##### Create new features from existing features and append to dataframe ########
    #################################################################################

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
        if (connLogLabeled_DF.shape[0] - index) > 1 and stored_pairs.count([row['id.orig_h'], row['id.resp_h']]) < 1:
            connLogLabeled_DF = updateDF(index, row['id.orig_h'], row['id.resp_h'], connLogLabeled_DF)
            stored_pairs.append([row['id.orig_h'], row['id.resp_h']])

    ####################################################################################################################
    #   Next create, rows of excluded features
    #       Note: these features exist in both the raw file and the derived file
    #        therefore, we do not need these features.
    #        we only need the derived features to be added to the raw features
    excluded = ['id.orig_h', 'id.orig_p', 'id.resp_h', 'id.resp_p', 'proto']

    #Check dimensions of connLoglabeled_DF - expect it to increase
    print(connLogLabeled_DF.shape)

    for index, row in connLogLabeled_DF.iterrows():
        derived_row = retrieve_derived_features(derived_DS,
                                                row['id.orig_h'], row['id.orig_p'],
                                                row['id.resp_h'], row['id.resp_p'])
        if derived_row.shape[0] > 0:
            for feature in derived_row.columns:
                if (feature not in excluded):
                    val = derived_row[feature].values[0]
                    connLogLabeled_DF.at[index, feature] = val
        else:
            for feature in derived_row.columns:
                if (feature not in excluded):
                    connLogLabeled_DF.at[index, feature] = 0

    #Check dimensions of connLog label - make sure it has increased
    print(connLogLabeled_DF.shape)

    # ----- Output current datafrom to csv file (conn_log_dataframe.csv)
    connLogLabeled_DF.to_csv (r'C:\Users\Gilles\PycharmProjects\Capstone\exp4_capstone_w_derived.csv', index = False, header=True)

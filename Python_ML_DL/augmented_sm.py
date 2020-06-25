### EN605.795.8VL.SP20

# ----- Importing libraries -----
import copy
import csv
import pandas as pd
import sklearn
import numpy as np
import matplotlib
import math
matplotlib.use('TkAgg') # Note: comment out this line if you're not using MacOS
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

######################################
#   Random Forest Hyper-parameters   #
######################################
estimators = 150
depth = 5
jobs = 4

######################################
#   Decision Tree Hyper-parameters   #
######################################
splits = 5
leafs = 5
fraction_leafs = 0.25
features_flags = 'sqrt'
impurity_decrease = 2.1
alpha_val = 3.5


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

def train_test_accuracy(_X_tr, _X_ts, _y_tr, _y_ts):
    """
    Summary: Performs model training and testing on input data using the following supervised ML models and prints accuracy score for each model:
    Random Forest, Gaussian Naive Bayes, Decision Tree, Logistic Regression

    Prints accuracy score for each model used.
    """

    scaler = StandardScaler() # scale features so that all of them can be uniformly evaluated
    scaler.fit(_X_tr)

    _X_tr = scaler.transform(_X_tr)
    _X_ts = scaler.transform(_X_ts)

    rf = RandomForestClassifier(n_estimators=estimators, max_depth=depth, random_state=None, n_jobs=jobs) # Create a new random forest classifier, with working 4 parallel cores
    model = rf.fit(_X_tr, _y_tr) # Train RF classifier on training data
    y_pred = rf.predict(_X_ts) # Test RF classifier on testing data
    print("For Random Forest Classifier:")
    print("     Hyperparams are %4d estimators, max depth of %2d, random_state set to None, and %2d jobs" %(estimators, depth, jobs))
    print("Accuracy: ", accuracy_score(_y_ts, y_pred)) # Print accuracy of RF classifier
    print("Precision:", precision_score(_y_ts, y_pred))
    print("Recall:", recall_score(_y_ts, y_pred))
    print("F1 score", f1_score(_y_ts, y_pred))
    print("Confusion Matrix:", confusion_matrix(_y_ts, y_pred))

    nb = GaussianNB() # Create a new Naive Bayes Classifier
    model = nb.fit(_X_tr, _y_tr) # Train NB classifier on training data
    y_pred = nb.predict(_X_ts) # Test NB classifier on testing data
    print("For Naive Bayes Classifier: ")
    print("Accuracy: ", accuracy_score(_y_ts, y_pred)) # Print accuracy of NB classifier
    print("Precision:", precision_score(_y_ts, y_pred))
    print("Recall:", recall_score(_y_ts, y_pred))
    print("F1 score", f1_score(_y_ts, y_pred))
    print("Confusion Matrix:", confusion_matrix(_y_ts, y_pred))

    dt = DecisionTreeClassifier() # Create a new Decision Tree Classifier
    model = dt.fit(_X_tr, _y_tr) # Train DT classifier on training data
    y_pred = dt.predict(_X_ts) # Test DT classifier on testing data
    print("For Decision Tree Classifier: ")
    print("Accuracy: ", accuracy_score(_y_ts, y_pred)) # Print accuracy of DT classifier
    print("Precision:", precision_score(_y_ts, y_pred))
    print("Recall:", recall_score(_y_ts, y_pred))
    print("F1 score", f1_score(_y_ts, y_pred))
    print("Confusion Matrix:", confusion_matrix(_y_ts, y_pred))

    lr = LogisticRegression() # Create a new Logistic Regression Classifier
    model = lr.fit(_X_tr, _y_tr) # Train LR classifier on training data
    y_pred = lr.predict(_X_ts) # Test LR classifier on testing data
    print("For Logistic Regression Classifier: ")
    print("Accuracy: ", accuracy_score(_y_ts, y_pred)) # Print accuracy of LR classifier
    print("Precision:", precision_score(_y_ts, y_pred))
    print("Recall:", recall_score(_y_ts, y_pred))
    print("F1 score", f1_score(_y_ts, y_pred))
    print("Confusion Matrix:", confusion_matrix(_y_ts, y_pred))

    # Takes too long to run?
    # sv = svm.SVC(kernel='linear') # Create a new SVM Classifier with Linear kernel
    # model = sv.fit(_X_tr, _y_tr) # Train the SVM Classifier on training data
    # y_pred = sv.predict(_X_ts) # Test SVM Classifier on testing data
    # print("For SVM Classifier: ")
    # print("Accuracy: ", accuracy_score(_y_ts, y_pred)) # Print accuracy of SVM classifier
    # print("Precision:", precision_score(y_test, y_pred))
    # print("Recall:", recall_score(y_test, y_pred))
    # print("F1 score", f1_score(y_test, y_pred))
    # print("Confusion Matrix:", confusion_matrix(y_test, y_pred))

    #knn = KNeighborsClassifier(n_neighbors=5) # Create a new K Nearest Neighbor Classifier
    #model = knn.fit(_X_tr, _y_tr)
    #y_pred = knn.predict(_X_ts)
    #print("For K Nearest Neighbors Classifier: ")
    #print("Accuracy: ", accuracy_score(_y_ts, y_pred)) # Print accuracy of knn classifier
    #print("Precision:", precision_score(_y_ts, y_pred))
    #print("Recall:", recall_score(_y_ts, y_pred))
    #print("F1 score", f1_score(_y_ts, y_pred))
    #print("Confusion Matrix:", confusion_matrix(_y_ts, y_pred))

    gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1)
    gbc.fit(_X_tr, _y_tr)
    y_pred = gbc.predict(_X_ts)
    print("For Gradient Boosting: ")
    print("Accuracy: ", accuracy_score(_y_ts, y_pred)) # Print accuracy of knn classifier
    print("Precision:", precision_score(_y_ts, y_pred))
    print("Recall:", recall_score(_y_ts, y_pred))
    print("F1 score", f1_score(_y_ts, y_pred))
    print("Confusion Matrix:", confusion_matrix(_y_ts, y_pred))
    #print("Precision:", precision_score(y_test, y_pred))
    #print("Recall:", recall_score(y_test, y_pred))
    #print("F1 score", f1_score(y_test, y_pred))
    #print("Confusion Matrix:", confusion_matrix(y_test, y_pred))
    # n_estimators: It controls the number of weak learners.
    # learning_rate:Controls the contribution of weak learners in the final combination. There is a trade-off between learning_rate and n_estimators.
    # max_depth: maximum depth of the individual regression estimators. The maximum depth limits the number of nodes in the tree. Tune this parameter for best performance; the best value depends on the interaction of the input variables

def rf_train_test(_X_tr, _X_ts, _y_tr, _y_ts):
    """
    Summary: Performs model training and testing on input data using the Random Forest classifier.
    Returns the accuracy score
    """
    # Create a new random forest classifier, with working 4 parallel cores
    rf = RandomForestClassifier(n_estimators=estimators, max_depth=depth, random_state=None, n_jobs=jobs)
    # Train on training data
    model = rf.fit(_X_tr, _y_tr)
    # Test on testing data
    y_pred = rf.predict(_X_ts)
    # Return accuracy
    return(accuracy_score(_y_ts, y_pred))

def updateDF(idx, orig, dest, _df):

    results = _df.loc[(_df['id_orig_h'] == orig) & (_df['id_resp_h'] == dest)]

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

def create_capstone_dataset(connLogLabeled_DF):
        # ----- Output current datafrom to csv file (conn_log_dataframe.csv)
    #connLogLabeled_DF.to_csv (r'C:\Users\Gilles\PycharmProjects\Capstone\connLogDerived_8_42_mod.csv', index = False, header=True)

    # ----- Preprocess and clean the data -----
    #preprocess_df(connLogLabeled_DF) # start to preprocess conn log df

    #print("--------------------------------")
    #print("Labels in the original dataset: ")
    #print (connLogLabeled_DF['tunnel_parents   label   detailed-label'].cat.categories) # Check the label (i.e. dependent variable) categories

    #Note: From the above step, we see there are 4 labels (multi-class) within this dataset: benign, malicious C&C, malicious DDoS, malicious PartOfAHorizontalPortScan
    #We can create an additional column with binary labels (i.e. 'benign' vs 'malicious') to have the flexibility to consider this as a binary classification problem as well
    #connLogLabeled_DF.insert(0, 'binary_label', True) # add new column to the end of the dataset for the binary label (True = malicious, False = benign)

    #for index, row in connLogLabeled_DF.iterrows(): # iterate through dataset and populate binary label as malicious or benign
        #current_mclassLabel = row['tunnel_parents   label   detailed-label']
    #    current_mclassLabel = row['label']
    #    if (type(current_mclassLabel) is float and math.isnan(current_mclassLabel)):
    #        current_mclassLabel = row['tunnel_parents']
    #        if ('-' in current_mclassLabel):
    #            connLogLabeled_DF.at[index, 'binary_label'] = False
    #        else:
    #            connLogLabeled_DF.drop(index, inplace=True)
    #    else:
    #        if ('Benign' in current_mclassLabel or 'benign' in current_mclassLabel):
    #            connLogLabeled_DF.at[index, 'binary_label'] = False

    # Create inter-packet spacing feature
    connLogLabeled_DF.insert(21, 'packet_spacing', 0) # add new column to the end of the dataset for spacing

    current_row = connLogLabeled_DF.at[0,'ts'] # initialize current row value

    for index, row in connLogLabeled_DF.iterrows(): # iterate through dataset and calculate and populate interpacket spacing value
       prev_row = copy.deepcopy(current_row)
       current_row = row['ts']
       time_delta = current_row - prev_row
       connLogLabeled_DF.at[index, 'packet_spacing'] = time_delta

    print("Adding interpacket_gap_time...")
    #Create average bytes for each transaction, based on current row
    connLogLabeled_DF.insert(22, 'interpacket_gap_time', 0)
    stored_pairs = []
    print(connLogLabeled_DF.head())

    #Iterate through dataset
    for index, row in connLogLabeled_DF.iterrows():
        if (connLogLabeled_DF.shape[0] - index) > 1 and stored_pairs.count([row['id_orig_h'], row['id_resp_h']]) < 1:
            connLogLabeled_DF = updateDF(index, row['id_orig_h'], row['id_resp_h'], connLogLabeled_DF)
            stored_pairs.append([row['id_orig_h'], row['id_resp_h']])

    ################################################################################
    ##### One-hot encode the categorical (i.e. non-numerical) features  ############
    ################################################################################

    # One-hot encode the nominal/categorical columns we'll be using as features
    #connLogLabeled_DF = encode_onehot(connLogLabeled_DF, 'proto') # one-hot encode 'proto' feature
    #cols = []
    #for f in list(connLogLabeled_DF.columns.values):
    #   if 'proto' in f:
    #        cols += [f]

    #connLogLabeled_DF = encode_onehot(connLogLabeled_DF, 'history') # one-hot encode 'd.orig_h' feature
    #cols = []
    #for f in list(connLogLabeled_DF.columns.values):
    #    if 'history' in f:
    #        cols += [f]

    #connLogLabeled_DF = encode_onehot(connLogLabeled_DF, 'conn_state') # one-hot encode 'd.orig_h' feature
    #cols = []
    #for f in list(connLogLabeled_DF.columns.values):
    #    if 'conn_state' in f:
    #        cols += [f]

    # ----- Set the features the model will use by dropping the ones we don't want to use -----
    #features_to_drop = ['ts', 'uid', 'id.orig_h', 'id.resp_h', 'service', 'duration', 'orig_bytes', 'resp_bytes', 'local_orig', 'local_resp', 'tunnel_parents   label   detailed-label']

    # ----- Features to Keep
    # 'id.orig_p', 'id.resp_p', 'missed_bytes', 'orig_pkts', 'resp_pkts', 'resp_ip_bytes'

    #Features to Drop
    #for feature in features_to_drop:
    #   if (feature in connLogLabeled_DF.columns):
    #       connLogLabeled_DF = connLogLabeled_DF.drop(feature, axis=1)

    #connLogLabeled_DF = connLogLabeled_DF.apply(pd.to_numeric) # ensure all features being used are numeric

    # sanity check
    #pd.set_option('display.max_columns', None)
    #pd.set_option('display.max_rows', None)

            ##### Run models and print results #####
    # ----- Create X and Y vectors -----
    #X = connLogLabeled_DF.loc[:, connLogLabeled_DF.columns != 'binary_label'].values # X can contains all features minus the features that were dropped minus the label
    #y = connLogLabeled_DF.loc[:, connLogLabeled_DF.columns == 'binary_label'].values.ravel() # Y contains just the binary label

    # ----- Print bar graph showing benign/malicious makeup of dataset that was read in -----#
    #count = 0
    #malicious_count = 0
    #benign_count = 0
    #for label in y:
    #   if (label == False):
    #       benign_count += 1
    #   else:
    #       malicious_count += 1

    #print("--------------------------------")
    #print("percent of data points labeled benign = ", (benign_count/len(y)) )
    #print("number of benign pts: ", benign_count)
    #print("percent of data points labeled malicious = ", (1-(benign_count/len(y))) )
    #print("--------------------------------")

    #label = ['Benign', 'Malicious']
    #counts = [benign_count, malicious_count]
    #index = np.arange(len(label))
    #plt.bar(index, counts)
    #plt.xlabel('Label', fontsize=12)
    #plt.ylabel('# Data Points', fontsize=12)
    #plt.xticks(index, label, fontsize=10)
    #plt.title('Benign vs Malicious Data Points in Dataset')
    #plt.savefig('benignVsmaliciousCount_unbalancedDataset.png')

    return connLogLabeled_DF

def create_capstone_benign_dataset(connLogLabeled_DF):
        # ----- Output current datafrom to csv file (conn_log_dataframe.csv)
    #connLogLabeled_DF.to_csv (r'C:\Users\Gilles\PycharmProjects\Capstone\connLogDerived_8_42_mod.csv', index = False, header=True)

    # ----- Preprocess and clean the data -----
    preprocess_df(connLogLabeled_DF) # start to preprocess conn log df

    print("--------------------------------")
    print("Labels in the original dataset: ")
    #print (connLogLabeled_DF['tunnel_parents   label   detailed-label'].cat.categories) # Check the label (i.e. dependent variable) categories

    #Note: From the above step, we see there are 4 labels (multi-class) within this dataset: benign, malicious C&C, malicious DDoS, malicious PartOfAHorizontalPortScan
    #We can create an additional column with binary labels (i.e. 'benign' vs 'malicious') to have the flexibility to consider this as a binary classification problem as well
    connLogLabeled_DF.insert(0, 'binary_label', False) # add new column to the end of the dataset for the binary label (True = malicious, False = benign)

    # Create inter-packet spacing feature
    connLogLabeled_DF.insert(21, 'packet_spacing', 0) # add new column to the end of the dataset for spacing

    current_row = connLogLabeled_DF.at[0,'ts'] # initialize current row value

    for index, row in connLogLabeled_DF.iterrows(): # iterate through dataset and calculate and populate interpacket spacing value
       prev_row = copy.deepcopy(current_row)
       current_row = row['ts']
       time_delta = current_row - prev_row
       connLogLabeled_DF.at[index, 'packet_spacing'] = time_delta

    print("Adding interpacket_gap_time...")
    #Create average bytes for each transaction, based on current row
    connLogLabeled_DF.insert(22, 'interpacket_gap_time', 0)
    stored_pairs = []
    #Iterate through dataset and add interpacket-gap-times
    for index, row in connLogLabeled_DF.iterrows():
        if (connLogLabeled_DF.shape[0] - index) > 1 and stored_pairs.count([row['id.orig_h'], row['id.resp_h']]) < 1:
            connLogLabeled_DF = updateDF(index, row['id.orig_h'], row['id.resp_h'], connLogLabeled_DF)
            stored_pairs.append([row['id.orig_h'], row['id.resp_h']])

    ################################################################################
    ##### One-hot encode the categorical (i.e. non-numerical) features  ############
    ################################################################################

    # One-hot encode the nominal/categorical columns we'll be using as features
    #connLogLabeled_DF = encode_onehot(connLogLabeled_DF, 'proto') # one-hot encode 'proto' feature
    #cols = []
    #for f in list(connLogLabeled_DF.columns.values):
    #   if 'proto' in f:
    #        cols += [f]

    #connLogLabeled_DF = encode_onehot(connLogLabeled_DF, 'history') # one-hot encode 'd.orig_h' feature
    #cols = []
    #for f in list(connLogLabeled_DF.columns.values):
    #    if 'history' in f:
    #        cols += [f]

    #connLogLabeled_DF = encode_onehot(connLogLabeled_DF, 'conn_state') # one-hot encode 'd.orig_h' feature
    #cols = []
    #for f in list(connLogLabeled_DF.columns.values):
    #    if 'conn_state' in f:
    #        cols += [f]

    # ----- Set the features the model will use by dropping the ones we don't want to use -----
    #features_to_drop = ['ts', 'uid', 'id.orig_h', 'id.resp_h', 'service', 'duration', 'orig_bytes', 'resp_bytes', 'local_orig', 'local_resp', 'tunnel_parents']

    # ----- Features to Keep
    # 'id.orig_p', 'id.resp_p', 'missed_bytes', 'orig_pkts', 'resp_pkts', 'resp_ip_bytes'

    #Features to Drop
    #for feature in features_to_drop:
    #   if (feature in connLogLabeled_DF.columns):
    #       connLogLabeled_DF = connLogLabeled_DF.drop(feature, axis=1)

    #connLogLabeled_DF = connLogLabeled_DF.apply(pd.to_numeric) # ensure all features being used are numeric

    # sanity check
    #pd.set_option('display.max_columns', None)
    #pd.set_option('display.max_rows', None)

            ##### Run models and print results #####
    # ----- Create X and Y vectors -----
    #X = connLogLabeled_DF.loc[:, connLogLabeled_DF.columns != 'binary_label'].values # X can contains all features minus the features that were dropped minus the label
    #y = connLogLabeled_DF.loc[:, connLogLabeled_DF.columns == 'binary_label'].values.ravel() # Y contains just the binary label

    # ----- Print bar graph showing benign/malicious makeup of dataset that was read in -----#
    #count = 0
    #malicious_count = 0
    #benign_count = 0
    #for label in y:
    #   if (label == False):
    #       benign_count += 1
    #   else:
    #       malicious_count += 1

    #print("--------------------------------")
    #print("percent of data points labeled benign = ", (benign_count/len(y)) )
    #print("number of benign pts: ", benign_count)
    #print("percent of data points labeled malicious = ", (1-(benign_count/len(y))) )
    #print("--------------------------------")

    #label = ['Benign', 'Malicious']
    #counts = [benign_count, malicious_count]
    #index = np.arange(len(label))
    #plt.bar(index, counts)
    #plt.xlabel('Label', fontsize=12)
    #plt.ylabel('# Data Points', fontsize=12)
    #plt.xticks(index, label, fontsize=10)
    #plt.title('Benign vs Malicious Data Points in Dataset')
    #plt.savefig('benignVsmaliciousCount_unbalancedDataset.png')

    return connLogLabeled_DF

if __name__ == "__main__":

    connLogLabeled_DF_1 = pd.read_csv(r'C:\Users\Gilles\PycharmProjects\Capstone\Large_Dataset_1_3M.csv')
    connLogLabeled_DF_1 = create_capstone_dataset(connLogLabeled_DF_1)
    connLogLabeled_DF_1.to_csv(r'C:\Users\Gilles\PycharmProjects\Capstone\Capstone_3M_v1.csv', index = False, header=True)

    #print(connLogLabeled_DF.head())

    # ----- Calculate correlation of each feature to the binary label (i.e. the target variable) and visualize results -----
    # Keep the target variable in the X
    #X = connLogLabeled_DF.values
    #N, M = X.shape

    # Average of each feature, i.e. sample mean
    #Xavg = np.zeros(M)
    #for i in range(M):
    #    Xavg[i] = np.sum(X[:,i]) / N

    # Stdev of each feature, i.e. sample standard deviation
    #Xvar = np.zeros(M)
    #for i in range(M):
    #    Xvar[i] = np.sqrt(np.sum((X[:,i]-Xavg[i])**2))

    # Correlation
    #Xcorr = np.zeros((M,M))
    #for i in range(M):
    #    for j in range(M):
    #        Xcorr[i,j] = np.sum((X[:,i]-Xavg[i])*(X[:,j]-Xavg[j])) / (Xvar[i]*Xvar[j])

    # Plot and save figure
    #plt.imshow(Xcorr, cmap='hsv', interpolation='nearest')
    #plt.yticks(np.arange(M), labels=connLogLabeled_DF.columns, fontsize=5)
    #plt.xticks(np.arange(M), labels=connLogLabeled_DF.columns, rotation=90, fontsize=5)
    #plt.colorbar()
    #plt.show()
    #plt.tight_layout()
    #plt.savefig('correlation_matrix.png')



    # # ----- Print some initial accuracy results using 80/20 train/test split -----
    #X = connLogLabeled_DF.loc[:, connLogLabeled_DF.columns != 'binary_label'].values # X contains the following features: ts, id.orig_p, id.resp_p, orig_ip_bytes, id.orig_h, id.resp_h, proto
    #y = connLogLabeled_DF.loc[:, connLogLabeled_DF.columns == 'binary_label'].values.ravel() # Y contains just the binary label
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None) # use 80/20 test/train split
    #train_test_accuracy(X_train, X_test, y_train, y_test)

    # ----- Evaluate highest performing model using 80-20 split run 10 times and 10-fold cross validation -----
    #print("--------------------------------")
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None) # use 80/20 test/train split
    #accuracies = []
    #for i in range(10): # Run 10 times with 80/20 split and collect statistics
    #   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=None)
    #   accuracies += [rf_train_test(X_train, X_test, y_train, y_test)]
    #print(f'80% train-test split accuracy using Random Forest is {np.mean(accuracies):.3f} {chr(177)}{np.std(accuracies):.4f}')

    #accuracies = []
    #kf = KFold(n_splits=10,shuffle=False,random_state=None) # Perform 10-fold cross validation
    #for train_index, test_index in kf.split(X, y):
    #   acc = rf_train_test(X[train_index], X[test_index], y[train_index], y[test_index])
    #   accuracies += [acc]
    #print(f'10-fold cross validation accuracy using Random Forest is {np.mean(accuracies):.3f} {chr(177)}{np.std(accuracies):.4f}')


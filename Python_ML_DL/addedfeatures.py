# ----- Inter-packet time gap -----
import copy
import csv
import pandas as pd
import sklearn
import numpy as np
import matplotlib
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


    rf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=None, n_jobs=4) # Create a new random forest classifier, with working 4 parallel cores
    model = rf.fit(_X_tr, _y_tr) # Train RF classifier on training data
    y_pred = rf.predict(_X_ts) # Test RF classifier on testing data
    print("For Random Forest Classifier: ")
    print("Accuracy: ", accuracy_score(_y_ts, y_pred)) # Print accuracy of RF classifier
    print("Precision:", precision_score(_y_ts, y_pred))
    print("Recall:", recall_score(_y_ts, y_pred))
    print("F1 score", f1_score(_y_ts, y_pred))
    print("Confusion Matrix:", confusion_matrix(y_test, y_pred))
    #print("Precision:", precision_score(y_test, y_pred))
    #print("Recall:", recall_score(y_test, y_pred))
    #print("F1 score", f1_score(y_test, y_pred))
    #print("Confusion Matrix:", confusion_matrix(y_test, y_pred))

    nb = GaussianNB() # Create a new Naive Bayes Classifier
    model = nb.fit(_X_tr, _y_tr) # Train NB classifier on training data
    y_pred = nb.predict(_X_ts) # Test NB classifier on testing data
    print("For Naive Bayes Classifier: ")
    print("Accuracy: ", accuracy_score(_y_ts, y_pred)) # Print accuracy of NB classifier
    print("Precision:", precision_score(_y_ts, y_pred))
    print("Recall:", recall_score(_y_ts, y_pred))
    print("F1 score", f1_score(_y_ts, y_pred))
    print("Confusion Matrix:", confusion_matrix(y_test, y_pred))
    #print("Precision:", precision_score(y_test, y_pred))
    #print("Recall:", recall_score(y_test, y_pred))
    #print("F1 score", f1_score(y_test, y_pred))
    #print("Confusion Matrix:", confusion_matrix(y_test, y_pred))

    dt = DecisionTreeClassifier() # Create a new Decision Tree Classifier
    model = dt.fit(_X_tr, _y_tr) # Train DT classifier on training data
    y_pred = dt.predict(_X_ts) # Test DT classifier on testing data
    print("For Decision Tree Classifier: ")
    print("Accuracy: ", accuracy_score(_y_ts, y_pred)) # Print accuracy of DT classifier
    print("Precision:", precision_score(_y_ts, y_pred))
    print("Recall:", recall_score(_y_ts, y_pred))
    print("F1 score", f1_score(_y_ts, y_pred))
    print("Confusion Matrix:", confusion_matrix(_y_ts, y_pred))
    #print("Precision:", precision_score(y_test, y_pred))
    #print("Recall:", recall_score(y_test, y_pred))
    #print("F1 score", f1_score(y_test, y_pred))
    #print("Confusion Matrix:", confusion_matrix(y_test, y_pred))

    lr = LogisticRegression() # Create a new Logistic Regression Classifier
    model = lr.fit(_X_tr, _y_tr) # Train LR classifier on training data
    y_pred = lr.predict(_X_ts) # Test LR classifier on testing data
    print("For Logistic Regression Classifier: ")
    print("Accuracy: ", accuracy_score(_y_ts, y_pred)) # Print accuracy of LR classifier
    print("Precision:", precision_score(_y_ts, y_pred))
    print("Recall:", recall_score(_y_ts, y_pred))
    print("F1 score", f1_score(_y_ts, y_pred))
    print("Confusion Matrix:", confusion_matrix(_y_ts, y_pred))
    #print("Precision:", precision_score(y_test, y_pred))
    #print("Recall:", recall_score(y_test, y_pred))
    #print("F1 score", f1_score(y_test, y_pred))
    #print("Confusion Matrix:", confusion_matrix(y_test, y_pred))

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

    knn = KNeighborsClassifier(n_neighbors=5) # Create a new K Nearest Neighbor Classifier
    model = knn.fit(_X_tr, _y_tr)
    y_pred = knn.predict(_X_ts)
    print("For K Nearest Neighbors Classifier: ")
    print("Accuracy: ", accuracy_score(_y_ts, y_pred)) # Print accuracy of knn classifier
    print("Precision:", precision_score(_y_ts, y_pred))
    print("Recall:", recall_score(_y_ts, y_pred))
    print("F1 score", f1_score(_y_ts, y_pred))
    print("Confusion Matrix:", confusion_matrix(_y_ts, y_pred))
    #print("Precision:", precision_score(y_test, y_pred))
    #print("Recall:", recall_score(y_test, y_pred))
    #print("F1 score", f1_score(y_test, y_pred))
    #print("Confusion Matrix:", confusion_matrix(y_test, y_pred))

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
    rf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=None, n_jobs=4)
    # Train on training data
    model = rf.fit(_X_tr, _y_tr)
    # Test on testing data
    y_pred = rf.predict(_X_ts)
    # Return accuracy
    return(accuracy_score(_y_ts, y_pred))

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

if __name__ == "__main__":

    connLogLabeled_DF = pd.read_csv("dataset_balanced.csv")

    # ----- Preprocess and clean the data -----
    preprocess_df(connLogLabeled_DF) # start to preprocess conn log df

    #################################################################################
    ##### Create new features from existing features and append to dataframe ########
    #################################################################################

    # Create inter-packet spacing feature
    if 'packet_spacing' not in connLogLabeled_DF:
        connLogLabeled_DF.insert(18, 'packet_spacing', 0) # add new column to the end of the dataset for spacing

    current_row = connLogLabeled_DF.at[0,'ts'] # initialize current row value

    for index, row in connLogLabeled_DF.iterrows(): # iterate through dataset and calculate and populate interpacket spacing value
         prev_row = copy.deepcopy(current_row)
         current_row = row['ts']
         time_delta = current_row - prev_row
         connLogLabeled_DF.at[index, 'packet_spacing'] = time_delta

    #Create average bytes for each transaction, based on current row
    connLogLabeled_DF.insert(19, 'interpacket_gap_time', 0)

    stored_pairs = []
    #Iterate through dataset
    for index, row in connLogLabeled_DF.iterrows():
        if (connLogLabeled_DF.shape[0] - index) > 1 and stored_pairs.count([row['id.orig_h'], row['id.resp_h']]) < 1:
            connLogLabeled_DF = updateDF(index, row['id.orig_h'], row['id.resp_h'], connLogLabeled_DF)
            stored_pairs.append([row['id.orig_h'], row['id.resp_h']])
    connLogLabeled_DF.to_csv (r'C:\Users\Gilles\PycharmProjects\Capstone\exp4_mod_dataframe.csv', index = False, header=True)


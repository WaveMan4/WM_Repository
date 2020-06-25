### Summary: Functions and main method to test extracting features and basic analysis of zeek/bro log data
### EN605.795.8VL.SP20
### Author: Mandira Hegde (mhegde1@jhu.edu)

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

if __name__ == "__main__":

    # ----- Convert conn log zeek/bro file to Pandas dataframe (df) -----
    # Note: you will have to change the path here depending on where you've saved the files...
    # connLogLabeled_DF = broLog_to_df('capstone_data/CTU-IoT-Malware-Capture-34-1/bro/conn.log.labeled') 
    #miraiData_DF = broLog_to_df('conn.log.labeled')
    # connLogLabeled_DF = broLog_to_df('capstone_data/benign_captures/bro/conn.log.normalAlexa.labeled')
    #alexaDataBenign_DF = broLog_to_df('conn.log.normalAlexa.labeled')
    #phillipsDataBenign_DF = broLog_to_df('conn.log.normalPhillips.labeled')
    #soomfyDataBenign_DF = broLog_to_df('conn.log.normalSoomfy.labeled')

    # Concatenate the dataframes into a single dataframe for  (https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html)

    #frames = [miraiData_DF, alexaDataBenign_DF, phillipsDataBenign_DF, soomfyDataBenign_DF]
    #connLogLabeled_DF = pd.concat(frames)

    # TEST
    # print(connLogLabeled_DF.head(15))


    connLogLabeled_DF = pd.read_csv('exp4_capstone_w_derived.csv')
    # ----- Preprocess and clean the data -----
    preprocess_df(connLogLabeled_DF) # start to preprocess conn log df 

    # ----- Output current datafrom to csv file (conn_log_dataframe.csv)
    #connLogLabeled_DF.to_csv (r'C:\Users\Gilles\PycharmProjects\Capstone\conn_log_dataframe.csv', index = False, header=True)

    # print("--------------------------------")
    # print("Labels in the original dataset: ")
    # print (connLogLabeled_DF['tunnel_parents   label   detailed-label'].cat.categories) # Check the label (i.e. dependent variable) categories

    # Note: From the above step, we see there are 4 labels (multi-class) within this dataset: benign, malicious C&C, malicious DDoS, malicious PartOfAHorizontalPortScan
    # We can create an additional column with binary labels (i.e. 'benign' vs 'malicious') to have the flexibility to consider this as a binary classification problem as well
    #connLogLabeled_DF.insert(0, 'binary_label', True) # add new column to the end of the dataset for the binary label (True = malicious, False = benign)

    #for index, row in connLogLabeled_DF.iterrows(): # iterate through dataset and populate binary label as malicious or benign
    #    current_mclassLabel = row['tunnel_parents   label   detailed-label']
    #    if ('Benign' in current_mclassLabel or 'benign' in current_mclassLabel):
    #        connLogLabeled_DF.at[index, 'binary_label'] = False

   # Create inter-packet spacing feature
   # connLogLabeled_DF.insert(21, 'packet_spacing', 0) # add new column to the end of the dataset for spacing

    #current_row = connLogLabeled_DF.at[0,'ts'] # initialize current row value

    #for index, row in connLogLabeled_DF.iterrows(): # iterate through dataset and calculate and populate interpacket spacing value
    #    prev_row = copy.deepcopy(current_row)
    #    current_row = row['ts']
    #    time_delta = current_row - prev_row
    #    connLogLabeled_DF.at[index, 'packet_spacing'] = time_delta

    # One-hot encode the nominal/categorical columns we'll be using as features
    #connLogLabeled_DF = encode_onehot(connLogLabeled_DF, 'proto') # one-hot encode 'proto' feature
    #cols = []
    #for f in list(connLogLabeled_DF.columns.values):
    #    if 'proto' in f:
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
    features_to_drop = ['ts', 'id.orig_h', 'id.resp_h', 'service', 'duration', 'orig_bytes', 'resp_bytes', 'local_orig', 'local_resp']

    # ----- Features to Keep
    # 'id.orig_p', 'id.resp_p', 'missed_bytes', 'orig_pkts', 'resp_pkts', 'resp_ip_bytes'

    #Features to Drop
    for feature in features_to_drop:
        connLogLabeled_DF = connLogLabeled_DF.drop(feature, axis=1)

    connLogLabeled_DF = connLogLabeled_DF.apply(pd.to_numeric) # ensure all features being used are numeric

    # sanity check
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
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
    # plt.show()
    #plt.tight_layout()
    #plt.savefig('correlation_matrix.png')

    # print(X[:5]) # sanity check: make sure X and y were created as expected 
    # print(y[:5])

    # ----- Plot number of malicious vs benign labels in dataset -----
    # count = 0
    # malicious_count = 0
    # benign_count = 0
    # for label in y:
    #     if (label == False):
    #         benign_count += 1
    #     else:
    #         malicious_count += 1

    # print("--------------------------------")
    # print("percent of data points labeled benign = ", (benign_count/len(y)) )
    # print("percent of data points labeled malicious = ", (1-(benign_count/len(y))) )
    # print("--------------------------------")

    # label = ['Benign', 'Malicious']
    # counts = [benign_count, malicious_count]
    # index = np.arange(len(label))
    # plt.bar(index, counts)
    # plt.xlabel('Label', fontsize=12)
    # plt.ylabel('# Data Points', fontsize=12)
    # plt.xticks(index, label, fontsize=10)
    # plt.title('Benign vs Malicious Data Points in Dataset')
    # plt.savefig('benignVsmaliciousCount.png')

    # ----- Print some initial accuracy results using 80/20 train/test split -----
    X = connLogLabeled_DF.loc[:, connLogLabeled_DF.columns != 'binary_label'].values # X contains the following features: ts, id.orig_p, id.resp_p, orig_ip_bytes, id.orig_h, id.resp_h, proto
    y = connLogLabeled_DF.loc[:, connLogLabeled_DF.columns == 'binary_label'].values.ravel() # Y contains just the binary label
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None) # use 80/20 test/train split
    train_test_accuracy(X_train, X_test, y_train, y_test)

    # ----- Evaluate highest performing model using 80-20 split run 10 times and 10-fold cross validation -----
    print("--------------------------------")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None) # use 80/20 test/train split

    accuracies = []
    for i in range(10): # Run 10 times with 80/20 split and collect statistics
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=None)
        accuracies += [rf_train_test(X_train, X_test, y_train, y_test)]
    print(f'80% train-test split accuracy using Random Forest is {np.mean(accuracies):.3f} {chr(177)}{np.std(accuracies):.4f}')

    accuracies = []
    kf = KFold(n_splits=10,shuffle=False,random_state=None) # Perform 10-fold cross validation
    for train_index, test_index in kf.split(X, y):
        acc = rf_train_test(X[train_index], X[test_index], y[train_index], y[test_index])
        accuracies += [acc]
    print(f'10-fold cross validation accuracy using Random Forest is {np.mean(accuracies):.3f} {chr(177)}{np.std(accuracies):.4f}')


    # captureLossLog_DF = broLog_to_df('CTU-IoT-Malware-Capture-34-1/bro/capture_loss.log')
    # dnsLog_DF = broLog_to_df('CTU-IoT-Malware-Capture-34-1/bro/dns.log')
    # httpLog_DF = broLog_to_df('CTU-IoT-Malware-Capture-34-1/bro/http.log')
    # ircLog_DF = broLog_to_df('CTU-IoT-Malware-Capture-34-1/bro/irc.log')

    # Check that dataframes are as expected
    # https://docs.zeek.org/en/current/script-reference/log-files.html
    # print("Conn Log:")
    # print(connLogLabeled_DF.head()) # Description: https://docs.zeek.org/en/current/scripts/base/protocols/conn/main.zeek.html#type-Conn::Info

    # cleanDF_util(connLogLabeled_DF)

    # print(connLogLabeled_DF.iloc[12330, :])

    # print("Capture Loss Log:")
    # print(captureLossLog_DF.head()) # Description: https://docs.zeek.org/en/current/scripts/policy/misc/capture-loss.zeek.html#type-CaptureLoss::Info
    # print("DNS Log:")
    # print(dnsLog_DF.head()) # Description: https://docs.zeek.org/en/current/scripts/base/protocols/dns/main.zeek.html#type-DNS::Info
    # print("http Log:")
    # print(httpLog_DF.head()) # Description: https://docs.zeek.org/en/current/scripts/base/protocols/http/main.zeek.html#type-HTTP::Info
    # print("irc Log:")
    # print(ircLog_DF.head()) # Description: https://docs.zeek.org/en/current/scripts/base/protocols/irc/main.zeek.html#type-IRC::Info

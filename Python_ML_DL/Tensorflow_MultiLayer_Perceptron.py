### Summary: Deep Learning - Custom Tensorflow MultiLayer Perceptron 
### Author: Gilles Kepnang

# ----- Importing libraries -----
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import copy
import csv
import pandas as pd
import sklearn as sk
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import zat
from zat.log_to_dataframe import LogToDataFrame
from zat.dataframe_to_matrix import DataFrameToMatrix
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

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

def encode_onehot(_df, f):
    """
    Summary: One-hot encodes the input categorical/nominal feature within the input dataframe.
    Returns updated dataframe with feature one-hot encoded.
    """
    _df2 = pd.get_dummies(_df[f], prefix='', prefix_sep='').max(level=0, axis=1).add_prefix(f+' - ')
    df3 = pd.concat([_df, _df2], axis=1)
    df3 = df3.drop([f], axis=1)
    return df3

def mlp_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels, n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode

def reduce2DMatrix(y_true, y_pred):
    y_return = np.arange(y_pred.shape[0])
    i = 0
    while i < y_true.shape[0]:
        if (y_true[i, 0] == 1):
            y_return[i] = y_pred[i, 0]
        else:
            y_return[i] = y_pred[i, 1]
        i = i+1

        #if(i == y_true.shape[0]):
        #    break
    return y_return

def convert_y_pred(y_pred):
    y_return = np.arange(y_pred.shape[0])
    i = 0
    while i < y_pred.shape[0]:
        if (y_pred[i, 0] == 1):
            y_return[i] = False
        else:
            y_return[i] = True
        i = i+1

        #if(i == y_true.shape[0]):
        #    break
    return y_return

## print stats
precision_scores_list = []
accuracy_scores_list = []

def print_stats_metrics(y_test, y_pred):
    print('Accuracy: %.2f' % accuracy_score(y_test,   y_pred) )
    #Accuracy: 0.84
    accuracy_scores_list.append(accuracy_score(y_test,   y_pred) )
    confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
    print ("confusion matrix")
    print(confmat)
    print (pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))
    precision_scores_list.append(precision_score(y_true=y_test, y_pred=y_pred, average='weighted', zero_division=0))
    print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred, average='weighted', zero_division=0))
    print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred, average='weighted', zero_division=0))
    print('F1-measure: %.3f' % f1_score(y_true=y_test, y_pred=y_pred, average='weighted', zero_division=0))

def updateDF(orig, dest, _df):

        #print(orig, " | ", dest)
    results = _df.loc[(_df['id.orig_h'] == orig) & (_df['id.resp_h'] == dest)]

    if results.shape[0] < 2:
        return _df

    prev_row = results.iloc[0]
    idx = 0
    for index, row in results.iterrows():
        if prev_row.equals(row) == False:     #Check that current row is not previous row
            if (row['duration'] != '-'):    #Confirm that next packet sent from that location had
                #print("Source #1: ", prev_row['id.orig_h'], " | Destinat1ion #1: ", prev_row['id.resp_h'],  " | Timestamp #1: ", prev_row['ts'])
                #print("Source #2: ", row['id.orig_h'], " | Destinat1ion #2: ", row['id.resp_h'],  " | Timestamp #2: ", row['ts'])
                interpacket_gap_time = row['ts'] - prev_row['ts'] #Retrieve ip_gap_time
                #print("Interpacket gap time is: ", interpacket_gap_time)

                results.at[idx, 'interpacket_gap_time'] = interpacket_gap_time #add i_g_t to results DF
                prev_row = copy.deepcopy(row) #Assign new previous copy
                idx = index                   #Assign index of new previous copy

    _df.update(results)#overwrite _df with results rows

    #two_rows = [row1, row2]

    return _df

if __name__ == "__main__":

    ################################################################################
    ##### Ingest and convert conn log zeek/bro file to Pandas dataframe (df) #######
    ################################################################################

    # Note: you will have to change the file paths here depending on how you named and where you've saved the files...

    # Ingest and convert the 20 malicious bro/zeek captures to dataframes
    # m1 = broLog_to_df('capstone_data/1-1/conn.log.labeled') # Hide and seek dataset (148 MB)

    m2 = broLog_to_df('3-1.conn.log.labeled') # Muhstik dataset (24 MB)

    # m3 = broLog_to_df('capstone_data/7-1/conn.log.labeled') # too big to read in on local machine

    m4 = broLog_to_df('8-1.conn.log.labeled') # Hakai dataset (1.4 MB)

    # m5 = broLog_to_df('capstone_data/9-1/conn.log.labeled') # too big to read in on local machine
    # m6 = broLog_to_df('capstone_data/17-1/conn.log.labeled') # too big to read in on local machine

    m7 = broLog_to_df('20-1.conn.log.labeled') # Torii dataset (420 KB)

    # m8 = broLog_to_df('capstone_data/21-1/conn.log.labeled')
    # m9 = broLog_to_df('capstone_data/33-1/conn.log.labeled') # too big to read in on local machine

    m10 = broLog_to_df('conn.log.labeled') # Mirai dataset (3 MB)

    # m11 = broLog_to_df('capstone_data/35-1/conn.log.labeled') # too big to read in on local machine
    # m12 = broLog_to_df('capstone_data/36-1/conn.log.labeled') # too big to read in on local machine
    # m13 = broLog_to_df('capstone_data/39-1/conn.log.labeled') # too big to read in on local machine

    m14 = broLog_to_df('42-1.conn.log.labeled') # Trojan dataset (586 KB)

    # m15 = broLog_to_df('capstone_data/43-1/conn.log.labeled') # too big to read in on local machine
    # m16 = broLog_to_df('capstone_data/44-1/conn.log.labeled')
    # m17 = broLog_to_df('capstone_data/48-1/conn.log.labeled')

    # m18 = broLog_to_df('capstone_data/49-1/conn.log.labeled')

    # m19 = broLog_to_df('capstone_data/52-1/conn.log.labeled') # too big to read in on local machine

    # m20 = broLog_to_df('capstone_data/60-1/conn.log.labeled')

    # Ingest and convert the 3 benign bro/zeek captures to dataframes
    b1 = broLog_to_df('conn.log.normalAlexa.labeled') # 182 KB
    b2 = broLog_to_df('conn.log.normalPhillips.labeled') # 62 KB
    b3 = broLog_to_df('conn.log.normalSoomfy.labeled') # 18 KB

    print("done ingesting")

    ################################################################################
    ##### Concatenate the dataframes into a single dataframe #######################
    ################################################################################

    # frames = [m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15, m16, m17, m18, m19, m20, b1, b2, b3]
    frames = [m2, m4, m7, m10, m14, b1, b2, b3]
    connLogLabeled_DF = pd.concat(frames)

    #connLogLabeled_DF.to_csv (r'C:\Users\Gilles\PycharmProjects\Capstone\conn_log_dataframe.csv', index = False, header=True)
    #################################################################################
    ##### Create new features from existing features and append to dataframe ########
    #################################################################################

    # Note: From the above step, we see there are 4 labels (multi-class) within this dataset: benign, malicious C&C, malicious DDoS, malicious PartOfAHorizontalPortScan
    # We can create an additional column with binary labels (i.e. 'benign' vs 'malicious') to have the flexibility to consider this as a binary classification problem as well
    connLogLabeled_DF.insert(0, 'binary_label', True) # add new column to the end of the dataset for the binary label (True = malicious, False = benign)

    for index, row in connLogLabeled_DF.iterrows(): # iterate through dataset and populate binary label as malicious or benign
        current_mclassLabel = row['tunnel_parents   label   detailed-label']
        if ('Benign' in current_mclassLabel or 'benign' in current_mclassLabel):
            connLogLabeled_DF.at[index, 'binary_label'] = False

   # Create inter-packet spacing feature
    connLogLabeled_DF.insert(21, 'packet_spacing', 0) # add new column to the end of the dataset for spacing

    current_row = connLogLabeled_DF.at[0,'ts'] # initialize current row value

    for index, row in connLogLabeled_DF.iterrows(): # iterate through dataset and calculate and populate interpacket spacing value
        prev_row = copy.deepcopy(current_row)
        current_row = row['ts']
        time_delta = current_row - prev_row
        connLogLabeled_DF.at[index, 'packet_spacing'] = time_delta

    #Create average bytes for each transaction, based on current row
    connLogLabeled_DF.insert(22, 'interpacket_gap_time', 0)

    stored_pairs = []
    #Iterate through dataset
    for index, row in connLogLabeled_DF.iterrows():
        if (connLogLabeled_DF.shape[0] - index) > 1 and stored_pairs.count([row['id.orig_h'], row['id.resp_h']]) < 1:
            print(index)
            connLogLabeled_DF = updateDF(row['id.orig_h'], row['id.resp_h'], connLogLabeled_DF)
            stored_pairs.append([row['id.orig_h'], row['id.resp_h']])

    ################################################################################
    ##### One-hot encode the categorical (i.e. non-numerical) features  ############
    ################################################################################

    # Assume following features are used : id.orig_p, id.resp_p

    # Experiment 4 features: missed_bytes, orig_ip_bytes, resp_pkts, resp_ip_bytes, packet_spacing, history, conn_state
    # Domain knowledge features: conn.state, resp_ip_bytes, proto, history,
    # Weka features: orig_ip_bytes, history, orig_pkts
    # Domain + Weka features: conn.state, resp_ip_bytes, proto, history, orig_ip_bytes, orig_pkts
    # Domain knowledge features + packet spacing: conn.state, resp_ip_bytes, proto, history, packet_spacing
    # Weka features + packet spacing: orig_ip_bytes, history, orig_pkts, packet_spacing
    # Domain + Weka features + packet spacing: conn.state, resp_ip_bytes, proto, history, orig_ip_bytes, orig_pkts, packet_spacing

    # One-hot encode the nominal/categorical columns we'll be using as features
    connLogLabeled_DF = encode_onehot(connLogLabeled_DF, 'proto') # one-hot encode 'proto' feature
    cols = []
    for f in list(connLogLabeled_DF.columns.values):
        if 'proto' in f:
            cols += [f]

    connLogLabeled_DF = encode_onehot(connLogLabeled_DF, 'history') # one-hot encode 'd.orig_h' feature
    cols = []
    for f in list(connLogLabeled_DF.columns.values):
        if 'history' in f:
            cols += [f]

    connLogLabeled_DF = encode_onehot(connLogLabeled_DF, 'conn_state') # one-hot encode 'd.orig_h' feature
    cols = []
    for f in list(connLogLabeled_DF.columns.values):
        if 'conn_state' in f:
            cols += [f]


    # ----- Set the features the model will use by dropping the ones we don't want to use -----
    features_to_drop = ['ts', 'uid', 'id.orig_h', 'id.resp_h', 'service', 'duration', 'orig_bytes', 'resp_bytes', 'local_orig', 'local_resp', 'tunnel_parents   label   detailed-label']

    # ----- Features to Keep
    # 'id.orig_p', 'id.resp_p',  'missed_bytes', 'orig_pkts', 'orig_ip_bytes', 'resp_pkts', 'resp_ip_bytes'

    for feature in features_to_drop:
        connLogLabeled_DF = connLogLabeled_DF.drop(feature, axis=1)

    connLogLabeled_DF = connLogLabeled_DF.apply(pd.to_numeric) # ensure all features being used are numeric

    # sanity check
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    print(connLogLabeled_DF.head())

    ################################################################################
    ##### Create correlation matrix  ###############################################
    ################################################################################

    # # ----- Calculate correlation of each feature to the binary label (i.e. the target variable) and visualize results -----
    # # Keep the target variable in the X
    # X = connLogLabeled_DF.values
    # N, M = X.shape

    # # Average of each feature, i.e. sample mean
    # Xavg = np.zeros(M)
    # for i in range(M):
    #     Xavg[i] = np.sum(X[:,i]) / N

    # # Stdev of each feature, i.e. sample standard deviation
    # Xvar = np.zeros(M)
    # for i in range(M):
    #     Xvar[i] = np.sqrt(np.sum((X[:,i]-Xavg[i])**2))

    # # Correlation
    # Xcorr = np.zeros((M,M))
    # for i in range(M):
    #     for j in range(M):
    #         Xcorr[i,j] = np.sum((X[:,i]-Xavg[i])*(X[:,j]-Xavg[j])) / (Xvar[i]*Xvar[j])

    # # Plot and save figure
    # plt.imshow(Xcorr, cmap='hsv', interpolation='nearest')
    # plt.yticks(np.arange(M), labels=connLogLabeled_DF.columns, fontsize=5)
    # plt.xticks(np.arange(M), labels=connLogLabeled_DF.columns, rotation=90, fontsize=5)
    # plt.colorbar()
    # # plt.show()
    # plt.tight_layout()
    # plt.savefig('correlation_matrix.png')

    ################################################################################
    ##### Plot number of benign vs malicious datapoints in dataset  ################
    ################################################################################

     # Create 2 dataframes: one will hold all malicious labeled data, other will hold all benign labeled data
    df_benign = connLogLabeled_DF[connLogLabeled_DF.binary_label == False]
    df_malicious = connLogLabeled_DF[connLogLabeled_DF.binary_label == True]

    random_malicious_rows = df_malicious.sample(n=1777)
    connLogLabeled_DF = pd.concat([df_benign, random_malicious_rows], axis=0) # put all benign datapoints (# benign pts = 33779) in dataframe and the randomly selected malicious data rows (# malicious pts = 1777)

    # ----- Create X and Y vectors -----
    X = connLogLabeled_DF.loc[:, connLogLabeled_DF.columns != 'binary_label'].values # X can contains all features minus the features that were dropped minus the label
    y = connLogLabeled_DF.loc[:, connLogLabeled_DF.columns == 'binary_label'].values.ravel() # Y contains just the binary labelclea

    count = 0
    malicious_count = 0
    benign_count = 0
    for label in y:
        if (label == False):
            benign_count += 1
        else:
            malicious_count += 1
    print("--------------------------------")
    print("percent of data points labeled benign = ", (benign_count/len(y)) )
    print("number of benign pts: ", benign_count)
    print("percent of data points labeled malicious = ", (1-(benign_count/len(y))) )
    print("--------------------------------")

    #label = ['Benign', 'Malicious']
    #counts = [benign_count, malicious_count]
    #index = np.arange(len(label))
    #plt.bar(index, counts)
    #plt.xlabel('Label', fontsize=12)
    #plt.ylabel('# Data Points', fontsize=12)
    #plt.xticks(index, label, fontsize=10)
    #plt.title('Benign vs Malicious Data Points in Dataset')
    #plt.savefig('benignVsmaliciousCount.png')

    ################################################################################
    ##### Run models and print results  ############################################
    ################################################################################
    # ----- Create X and Y vectors -----
    X = connLogLabeled_DF.loc[:, connLogLabeled_DF.columns != 'binary_label'].values # X can contains all features minus the features that were dropped minus the label
    y = connLogLabeled_DF.loc[:, connLogLabeled_DF.columns == 'binary_label'].values.ravel() # Y contains just the binary label

    # ----- Print some initial accuracy results using 80/20 train/test split -----
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None) # use 80/20 test/train split


    #####################################################################################
    ### AUTHOR: GILLES KEPNANG
    #####################################################################################
    ###                  MULTILAYER PERCEPTRON IMPLEMENTATION (MLP)
    #####################################################################################
    # Perform a shape-check of the training arrays and testing arrays
    print("--------------------------------")
    print("shape of X-train = ", X_train.shape )
    print("shape of Y-train = ", y_train.shape )
    print("shape of X-test = ", X_test.shape )
    print("--------------------------------")

    print(tf.__version__)

    #####################################################################################
    ### BEFORE MLP
    ###   Encode the singular columns of y-train and y-test
    #####################################################################################
    print("-------------------------------------")
    print("     SPECIAL ENCODE FOR COlUMN Y     ")
    print("-------------------------------------")

    #Apply secondary encoding and labeling to the y_train
    encoder = LabelEncoder()  #initialize encoder (for y_train)
    encoder.fit(y_train)      #fit encoder to column of y_train
    y1 = encoder.transform(y_train) #provide encoded labels again
    Y_train = mlp_encode(y1)  #generate n columns, each with unique encoding (where n is the number of classes)

    #Apply secondary encoding and labeling to the y_test
    encoder2 = LabelEncoder() #initialize encoder (for y_test)
    encoder2.fit(y_test)      #fit encoder to column of y_test
    y2 = encoder2.transform(y_test) #provide encoded labels again
    Y_test = mlp_encode(y2)   #generate n columns, each with unique encoding (where n is the number of classes)

    print("--------------------------------")
    print("shape of encoded Y-train = ", Y_train.shape )
    print("shape of encoded Y-test = ", Y_test.shape )
    print("--------------------------------")

    #####################################################################################
    ### NEXT UP:
    #####################################################################################
    # 1) Define hyperparameters
    # Tensor parameters and Tensor variables
    rate = 0.4
    epochs = 500
    cost_history = np.empty(shape=[1], dtype=float) #this is a loss function (creates an array object with 1-dimension and float data type)
    n_dimensions = X_train.shape[1]
    n_classes = Y_train.shape[1]
    model_path = "MPL_model"

    #   Assign only hidden layers (with nodes 24-16-8)
    n_hidden1 = 12
    n_hidden2 = 8
    n_hidden3 = 4

    #   Assign x and y_ placeholders
    x = tf.placeholder(tf.float32, [None, n_dimensions])
    y_ = tf.placeholder(tf.float32, [None, n_classes])

def multi_layer_perceptron(x, weights, biases):
   """
   Summary:    Define MLP model (hidden layers and output layers have both weights & biases)
   Return:     Array for output layer (and its associated nodes)
   """
   layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
   layer_1 = tf.nn.sigmoid(layer_1)

   layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
   layer_2 = tf.nn.sigmoid(layer_2)

   layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
   layer_3 = tf.nn.relu(layer_3)

   out_layer = tf.matmul(layer_3, weights['out']) + biases['out']
   return out_layer

 # 2) Assign weights
weights = {
    'h1': tf.Variable(tf.truncated_normal([n_dimensions, n_hidden1])),
    'h2': tf.Variable(tf.truncated_normal([n_hidden1, n_hidden2])),
    'h3': tf.Variable(tf.truncated_normal([n_hidden2, n_hidden3])),
    'out': tf.Variable(tf.truncated_normal([n_hidden3, n_classes]))
}

 # 3) Create Bias
biases = {
    'b1': tf.Variable(tf.truncated_normal([n_hidden1])),
    'b2': tf.Variable(tf.truncated_normal([n_hidden2])),
    'b3': tf.Variable(tf.truncated_normal([n_hidden3])),
    'out': tf.Variable(tf.truncated_normal([n_classes]))
}

 # Initialize variables and create saver
init = tf.global_variables_initializer()
saver = tf.train.Saver()

y_mlp = multi_layer_perceptron(x, weights, biases)

print("--------------------------------")
print("shape of x = ", x.shape )
print("shape of y_ = ", y_.shape )
print("--------------------------------")

print("--------------------------------")
print("shape of Y-MLP = ", y_mlp.shape)
print("--------------------------------")

# Computes mean of error-cost
# --> "softmax_cross_entropy_" will measure probability error between classes;
#                               also perform backpropagation in logits and labels
cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_mlp, labels=y_))
# Determine vector for weigh adjustment (based on gradient descent)
training_step = tf.train.GradientDescentOptimizer(rate). minimize(cost_function)

tensorflowSession = tf.Session()
tensorflowSession.run(init)

# Initialize history arrays for 1) mean-square-history and 2) accuracy
mean_sq_error_history = []
accuracy_history = []

for epoch in range(epochs):
    #Feed training data
    tensorflowSession.run(training_step, feed_dict={x: X_train, y_: Y_train})
    #Run loss function (or cost)
    cost = tensorflowSession.run(cost_function, feed_dict={x: X_train, y_: Y_train})
    #Add cost result to history var
    cost_history = np.append(cost_history, cost)
    #Check correct prediction
    correct_prediction = tf.equal(tf.argmax(y_mlp, 1), tf.argmax(y_, 1))
    #Assign accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    #Reduce mean square error (actual output - expected output)
    pred_y = tensorflowSession.run(y_mlp, feed_dict={x: X_test})
    mean_sq_error = tf.reduce_mean(tf.square(pred_y - Y_test))
    mean_sq_error_results = tensorflowSession.run(mean_sq_error)
    mean_sq_error_history.append(mean_sq_error_results)
    accuracy = (tensorflowSession.run(accuracy, feed_dict={x: X_train, y_: Y_train}))
    accuracy_history.append(accuracy)
    print('epoch: ', epoch, ' - ', 'cost: ', cost, " - MSE: ", mean_sq_error_results)
    print("Train Accuracy: ", accuracy)

save_path = saver.save(tensorflowSession, model_path)
print("Model saved in file: %s" % save_path)

#Plot mse and accuracy graph
#plt.plot(mean_sq_error_history, 'r')
#plt.show()
#plt.plot(accuracy_history)
#plt.show()

#Print the final accuracy
correct_prediction = tf.equal(tf.argmax(y_mlp, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#Print the final mean square error
pred_y = tensorflowSession.run(y_mlp, feed_dict={x: X_test})
mean_sq_error = tf.reduce_mean(tf.square(pred_y - Y_test))
print("MSE: %.4f" % tensorflowSession.run(mean_sq_error))

#tensorY = tf.argmax(y_mlp, 1)
accuracy = (tensorflowSession.run(accuracy, feed_dict={x: X_test, y_: Y_test}))
print("Test Accuracy: ", accuracy)

output1 = tf.nn.softmax(y_mlp, name="output")
y_pred = tensorflowSession.run(output1, feed_dict={x: X_test})
y_pred = y_pred.round()
y_pred = y_pred.astype(int)
y_pred = convert_y_pred(y_pred)

print(y_test)
print(y_pred)

print('All Features used')
print_stats_metrics(y_test, y_pred)

#print("Precision : ", sk.metrics.precision_score(y_test.astype(int), y_pred))
#print("Recall: ", sk.metrics.recall_score(y_test.astype(int), y_pred))
#print("F1-Score: ", sk.metrics.f1_score(y_test.astype(int), y_pred))

# Plotting the confusion matrix using matplotlib

#confusion = sk.metrics.confusion_matrix(y_test.astype(int), y_pred)
#print("Confusion Matrix - ")
#print("    ", confusion)
# Plot non-normalized confusion matrix
#plt.figure()
#sk.metrics.plot_confusion_matrix(confusion, X_test, y_test)
#title=('Confusion matrix, without normalization')


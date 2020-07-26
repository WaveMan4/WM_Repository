### Summary: Deep Learning - Tensorflow Keras MLL
### Author: Gilles Kepnang

# ----- Importing libraries -----
import pandas as pd
import sklearn
import numpy as np
import csv
import matplotlib
import matplotlib.pyplot as plt
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam

# Initialize the MLP
def initialize_nn(frame_size):
    model = Sequential() # The Keras Sequential model is a linear stack of layers.
    model.add(Dense(200, init='uniform', input_dim=frame_size)) # Dense layer
    model.add(Activation('tanh')) # Activation layer
    model.add(Dropout(0.5)) # Dropout layer
    model.add(Dense(200, init='uniform')) # Another dense layer
    model.add(Activation('tanh')) # Another activation layer
    model.add(Dropout(0.5)) # Another dropout layer
    model.add(Dense(2, init='uniform')) # Last dense layer
    model.add(Activation('softmax')) # Softmax activation at the end
    #===============================================================================================
    #   SGD Optimizer
    #===============================================================================================
    #sgd = SGD(learning_rate=0.1, decay=1e-6, momentum=0.9, nesterov=True) # Using Nesterov momentum
    #model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy']) # Using logloss

    #===============================================================================================
    #   RMS Optimizer
    #===============================================================================================
    #rms = RMSprop(learning_rate=0.001, rho=0.9)
    #model.compile(loss='binary_crossentropy', optimizer=rms, metrics=['accuracy']) # Using logloss

    #===============================================================================================
    #   Adagrad Optimizer
    #===============================================================================================
    #adg = Adagrad(learning_rate=0.01)
    #model.compile(loss='binary_crossentropy', optimizer=adg, metrics=['accuracy']) # Using logloss

    #===============================================================================================
    #   Adadelta Optimizer
    #===============================================================================================
    #dta = Adadelta(learning_rate=1.0, rho=0.95)
    #model.compile(loss='binary_crossentropy', optimizer=dta, metrics=['accuracy']) # Using logloss

    #===============================================================================================
    #   Adam Optimizer
    #===============================================================================================
    #dam = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    #model.compile(loss='binary_crossentropy', optimizer=dam, metrics=['accuracy']) # Using logloss

    #===============================================================================================
    #   Adamax Optimizer
    #===============================================================================================
    max = Adamax(learning_rate=0.002, beta_1=0.9, beta_2=0.999)
    model.compile(loss='binary_crossentropy', optimizer=max, metrics=['accuracy']) # Using logloss

    #===============================================================================================
    #   Nadam Optimizer
    #===============================================================================================
    #ndm = Nadam(learning_rate=0.002, beta_1=0.9, beta_2=0.999)
    #model.compile(loss='binary_crossentropy', optimizer=ndm, metrics=['accuracy']) # Using logloss

    return model

def generate_results(y_test, y_score):
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic curve')
    plt.show()
    print('AUC: %f' % roc_auc)

def mlp_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels, n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode

if __name__ == "__main__":

    # ----- Read in the data -----
    connLogLabeled_DF = pd.read_csv('exp4_dataset.csv')

    # ----- Preprocess and clean the data -----
    #preprocess_df(connLogLabeled_DF) # start to preprocess conn log df

    # ----- Set the features the model will use by dropping the ones we don't want to use -----
    features_to_drop = ['ts', 'id.orig_h', 'id.resp_h', 'service', 'duration', 'orig_bytes', 'resp_bytes', 'local_orig', 'local_resp']

    #Features to Drop
    for feature in features_to_drop:
        if (feature in connLogLabeled_DF.columns):
            connLogLabeled_DF = connLogLabeled_DF.drop(feature, axis=1)
    connLogLabeled_DF = connLogLabeled_DF.apply(pd.to_numeric) # ensure all features being used are numeric

    # sanity check
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    X = connLogLabeled_DF.loc[:, connLogLabeled_DF.columns != 'binary_label'].values # X contains the following features: ts, id.orig_p, id.resp_p, orig_ip_bytes, id.orig_h, id.resp_h, proto
    y = connLogLabeled_DF.loc[:, connLogLabeled_DF.columns == 'binary_label'].values.ravel() # Y contains just the binary label

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

    print("frame-size is ", X.shape[1])
    model = initialize_nn(X.shape[1])

    #Apply secondary encoding and labeling to the y_train
    encoder = LabelEncoder()  #initialize encoder (for y_train)
    encoder.fit(y_train)      #fit encoder to column of y_train
    y1 = encoder.transform(y_train) #provide encoded labels again
    y_train = mlp_encode(y1)  #generate n columns, each with unique encoding (where n is the number of classes)

    #Apply secondary encoding and labeling to the y_test
    encoder2 = LabelEncoder() #initialize encoder (for y_test)
    encoder2.fit(y_test)      #fit encoder to column of y_test
    y2 = encoder2.transform(y_test) #provide encoded labels again
    y_test = mlp_encode(y2)   #generate n columns, each with unique encoding (where n is the number of classes)

    print('Training model')
    model.fit(X_train, y_train,
              batch_size=32, epochs=3000,
              verbose=1, callbacks=[],
              validation_data=None, shuffle=True,
              class_weight=None, sample_weight=None)

    print('Predicting on test data')
    y_score = model.predict(X_test)

    print('Generating results')
    generate_results(y_test[:, 0], y_score[:, 0])

    print('=========================')
    print(' Just finished max model')
    print('=========================')

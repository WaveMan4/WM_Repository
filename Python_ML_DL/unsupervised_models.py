### Summary: Unsupervised Learning models
### EN605.795.8VL.SP20
### Author: Gilles Kepnang

# ----- Importing libraries -----
import pandas as pd
import sklearn
import numpy as np
import csv
import matplotlib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns; sns.set()  # for plot styling

if __name__ == "__main__":

    # ----- Read in the data -----
    connLogLabeled_DF = pd.read_csv('exp4_capstone_dataset.csv')

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

    kmeans = KMeans(n_clusters=2) # You want cluster the binary values into two values: 'True' or 'False'
    kmeans.fit(X)

    y_kmeans = kmeans.predict(X)
    #plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
    #plt.show()
    centers = kmeans.cluster_centers_
    #plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
    #plt.show()

    correct = 0
    incorrect = 0
    predictions = []
    for i in range(len(X)):
        predict_me = np.array(X[i].astype(float))
        predict_me = predict_me.reshape(-1, len(predict_me))
        prediction = kmeans.predict(predict_me)
        predictions.append(kmeans.predict(predict_me))

        if prediction[0] == y[i]:
            correct += 1
        else:
            incorrect += 1

    #print()
    #print("K-Means Clustering: ")
    #print()
    #print("%4d correct points out of %5d data points" %(correct, (len(X) * len(predict_me))))
    #print("  percentage of correct: ")
    #print("    ", correct/len(X))
    #print()
    #print("%4d incorrect points out of %5d data points" %(incorrect, (len(X) * len(predict_me))))
    #print("  percentage of incorrect: ")
    #print("    ", incorrect/len(X))

    pred_y = np.array(predictions)

    print("  K-Means accuracy is: ", accuracy_score(predictions, y))
    print("  K-Means confusion matrix below")
    mat = confusion_matrix(predictions, y)
    sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
    plt.xlabel('true label')
    plt.ylabel('predicted label')
    plt.show()

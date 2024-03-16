#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 16:12:41 2022
Modified on Mon Mar 13 2023

@author: CHANGE THE NAME

This script carries out a classification experiment of the spambase dataset by
means of the kNN classifier, USING THE SCIKIT-LEARN PACKAGE
"""

# Import whatever else you need to import
import pandas as pd
import os
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# 
# -------------
# MAIN PROGRAM
# -------------
if __name__ == "__main__":


    # Load csv with data into a pandas dataframe
    """
    Each row in this dataframe contains a feature vector, which represents an
    email.
    Each column represents a variable, EXCEPT LAST COLUMN, which represents
    the true class of the corresponding element (i.e. row): 1 means "spam",
    and 0 means "not spam"
    """
    dir_data = "Data"
    spam_df = pd.read_csv(os.path.join(dir_data, "spambase_data.csv"))
    y_df = spam_df[['Class']].copy()
    X_df = spam_df.copy()
    X_df = X_df.drop('Class', axis=1)

    # Convert dataframe to numpy array
    X = X_df.to_numpy()
    y = y_df.to_numpy()
    
    """
    Parameter that indicates the proportion of elements that the test set will
    have
    """
    proportion_test = 0.3

    """
    Partition of the dataset into training and test sets is done. 
    Use the function train_test_split from scikit_learn
    """
    # ====================== YOUR CODE HERE ======================
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=proportion_test, random_state=42)
    # ============================================================

    """
    Create an instance of the kNN classifier using scikit-learn
    """
    # ====================== YOUR CODE HERE ======================
    k = 5  # Number of neighbors
    metric = 'euclidean'  # Distance metric
    knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
    # ============================================================

    """
    Train the classifier
    """
    # ====================== YOUR CODE HERE ======================
    knn.fit(X_train, y_train)
    # ============================================================

    """
    Get the predictions for the test set samples given by the classifier
    """
    # ====================== YOUR CODE HERE ======================
    y_pred = knn.predict(X_test)
    # ============================================================
    
    """
    Show the confusion matrix. Use the same methods that were used in the
    first part of the lab (i.e., see Lab5.py)
    """
    # ====================== YOUR CODE HERE ======================
    confusion_matrix_kNN = confusion_matrix(y_true=y_test, y_pred=y_pred)
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(confusion_matrix_kNN, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(confusion_matrix_kNN.shape[0]):
        for j in range(confusion_matrix_kNN.shape[1]):
            ax.text(x=j, y=i, s=confusion_matrix_kNN[i, j], va='center',
                    ha='center', size='xx-large')

    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    plt.show()
    # ============================================================
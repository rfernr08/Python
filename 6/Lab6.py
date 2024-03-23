#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 16:02:27 2022
Modified on March 2024

@author: YOUR NAME HERE

"""

import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
import numpy as np
import h5py
import matplotlib.pyplot as plt
from math import exp, log


def calculate_cost_log_reg(y, y_hat):
    """
    Calculates the cost of the OUTPUT OF THE PATTERNS OF THE WHOLE TRAINING SET
    from the logistic regression classifier (i.e. the result of applying the h
    function to all the patterns in the training set) and their real classes.

    Parameters
        ----------
        y: numpy vector
            Vector with the real classes of the training set.
        y_hat: numpy vector
            Output of the h function (i.e. the hypothesis of the logistic
            regression classifier for every element of the training set.
         ----------

    Returns
        -------
        cost_set: float
            Value of the cost of the estimated outputs of the training set.
        -------
    """

    # ====================== YOUR CODE HERE ======================
    # cost_set = (-1 / m) * np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
    cost_set = (-1 / y) * np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
    # ============================================================

    return cost_set


# **************************************************************************
# **************************************************************************
def fun_sigmoid(theta_sigmoid, x):
    """
    TThis function calculates the sigmoid function g(z), where z is a linear
    combination of the parameters theta and the components of feature vector x

    Parameters
        ----------
        theta_sigmoid: numpy vector
            Parameters of the h function of the logistic regression classifier.
        x: Numpy vector
            Vector containing the data of one pattern.
         ----------

    Returns
        -------
        g: float
            Result of applying the sigmoid function using the linear
            combination of theta and X.
        -------
    """
    # ====================== YOUR CODE HERE ======================
    g = 1 / (1 + exp(-np.dot(theta_sigmoid, x)))
    # ============================================================

    return g


# **************************************************************************
# **************************************************************************
def train_logistic_regression(X_train, y_train, alpha_train,
                              verbose=True, max_iter=100):
    """
    This function implements the training of a logistic regression classifier
    using the training data (X_train) and its classes (y_train).

    Parameters
        ----------
        X_train: Numpy array
            Matrix with dimensions (m x n) with the training data, where m is
            the number of training patterns (i.e. elements) and n is the number
            of features (i.e. the length of the feature vector which
            characterizes the object).
        y_train: Numpy vector
            Vector that contains the classes of the training patterns. Its
            length is n.
        alpha_train: float
            Scalar that contains the learning rate

    Returns
        -------
        theta: numpy vector
            Vector with length n (i.e, the same length as the number of
            features on each pattern). It contains the parameters theta of the
            hypothesis function obtained after the training.

    """

    # Number of training patterns.
    m = np.shape(X_train)[0]

    # Allocate space for the outputs of the hypothesis function for each
    # training pattern
    h_train = np.zeros(shape=m)

    # Allocate spaces for the values of the cost function on each iteration
    cost_values = np.zeros(shape=(1 + max_iter))

    # Initialize the vector to store the parameters of the hypothesis function
    # All values are in the initialization are zero
    # heta_train = np.zeros(shape=(1, 1 + np.shape(x_train)[1]))  -> all zero
    # All values in the intialization fall within the interval [a, b)
    a = -10
    b = 10
    theta_train = np.random.uniform(low=a,
                                    high=b,
                                    size=(1 + np.shape(X_train)[1]))

    # -------------
    # CALCULATE THE VALUE OF THE COST FUNCTION FOR THE INITIAL THETAS
    # -------------
    # a. Intermediate result: Get the estimation (i.e. logistic regression
    # output) for each element
    for i in range(m):
        # Add a 1 (i.e., the value for x0) at the beginning of each pattern
        x_i = np.insert(np.array([X_train[i]]), 0, 1)

        # Expected output (i.e. result of the sigmoid function) for i-th
        # pattern, and store it in h_train for future use
        # ====================== YOUR CODE HERE ======================
        h_train[i] = fun_sigmoid(theta_train, x_i)
        # ============================================================

    # b. Calculate the cost
    # ====================== YOUR CODE HERE ======================
    cost_values[0] = calculate_cost_log_reg(y_train, h_train)
    # ============================================================

    # -------------
    # GRADIENT DESCENT ALGORITHM TO UPDATE THE THETAS
    # -------------
    # Iterative method carried out during a maximum number (max_iter) of
    # iterations
    for num_iter in range(max_iter):

        # ------
        # STEP 1. Update the theta values. To do it, follow the update 
        # equations studied in the theoretical sessions.
        #
        # REMEMBER THAT, AT THIS POINT OF THE LOOP, THE ESTIMATIONS GIVEN BY
        # THE SIGMOID FUNCTION ARE ALREADY CALCULATED AND STORED IN h_train
        # 
        # a. Intermediate result: Calculate the (h_i-y_i)*x for EACH element
        # from the training set
        aux = 0
        theta_old = np.copy(theta_train)
        for i in range(m):
            # Add a 1 (i.e., the value for x0) at the beginning of each pattern
            x_i = np.insert(np.array([X_train[i]]), 0, 1)

            # ====================== YOUR CODE HERE ======================
            aux += (h_train[i] - y_train[i]) * x_i
            # ============================================================

        # b. Update of the thetas
        # ====================== YOUR CODE HERE ======================
        theta_train = theta_old - alpha_train * aux
        # ============================================================
        
        
        # ------
        # STEP 2: Calculate the cost on this iteration and store it on
        # vector cost_values.
        # a. Intermediate result: Get the error for each element to sum it up.
        for i in range(m):
            # Add a 1 (i.e., the value for x0) at the beginning of each pattern
            x_i = np.insert(np.array([X_train[i]]), 0, 1)

            # Expected output (i.e. result of the sigmoid function) for i-th
            # pattern, and store it in h_train for future use
            # ====================== YOUR CODE HERE ======================
            h_train[i] = fun_sigmoid(theta_train, x_i)
            # ============================================================

        # b. Calculate the cost
        # ====================== YOUR CODE HERE ======================
        cost_values[num_iter+1] = calculate_cost_log_reg(y_train, h_train)
        # ============================================================

        '''
        EARLY STOP CRITERION: If the absolute value of the cost in the actual
        iteration with respect to the previous iteration is lower than 0.0005,
        stop the training
        '''
        # ====================== YOUR CODE HERE ======================
        if abs(cost_values[num_iter] - cost_values[num_iter+1]) < 0.0005:
            break
        # ============================================================


    # If verbose is True, plot the cost as a function of the iteration number
    if verbose:
        plt.plot(cost_values, color='red')
        plt.xlabel('Number of iterations')
        plt.ylabel('Cost J')
        plt.title(r'Cost function over the iterations with $\alpha=${}'.
                  format(alpha), fontsize=14)
        plt.show()

    return theta_train


# **************************************************************************
# **************************************************************************
def classify_logistic_regression(X_test, theta_classif):
    """
    This function returns the probability for each pattern of the test set to
    belong to the positive class using the logistic regression classifier.

    Parameters
        ----------
        X_test: Numpy array
            Matrix with dimension (m_t x n) with the test data, where m_t
            is the number of test patterns and n is the number of features
            (i.e. the length of the feature vector that define each element).
        theta_classif: numpy vector
            Parameters of the h function of the logistic regression classifier.

    Returns
        -------
        y_hat: numpy vector
            Vector of length m_t with the estimations made for each test
            element by means of the logistic regression classifier. These
            estimations corredspond to the probabilities that these elements
            belong to the positive class.
    """

    num_elem_test = np.shape(X_test)[0]
    y_hat = np.zeros(shape=(num_elem_test, 1))

    for i in range(num_elem_test):
        # Add a 1 (value for x0) at the beginning of each pattern
        x_test_i = np.insert(np.array([X_test[i]]), 0, 1)
        # ====================== YOUR CODE HERE ======================
        y_hat[i] = fun_sigmoid(theta_classif, x_test_i)
        # ============================================================

    return y_hat


# **************************************************************************
# **************************************************************************
# -------------
# MAIN PROGRAM
# -------------
if __name__ == "__main__":

    np.random.seed(324)
    
    plt.close('all')

    dir_data = "Data"
    data_path = os.path.join(dir_data, "mammographic_data.h5")
    test_size = 0.3
    decision_treshold = 0.5

    # -------------
    # PRELIMINARY: LOAD DATASET AND PARTITION TRAIN-TEST SETS (NO NEED TO
    # CHANGE ANYTHING)
    # -------------

    # Import data from the csv file
    '''
    # Import data from the csv using pandas
    mammographic_data_df = pd.read_csv(data_path)
    y_df = mammographic_data_df[['Class']].copy()
    X_df = mammographic_data_df.copy()
    X_df = X_df.drop('Class', axis=1)

    X = X_df.to_numpy()
    y = y_df.to_numpy().flatten()
    '''

    # Import data from the h5 file
    # import features and labels
    h5f_data = h5py.File(data_path, 'r')

    features_ds = h5f_data['data']
    labels_ds = h5f_data['labels']

    X = np.array(features_ds)
    y = np.array(labels_ds).flatten()

    h5f_data.close()

    # SPLIT DATA INTO TRAINING AND TEST SETS
    # ====================== YOUR CODE HERE ======================
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    # ============================================================

    # STANDARDIZE DATA
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # print("Mean of the training set: {}".format(X_train.mean(axis=0)))
    # print("Std of the training set: {}".format(X_train.std(axis=0)))
    # print("Mean of the test set: {}".format(X_test.mean(axis=0)))
    # print("Std of the test set: {}".format(X_test.std(axis=0)))

    # -------------
    # PART 1: TRAINING OF THE CLASSIFIER AND CLASSIFICATION OF THE TEST SET
    # -------------

    # TRAINING

    # Learning rate. Change it accordingly, depending on how the cost function
    # evolve along the iterations
    alpha = 2

    # The function fTrain_LogisticReg implements the logistic regression
    # classifier. Open it and complete the code.
    theta = train_logistic_regression(X_train, y_train, alpha)

    # print(theta)

    # -------------
    # CLASSIFICATION OF THE TEST SET
    # -------------
    y_test_hat = classify_logistic_regression(X_test, theta)

    # Assignation of the class
    y_test_assig = y_test_hat >= decision_threshold

    # -------------
    # PART 2: PERFORMANCE OF THE CLASSIFIER: CALCULATION OF THE ACCURACy AND FSCORE
    # -------------

    # Show confusion matrix
    # y_test = np.array([y_test.astype(bool)])
    # confm = confusion_matrix(y_test.T, y_test_assig.T)
    confm = confusion_matrix(y_true=y_test, y_pred=y_test_assig)
    print(confm)
    # classNames = np.arange(0,1)
    # disp = ConfusionMatrixDisplay(confusion_matrix=confm,display_labels=classNames)
    disp = ConfusionMatrixDisplay(confusion_matrix=confm)
    disp.plot()
    plt.title('Confusion Matrix', fontsize=14)
    plt.show()

    # -------------
    # PART 3: ACCURACY AND F-SCORE
    # -------------
    TP = confm[1, 1]
    FN = confm[1, 0]
    FP = confm[0, 1]
    TN = confm[0, 0]
    # Accuracy
    # ====================== YOUR CODE HERE ======================
    accuracy = TP + TN / (TP + TN + FP + FN)
    # ============================================================
    print("***************")
    print("The accuracy of the Logistic Regression classifier is {:.4f}".
          format(accuracy))
    print("***************")

    # F1 score
    # ====================== YOUR CODE HERE ======================
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f_score = (2 * precision * recall) / (precision + recall)
    # ============================================================
    print("")
    print("***************")
    print("The F1-score of the Logistic Regression classifier is {:.4f}".
          format(f_score))
    print("***************")

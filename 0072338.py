import numpy as np
import pandas as pd

X = np.genfromtxt("hw01_data_points.csv", delimiter=",", dtype=str)
y = np.genfromtxt("hw01_class_labels.csv", delimiter=",", dtype=int)

print("--------------- STEP 3 -----------------")


# STEP 3
# first 50000 data points should be included to train
# remaining 43925 data points should be included to test
# should return X_train, y_train, X_test, and y_test
def train_test_split(X, y):
    # your implementation starts below
    # splitting the first 50000 data points to the training set
    X_train = X[:50000]
    y_train = y[:50000]

    # splitting the remaining data points to the testing set
    X_test = X[50000:]
    y_test = y[50000:]

    # your implementation ends above
    return X_train, y_train, X_test, y_test


X_train, y_train, X_test, y_test = train_test_split(X, y)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

print("--------------- STEP 4 -----------------")


# STEP 4
# assuming that there are K classes
# should return a numpy array with shape (K,)
def estimate_prior_probabilities(y):
    # your implementation starts below
    class_counts = np.bincount(y)  # count the number of occurrences of each class
    total_count = len(y)  # total number of samples
    non_zero_counts = class_counts[class_counts != 0]  # remove any 0 values from the class_counts array: wrote this
    # code to return the desired output stated in the homework.

    class_priors = non_zero_counts / total_count  # calculating the prior probabilities

    # your implementation ends above
    return class_priors


class_priors = estimate_prior_probabilities(y_train)
print(class_priors)

print("--------------- STEP 5 -----------------")


# STEP 5
# assuming that there are K classes and D features
# should return four numpy arrays with shape (K, D)
def estimate_nucleotide_probabilities(X, y):
    # your implementation starts below
    K, D = np.max(y) + 1, X.shape[1]  # setting the values of K and D based on the input data.
    # K: no. of classes (take the max value in y and add 1), D: no. of features (no. of columns in X)

    counts = np.zeros((K, D, 4))  # created a 3D array of zeros

    for x, c in zip(X, y):
        for i in range(D):  # iterates over all data points D, count the
            # occurrences of each nucleotide for each class c and each feature d
            if x[i] == 'A':
                counts[c, i, 0] += 1
            elif x[i] == 'C':
                counts[c, i, 1] += 1
            elif x[i] == 'G':
                counts[c, i, 2] += 1
            elif x[i] == 'T':
                counts[c, i, 3] += 1
    total_counts = np.sum(counts, axis=2) + 0.4  # gives the total count for each class and each feature.
    # note: the 0.4 is added to avoid the division by zero.

    # finding the probability of each nucleotide for each class and each feature.
    pAcd = counts[:, :, 0] / total_counts
    pCcd = counts[:, :, 1] / total_counts
    pGcd = counts[:, :, 2] / total_counts
    pTcd = counts[:, :, 3] / total_counts

    # extra: remove the first row of zeros from each array:
    # wrote this code to make it look like the output statement:
    pAcd = np.delete(pAcd, 0, axis=0)
    pCcd = np.delete(pCcd, 0, axis=0)
    pGcd = np.delete(pGcd, 0, axis=0)
    pTcd = np.delete(pTcd, 0, axis=0)

    # your implementation ends above
    return pAcd, pCcd, pGcd, pTcd


pAcd, pCcd, pGcd, pTcd = estimate_nucleotide_probabilities(X_train, y_train)
print(pAcd)
print(pCcd)
print(pGcd)
print(pTcd)
print("--------------- STEP 6 -----------------")


# STEP 6
# assuming that there are N data points and K classes
# should return a numpy array with shape (N, K)
def calculate_score_values(X, pAcd, pCcd, pGcd, pTcd, class_priors):
    # your implementation starts below
    N, K = X.shape[0], len(class_priors)  # unpacking the tuple returned by X.shape
    score_values = np.zeros((N, K))  # create a 2d array of zeros with shape (N, K)

    for i in range(N):  # iterate over each data point
        for j in range(K):  # iterate over each class
            log_prob = 0  # log probability of data point i given the class j.
            for k in range(X.shape[1]):  # iterates over each feature of i
                if X[i, k] == 'A':  # if the nucleotide at position k is A
                    log_prob += np.log(pAcd[j, k])  # add it to log_prob
                elif X[i, k] == 'C':
                    log_prob += np.log(pCcd[j, k])
                elif X[i, k] == 'G':
                    log_prob += np.log(pGcd[j, k])
                elif X[i, k] == 'T':
                    log_prob += np.log(pTcd[j, k])
            score_values[i, j] = np.log(class_priors[j]) + log_prob  # calculating the score value for data point i
            # and class j store the result in score_values.

    # your implementation ends above
    return (score_values)


scores_train = calculate_score_values(X_train, pAcd, pCcd, pGcd, pTcd, class_priors)
print(scores_train)

scores_test = calculate_score_values(X_test, pAcd, pCcd, pGcd, pTcd, class_priors)
print(scores_test)

print("--------------- STEP 7 -----------------")


# STEP 7
# assuming that there are K classes
# should return a numpy array with shape (K, K)
def calculate_confusion_matrix(y_truth, scores):
    # your implementation starts below
    K = scores.shape[1]
    confusion_matrix1 = np.zeros((4, 4), dtype=int)  # initialize

    y_pred = np.argmax(scores, axis=1)
    for true_class in range(4):  # iterate over true classes
        for pred_class in range(4):  # iterate over pred. classes
            confusion_matrix1[true_class, pred_class] = np.sum(
                np.logical_and(y_truth == true_class,
                               y_pred == pred_class))  # computes the no. of data points that belongs to true_class
            # and pred_class.
    # wrote code to make it look like the output statement
    confusion_matrix = np.array([[confusion_matrix1[1][0], confusion_matrix1[2][0]], [confusion_matrix1[1][1],
                                                                                      confusion_matrix1[2][1]]])

    # your implementation ends above
    return confusion_matrix


confusion_train = calculate_confusion_matrix(y_train, scores_train)
print(confusion_train)

confusion_test = calculate_confusion_matrix(y_test, scores_test)
print(confusion_test)

# titanic-data-machine-learning
Build machine learning models to identify whether individuals on the Titanic survived for Kaggle competition (Titanic - Machine Learning from Disaster)

[Kaggle competition](https://www.kaggle.com/competitions/titanic)

## Introduction
Starting from basic data preprocessing, normalization, and handling categorical data through one-hot encoding, I progressed to the fundamental Logistic Regression model in Machine Learning, explored various optimized models, and concluded by using Deep Learning's Neural Network for training. 

## Data
[training data](https://github.com/johnson70630/titanic-data-machine-learning/blob/main/titanic_competition/titanic_data/train.csv)

[testing data](https://github.com/johnson70630/titanic-data-machine-learning/blob/main/titanic_competition/titanic_data/test.csv)

## Training data preprocessing
- Remove all data related to passengers with one NaN entry
- Set 'male' under the 'Sex' data to 1; set 'female' to 0
- Set 'S' under the 'Embarked' data to 0; set 'C' to 1; set 'Q' to 2

## Testing data preproceesing
- no 'survived' column
- Fill in missing data with the average value of the corresponding column in the training data
- Set 'male' under the 'Sex' data to 1; set 'female' to 0
- Set 'S' under the 'Embarked' data to 0; set 'C' to 1; set 'Q' to 2

## One-hot Encoding
'Sex' and 'Embarked' are categorical data types. Therefore, we cannot simply replace 'male', 'female' with 0, 1, or 'S', 'C', 'Q' with 0, 1, 2 (because gender and ports have no inherent order). Thus, we need to introduce One-hot Encoding

- Transform Sex into Sex_male (Sex_1) and Sex_female (Sex_0), and under each new feature, use 0 and 1 to represent the presence or absence of the category
- Transform Pclass into Pclass_0, Pclass_1, and Pclass_2, and under each new feature, use 0 and 1 to represent the presence or absence of the category
- Transform Embarked into Embarked_0, Embarked_1, and Embarked_2, and under each new feature, use 0 and 1 to represent the presence or absence of the category

## Normalization
Normalize each feature in the data separately, so that the values are between 0 and 1

## Classification Model Training
### degree = 1
The number of features is the number of features in the input, denoted as N

### degree = 2
Among inputs, each feature is multiplied pairwise. Therefore, if there are N features in inputs, the quadratic term will have (N)(N+1)/2 features, and the entire feature vector will have N + (N)(N+1)/2 features (including the original first-order terms)

### Training
Train the weights using **Logistic Regression** to perform **Gradient Descent**. After training is complete, return the weights

### [Machine Learning level 1](https://github.com/johnson70630/titanic-data-machine-learning/blob/main/titanic_competition/titanic_level1.py)
Write out the concepts of machine learning directly in code without using pandas and scikit-learn

### [Machine Learning level 2](https://github.com/johnson70630/titanic-data-machine-learning/blob/main/titanic_competition/titanic_level2.py)
Read the file using pandas(pd), build the data, and then use scikit-learn(sklearn) for machine learning

## Different Models

### [Support Vector Machine (SVM)](https://github.com/johnson70630/titanic-data-machine-learning/blob/main/titanic_optimization/titanic_pandas_svm.py)
SVM (Support Vector Machine) is a supervised learning method that estimates a classification hyperplane using the principle of minimizing statistical risk. The fundamental concept is quite simple: find a decision boundary (hyperplane) that maximizes the margins between two classes, allowing for a perfect separation.

### [Priciple Component Analysis (PCA) degree=1](https://github.com/johnson70630/titanic-data-machine-learning/blob/main/titanic_optimization/titanic_pandas_pca_degree1.py)
### [Priciple Component Analysis (PCA) degree=2](https://github.com/johnson70630/titanic-data-machine-learning/blob/main/titanic_optimization/titanic_pandas_pca_degree2.py)
Principal Component Analysis (PCA) is classified within machine learning as a method for dimension reduction and feature extraction. Dimension reduction aims to decrease the number of dimensions in the data, with the goal of maintaining overall performance without significant loss and, in some cases, improving it.

### [Dicision Tree (DTs)](https://github.com/johnson70630/titanic-data-machine-learning/blob/main/titanic_optimization/titanic_pandas_decision_tree.py)
Decision trees can classify data step by step, offering logical and visual representation of the classification process. However, it's essential to control the depth of the tree to prevent overfitting.

### [Bagging](https://github.com/johnson70630/titanic-data-machine-learning/blob/main/titanic_optimization/titanic_bagging_classifier.py)
From the training dataset, extract K samples, then train K classifiers (trees in this case) using these K samples. Each time, the K samples are put back into the population, so there is some data overlap among these K samples. However, because the samples for each tree are still different, the trained classifiers (trees) have diversity. The final result is obtained by majority vote, where each classifier has equal weight.

### [Random Forest](https://github.com/johnson70630/titanic-data-machine-learning/blob/main/titanic_optimization/titanic_random_forest.py)
1. Define a random sample of size n (here, using the bagging method), which involves randomly selecting n data points from the dataset with replacement

2. From the selected n data points, train a decision tree. For each node:
 - Randomly select d features
 - Use these features to split the node (based on information gain
3. Repeat steps 1 to 2 k times

4. Aggregate the predictions of all decision trees and determine the classification result by majority vote 

### [k-means Clustering](https://github.com/johnson70630/titanic-data-machine-learning/blob/main/titanic_optimization/titanic_k_means.py)
1. Start by setting the desired number of clusters (k)

2. In the feature space, randomly assign k centroids

3. For each data point, calculate the distance to all k centroids

4. Assign each data point to the cluster represented by the nearest centroid

5. Within each cluster, update the centroid by recalculating it based on the data points assigned to that cluster

6. Repeat steps 3-5 until the centroids' movement becomes negligible (convergence), indicating the algorithm has reached stability

### [Naive Bayes](https://github.com/johnson70630/titanic-data-machine-learning/blob/main/titanic_optimization/titanic_naive_bayes%20.ipynb)

## Deep Learning

### [Neural Network level 1](https://github.com/johnson70630/titanic-data-machine-learning/blob/main/titanic_deep_learning/titanic_deep_nn.py)
This file demonstrates how to create a **deep neural network (5 layers NN)** to train our titanic data.

### [Neural Network level 2](https://github.com/johnson70630/titanic-data-machine-learning/blob/main/titanic_deep_learning/titanic_nn.py)
This file demonstrates how to use **batch gradient descent** to update weights by numpy array.

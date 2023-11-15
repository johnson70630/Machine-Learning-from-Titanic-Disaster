"""
File: titanic_k_means.py
Name: Johnson
---------------------------
This file shows how to use pandas and sklearn svm
packages to build a simple classification model
"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn import cluster
from sklearn import decomposition

# Constants - filenames for data set
TRAIN_FILE = 'titanic_data/train.csv'
TEST_FILE = 'titanic_data/test.csv'

# Global Variable - cache for nan data in training data
nan_cache = {}


def main():
    # Data cleaning
    data = data_preprocess(TRAIN_FILE, mode='Train')

    # Extract features ('Pclass', 'Age', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked')
    feature_names = ['Pclass', 'Age', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked']
    x_train = data[feature_names]

    # Construct K Means
    k_means = cluster.KMeans(n_clusters=2)
    k_means.fit(x_train)
    centroids = k_means.cluster_centers_

    # PCA
    pca = decomposition.PCA(n_components=3)
    x_train_3d = pca.fit_transform(x_train)
    x = list(x_train_3d[i][0] for i in range(len(x_train_3d)))
    y = list(x_train_3d[i][1] for i in range(len(x_train_3d)))
    z = list(x_train_3d[i][2] for i in range(len(x_train_3d)))
    ax = plt.axes(projection='3d')
    ax.scatter3D(x, y, z, color='gray')

    centroids_3d = pca.transform(centroids)
    cx = list(centroids_3d[i][0] for i in range(len(centroids_3d)))
    cy = list(centroids_3d[i][1] for i in range(len(centroids_3d)))
    cz = list(centroids_3d[i][2] for i in range(len(centroids_3d)))
    ax.scatter3D(cx, cy, cz, color='red', s=300)
    plt.show()


def data_preprocess(filename, mode='Train'):
    """
    : param filename: str, the csv file to be read into by pd
    : param mode: str, the indicator of training mode or testing mode
    -----------------------------------------------
    This function reads in data by pd, changing string data to float,
    and finally tackling missing data showing as NaN on pandas
    """
    # Read in data as a column based DataFrame
    data = pd.read_csv(filename)
    if mode == 'Train':
        # Cleaning the missing data in Fare column by replacing NaN with its median
        fare_median = data['Fare'].dropna().median()
        data['Fare'] = data['Fare'].fillna(fare_median)

        # Cleaning the missing data in Age column by replacing NaN with its median
        age_median = data['Age'].dropna().median()
        data['Age'] = data['Age'].fillna(age_median)

        # Cache some data for test set
        nan_cache['Age'] = age_median
        nan_cache['Fare'] = fare_median

    else:
        # Fill in the NaN cells by the values in nan_cache to make it consistent
        data['Fare'] = data['Fare'].fillna(nan_cache['Fare'])
        data['Age'] = data['Age'].fillna(nan_cache['Age'])

    # Changing 'male' to 1, 'female' to 0
    data.loc[data.Sex == 'male', 'Sex'] = 1
    data.loc[data.Sex == 'female', 'Sex'] = 0

    # Changing 'S' to 0, 'C' to 1, 'Q' to 2
    data['Embarked'] = data['Embarked'].fillna('S')
    data.loc[data.Embarked == 'S', 'Embarked'] = 0
    data.loc[data.Embarked == 'C', 'Embarked'] = 1
    data.loc[data.Embarked == 'Q', 'Embarked'] = 2

    return data


if __name__ == '__main__':
    main()

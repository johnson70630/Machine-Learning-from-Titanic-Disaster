"""
File: titanic_pandas_decision_tree.py
Name: Johnson
---------------------------
This file shows how to use pandas and sklearn
packages to build a decision tree, which enables
students to see the most important features 
on Webgraphviz.com
"""

import pandas as pd
from sklearn import tree

# Constants - filenames for data set
TRAIN_FILE = 'titanic_data/train.csv'
TEST_FILE = 'titanic_data/test.csv'

# Global Variable - cache for nan data in training data
nan_cache = {}


def main():
    # Data cleaning
    data = data_preprocess(TRAIN_FILE, mode='Train')

    # Extract true labels
    y = data.Survived


    # Extract features ('Pclass', 'Age', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked')
    feature_names = ['Pclass', 'Age', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked']
    x_train = data[feature_names]

    # Construct Tree
    d_tree = tree.DecisionTreeClassifier(max_depth=5, min_samples_leaf=6)
    d_tree_classifier = d_tree.fit(x_train, y)
    acc = d_tree_classifier.score(x_train, y)
    print('Training Acc: ', acc)
    tree.export_graphviz(d_tree_classifier, out_file='tree', feature_names=feature_names)

    test_data = data_preprocess(TEST_FILE, mode='Test')
    x_test = test_data[feature_names]
    predictions = d_tree_classifier.predict(x_test)
    out_file(predictions, 'd_tree.csv')


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


def out_file(predictions, filename):
    """
    : param predictions: numpy.array, a list-like data structure that stores 0's and 1's
    : param filename: str, the filename you would like to write the results to
    """
    print('\n===============================================')
    print(f'Writing predictions to --> {filename}')
    with open(filename, 'w') as out:
        out.write('PassengerId,Survived\n')
        start_id = 892
        for ans in predictions:
            out.write(str(start_id) + ',' + str(ans) + '\n')
            start_id += 1
    print('===============================================')


if __name__ == '__main__':
    main()

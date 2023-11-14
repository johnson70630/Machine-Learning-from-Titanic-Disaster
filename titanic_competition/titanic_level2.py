"""
File: titanic_level2.py
Name: Johnson
----------------------------------
This file builds a machine learning algorithm by pandas and sklearn libraries.
We'll be using pandas to read in dataset, store data into a DataFrame,
standardize the data by sklearn, and finally train the model and
test it on kaggle website. Hyper-parameters tuning are not required due to its
high level of abstraction, which makes it easier to use but less flexible.
You should find a good model that surpasses 77% test accuracy on kaggle.
"""

import math
import pandas as pd
from sklearn import preprocessing, linear_model

TRAIN_FILE = 'titanic_data/train.csv'
TEST_FILE = 'titanic_data/test.csv'


def data_preprocess(filename, mode='Train', training_data=None):
	"""
	:param filename: str, the filename to be read into pandas
	:param mode: str, indicating the mode we are using (either Train or Test)
	:param training_data: DataFrame, a 2D data structure that looks like an excel worksheet
						  (You will only use this when mode == 'Test')
	:return: Tuple(data, labels), if the mode is 'Train'; or return data, if the mode is 'Test'
	"""
	data = pd.read_csv(filename)
	data.pop('PassengerId')
	data.pop('Name')
	data.pop('Ticket')
	data.pop('Cabin')
	data['Sex'].replace(['male', 'female'], [1, 0], inplace=True)
	data['Embarked'].replace(['S', 'C', 'Q'], [0, 1, 2], inplace=True)
	if mode == 'Train':
		data = data.dropna()
		labels = data.pop('Survived')
		return data, labels
	elif mode == 'Test':
		data['Age'].fillna(training_data.Age.mean(), inplace=True)
		data['Fare'].fillna(round(training_data.Fare.mean(), 3), inplace=True)
		return data


def one_hot_encoding(data, feature):
	"""
	:param data: DataFrame, key is the column name, value is its data
	:param feature: str, the column name of interest
	:return data: DataFrame, remove the feature column and add its one-hot encoding features
	"""
	if feature == 'Pclass':
		data.Pclass.replace([1, 2, 3], [0, 1, 2], inplace=True)
	data = pd.get_dummies(data, columns=[feature])
	return data
	# if feature == 'Sex':
	# 	data['Sex_0'] = 0
	# 	data.loc[data.Sex == 0, 'Sex_0'] = 1
	# 	data['Sex_1'] = 0
	# 	data.loc[data.Sex == 1, 'Sex_1'] = 1
	# 	data.pop('Sex')
	# elif feature == 'Pclass':
	# 	data['Pclass_0'] = 0
	# 	data.loc[data.Pclass == 1, 'Pclass_0'] = 1
	# 	data['Pclass_1'] = 0
	# 	data.loc[data.Pclass == 2, 'Pclass_1'] = 1
	# 	data['Pclass_2'] = 0
	# 	data.loc[data.Pclass == 3, 'Pclass_2'] = 1
	# 	data.pop('Pclass')
	# elif feature == 'Embarked':
	# 	data['Embarked_0'] = 0
	# 	data.loc[data.Embarked == 0, 'Embarked_0'] = 1
	# 	data['Embarked_1'] = 0
	# 	data.loc[data.Embarked == 1, 'Embarked_1'] = 1
	# 	data['Embarked_2'] = 0
	# 	data.loc[data.Embarked == 2, 'Embarked_2'] = 1
	# 	data.pop('Embarked')
	# return data


def standardization(data, mode='Train'):
	"""
	:param data: DataFrame, key is the column name, value is its data
	:param mode: str, indicating the mode we are using (either Train or Test)
	:return data: DataFrame, standardized features
	"""
	standardizer = preprocessing.StandardScaler()
	if mode == 'Train':
		data = standardizer.fit_transform(data)
	else:
		data = standardizer.transform(data)
	return data


def main():
	"""
	You should call data_preprocess(), one_hot_encoding(), and
	standardization() on your training data. You should see ~80% accuracy on degree1;
	~83% on degree2; ~87% on degree3.
	Please write down the accuracy for degree1, 2, and 3 respectively below
	(rounding accuracies to 8 decimal places)
	TODO: real accuracy on degree1 -> 0.80196629
	TODO: real accuracy on degree2 -> 0.83707865
	TODO: real accuracy on degree3 -> 0.87640449
	"""
	# data preprocessing
	X_train, Y = data_preprocess(TRAIN_FILE, mode='Train')
	X_train = one_hot_encoding(X_train, 'Sex')
	X_train = one_hot_encoding(X_train, 'Pclass')
	X_train = one_hot_encoding(X_train, 'Embarked')
	standardizer = preprocessing.StandardScaler()
	X_train = standardizer.fit_transform(X_train)

	# training
	h = linear_model.LogisticRegression(max_iter=1000)
	poly_phi = preprocessing.PolynomialFeatures(degree=3)
	X_train_poly = poly_phi.fit_transform(X_train)
	classifier_poly = h.fit(X_train_poly, Y)
	acc = classifier_poly.score(X_train_poly, Y)
	print('Training Acc: ', acc)


if __name__ == '__main__':
	main()

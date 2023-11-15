"""
File: titanic_pandas_pca_degree1.py
Name: Johnson
---------------------------
This file shows how to use pandas and sklearn
packages to build a machine learning project
from scratch by their high order abstraction.
The steps of this project are:
1) Data pre-processing by pandas
2) PCA to extract components
3) Learning by sklearn
4) Test on D_test
"""

import pandas as pd
from sklearn import linear_model, preprocessing, decomposition


# Constants - filenames for data set
TRAIN_FILE = 'titanic_data/train.csv'                    # Training set filename
TEST_FILE = 'titanic_data/test.csv'                      # Test set filename

# Global variable
nan_cache = {}                                           # Cache for test set missing data


def main():

	# Data cleaning
	data = data_preprocess(TRAIN_FILE, mode='Train')

	# Extract true labels
	y = data.Survived

	# Extract features ('Pclass', 'Age', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked')
	x_train = data[['Pclass', 'Age', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked']]

	# Standardization
	standardizer = preprocessing.StandardScaler()
	x_train = standardizer.fit_transform(x_train)

	pca = decomposition.PCA(n_components=5)
	x_train = pca.fit_transform(x_train)
	print('Var retained: ', sum(pca.explained_variance_ratio_))

	# Training
	h = linear_model.LogisticRegression(max_iter=10000)
	classifier = h.fit(x_train, y)
	acc = classifier.score(x_train, y)
	print('Degree 1 Training Acc:', acc)

	# Test
	x_test = data_preprocess(TEST_FILE, mode='Test')
	# Extract features ('Pclass', 'Age', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked')
	x_test = x_test[['Pclass', 'Age', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked']]
	# Standardization
	x_test = standardizer.transform(x_test)

	x_test = pca.transform(x_test)

	# Predict
	predictions = classifier.predict(x_test)
	out_file(predictions, 'pandas_submission_degree1.csv')


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
			out.write(str(start_id)+','+str(ans)+'\n')
			start_id += 1
	print('===============================================')


if __name__ == '__main__':
	main()

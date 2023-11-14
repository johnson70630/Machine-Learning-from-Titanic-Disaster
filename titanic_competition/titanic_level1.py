"""
File: titanic_level1.py
Name: Johnson
----------------------------------
This file builds a machine learning algorithm from scratch 
by Python. We'll be using 'with open' to read in dataset,
store data into a Python dict, and finally train the model and 
test it on kaggle website. This model is the most flexible among all
levels. You should do hyper-parameter tuning to find the best model.
"""

import math
from util import *
TRAIN_FILE = 'titanic_data/train.csv'
TEST_FILE = 'titanic_data/test.csv'


def data_preprocess(filename: str, data: dict, mode='Train', training_data=None):
	"""
	:param filename: str, the filename to be processed
	:param data: an empty Python dictionary
	:param mode: str, indicating if it is training mode or testing mode
	:param training_data: dict[str: list], key is the column name, value is its data
						  (You will only use this when mode == 'Test')
	:return data: dict[str: list], key is the column name, value is its data
	"""
	with open(filename, 'r') as f:
		if mode == 'Train':
			start = 2
		else:
			start = 1
		first_line = True
		for line in f:
			tokens = line.strip().split(',')
			if first_line:
				for i in range(len(tokens)):
					if i != 0 and i != start+1 and i != start+6 and i != start+8:
						data[tokens[i]] = []
				first_line = False
			else:
				if mode == 'Test':
					if not tokens[start+4]:
						tokens[start+4] = sum(training_data['Age'])/len(training_data['Age'])
					if not tokens[start+8]:
						tokens[start+8] = round(sum(training_data['Fare'])/len(training_data['Fare']), 3)
				if '' not in tokens[:-3] and tokens[-1] != '':
					for i in range(len(tokens)):
						if i == 1 and mode == 'Train':
							data['Survived'].append(int(tokens[i]))
						elif i == start:
							data['Pclass'].append(int(tokens[i]))
						elif i == start+3:
							if tokens[i] == 'male':
								data['Sex'].append(1)
							else:
								data['Sex'].append(0)
						elif i == start+4:
							data['Age'].append(float(tokens[i]))
						elif i == start+5:
							data['SibSp'].append(int(tokens[i]))
						elif i == start+6:
							data['Parch'].append(int(tokens[i]))
						elif i == start+8:
							data['Fare'].append(float(tokens[i]))
						elif i == start+10:
							if tokens[i] == 'S':
								data['Embarked'].append(0)
							elif tokens[i] == 'C':
								data['Embarked'].append(1)
							else:
								data['Embarked'].append(2)
	return data


def one_hot_encoding(data: dict, feature: str):
	"""
	:param data: dict[str, list], key is the column name, value is its data
	:param feature: str, the column name of interest
	:return data: dict[str, list], remove the feature column and add its one-hot encoding features
	"""
	feature_num = {'Pclass': [1, 2, 3], 'Sex': [0, 1], 'Embarked': [0, 1, 2]}
	for val in data[feature]:
		for i in range(len(feature_num[feature])):
			new_key = f'{feature}_{str(i)}'
			if new_key not in data:
				data[new_key] = []
			if val == feature_num[feature][i]:
				data[new_key].append(1)
			else:
				data[new_key].append(0)
	data.pop(feature)
	return data
	# if feature == 'Sex':
	# 	data['Sex_0'] = []
	# 	data['Sex_1'] = []
	# 	for sex in data['Sex']:
	# 		if sex == 0:
	# 			data['Sex_0'].append(1)
	# 			data['Sex_1'].append(0)
	# 		else:
	# 			data['Sex_0'].append(0)
	# 			data['Sex_1'].append(1)
	# 	data.pop('Sex')
	# elif feature == 'Pclass':
	# 	data['Pclass_0'] = []
	# 	data['Pclass_1'] = []
	# 	data['Pclass_2'] = []
	# 	for plcass in data['Pclass']:
	# 		if plcass == 1:
	# 			data['Pclass_0'].append(1)
	# 			data['Pclass_1'].append(0)
	# 			data['Pclass_2'].append(0)
	# 		elif plcass == 2:
	# 			data['Pclass_0'].append(0)
	# 			data['Pclass_1'].append(1)
	# 			data['Pclass_2'].append(0)
	# 		else:
	# 			data['Pclass_0'].append(0)
	# 			data['Pclass_1'].append(0)
	# 			data['Pclass_2'].append(1)
	# 	data.pop('Pclass')
	# elif feature == 'Embarked':
	# 	data['Embarked_0'] = []
	# 	data['Embarked_1'] = []
	# 	data['Embarked_2'] = []
	# 	for embark in data['Embarked']:
	# 		if embark == 0:
	# 			data['Embarked_0'].append(1)
	# 			data['Embarked_1'].append(0)
	# 			data['Embarked_2'].append(0)
	# 		elif embark == 1:
	# 			data['Embarked_0'].append(0)
	# 			data['Embarked_1'].append(1)
	# 			data['Embarked_2'].append(0)
	# 		elif embark == 2:
	# 			data['Embarked_0'].append(0)
	# 			data['Embarked_1'].append(0)
	# 			data['Embarked_2'].append(1)
	# 	data.pop('Embarked')
	# return data


def normalize(data: dict):
	"""
	:param data: dict[str, list], key is the column name, value is its data
	:return data: dict[str, list], key is the column name, value is its normalized data
	"""
	for key, lst in data.items():
		max_num = max(lst)
		min_num = min(lst)
		for i in range(len(lst)):
			new_num = (lst[i]-min_num) / (max_num-min_num)
			lst[i] = new_num
	return data


def learnPredictor(inputs: dict, labels: list, degree: int, num_epochs: int, alpha: float):
	"""
	:param inputs: dict[str, list], key is the column name, value is its data
	:param labels: list[int], indicating the true label for each data
	:param degree: int, degree of polynomial features
	:param num_epochs: int, the number of epochs for training
	:param alpha: float, known as step size or learning rate
	:return weights: dict[str, float], feature name and its weight
	"""
	# Step 1 : Initialize weights
	weights = {}  # feature => weight
	keys = list(inputs.keys())
	if degree == 1:
		for i in range(len(keys)):
			weights[keys[i]] = 0
	elif degree == 2:
		for i in range(len(keys)):
			weights[keys[i]] = 0
		for i in range(len(keys)):
			for j in range(i, len(keys)):
				weights[keys[i] + keys[j]] = 0
	print(weights)
	# Step 2 : Start training
	for epoch in range(num_epochs):
		# Step 3 : Feature Extract
		for i in range(len(inputs['Age'])):
			y = labels[i]
			d = {}
			for j in range(len(keys)):
				d[keys[j]] = inputs[keys[j]][i]
				if degree == 2:
					for k in range(j, len(keys)):
						d[keys[j]+keys[k]] = inputs[keys[j]][i]*inputs[keys[k]][i]
			# Step 4 : Update weights
			k = dotProduct(d, weights)
			h = 1 / (1 + math.exp(-k))
			increment(weights, -alpha*(h-y), d)
	return weights

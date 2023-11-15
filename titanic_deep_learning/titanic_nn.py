"""
File: titanic_nn.py
Name: Johnson
-----------------------------
This file demonstrates how to use batch
gradient descent to update weights by numpy 
array. The training process should be way
faster than stochastic gradient descent
(You should see a smooth training curve as well)
-----------------------------
W1.shape = (6, 3)
W2.shape = (3, 1)
X.shape = (6, m)
Y.shape = (1, m)
-----------------------------
If you correctly implement this NN, you should see the following acc:
0.8151260504201681
"""


import numpy as np


TRAIN = 'titanic_data/train.csv'     # This is the filename of interest
NUM_EPOCHS = 10000                   # This constant controls the total number of epochs
ALPHA = 0.02                         # This constant controls the learning rate α
N1 = 3                               # This constant controls the number of neurons in Layer 1
N2 = 1                               # This constant controls the number of neurons in Layer 2


def main():
	X_train, Y = data_preprocessing()
	_, m = X_train.shape
	print('Y.shape', Y.shape)
	print('X.shape', X_train.shape)
	# ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
	X = normalize(X_train)
	# classifier = h.fit(X, Y)
	W1, W2, B1, B2 = two_layer_nn(X, Y)

	# The last forward prop - inference
	K1 = W1.T.dot(X) + B1
	A1 = np.maximum(0, K1)
	scores = W2.T.dot(A1) + B2
	predictions = np.where(scores > 0, 1, 0)
	acc = np.equal(predictions, Y)
	print('Training Acc: ', np.sum(acc)/m)


def normalize(X):
	"""
	:param X: numpy_array, the dimension is (n, m)
	:return: numpy_array, the values are normalized, where the dimension is still (n, m)
	"""
	min_array = np.min(X, axis=1, keepdims=True)
	max_array = np.max(X, axis=1, keepdims=True)
	return (X-min_array)/(max_array-min_array)


def two_layer_nn(X, Y):
	"""
	:param X: numpy_array, the array holding all the training data
	:param Y: numpy_array, the array holding all the ture labels in X
	:return W1, W2, B1, B2: numpy_array, the array holding all the parameters 
	"""
	n, m = X.shape
	np.random.seed(1)

	# Initialize W1
	W1 = np.random.rand(n, N1)-0.5
	# Initialize W2
	W2 = np.random.rand(N1, N2)-0.5
	# Initialize B1
	B1 = np.random.rand(N1, 1)-0.5
	# Initialize B2
	B2 = np.random.rand(N2, 1)-0.5

	print_every = 500
	for epoch in range(NUM_EPOCHS):
		# Forward Pass
		K1 = W1.T.dot(X) + B1
		A1 = np.maximum(0, K1)
		K2 = W2.T.dot(A1) + B2
		H = 1/(1+np.exp(-K2))
		J = (1/m)*np.sum(-(Y*np.log(H)+(1-Y)*np.log(1-H)))
		if epoch % print_every == 0:
			print('Cost: ', J)

		# Backward Pass
		dK2 = (1/m)*np.sum(H-Y, axis=0, keepdims=True)
		dW2 = A1.dot(dK2.T)
		dB2 = np.sum(dK2, axis=1, keepdims=True)
		dA1 = W2.dot(dK2)
		dK1 = dA1*np.where(K1 <= 0, 0, 1)
		dW1 = X.dot(dK1.T)
		dB1 = np.sum(dK1, axis=1, keepdims=True)

		# Updates all the weights and biases
		W1 = W1 - ALPHA * dW1
		W2 = W2 - ALPHA * dW2
		B1 = B1 - ALPHA * dB1
		B2 = B2 - ALPHA * dB2
	return W1, W2, B1, B2


def data_preprocessing(mode='train'):
	"""
	:param mode: str, indicating if it's training mode or testing mode
	:return: Tuple(numpy_array, numpy_array), the first one is X, the other one is Y
	"""
	data_lst = []
	label_lst = []
	first_data = True
	if mode == 'train':
		with open(TRAIN, 'r') as f:
			for line in f:
				data = line.split(',')
				# ['0PassengerId', '1Survived', '2Pclass', '3Last Name', '4First Name', '5Sex', '6Age', '7SibSp', '8Parch', '9Ticket', '10Fare', '11Cabin', '12Embarked']
				if first_data:
					first_data = False
					continue
				if not data[6]:
					continue
				label = [int(data[1])]
				if data[5] == 'male':
					sex = 1
				else:
					sex = 0
				# ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
				passenger_lst = [int(data[2]), sex, float(data[6]), int(data[7]), int(data[8]), float(data[10])]
				data_lst.append(passenger_lst)
				label_lst.append(label)
	else:
		pass
	return np.array(data_lst).T, np.array(label_lst).T


if __name__ == '__main__':
	main()

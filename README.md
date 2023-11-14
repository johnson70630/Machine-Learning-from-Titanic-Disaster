# titanic-data-machine-learning
Build a machine learning model to identify whether individuals on the Titanic survived for Kaggle competition (Titanic - Machine Learning from Disaster)

[Kaggle competition](https://www.kaggle.com/competitions/titanic)


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
- Normalize each feature in the data separately, so that the values are between 0 and 1

## Classification Model Training
### degree = 1
The number of features is the number of features in the input, denoted as N

### degree = 2
Among inputs, each feature is multiplied pairwise. Therefore, if there are N features in inputs, the quadratic term will have (N)(N+1)/2 features, and the entire feature vector will have N + (N)(N+1)/2 features (including the original first-order terms)

### Training
Train the weights using Gradient Descent! After training is complete, please return the weights


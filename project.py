# --- NOTE: My earlier submissions were incorrectly formatted when I pasted it into CoderByte from my Notepad. This submission is the same code, but formatted correctly :)

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# Load iris data set into an object called data, containing data sets and attributes 
data = load_iris()


# Iris data set - 50 samples of 3 different iris species - 150 samples total
# Each sample: sepal length, sepal width, petal length, pedal width
# Labeled data: Supervised learning. 


# retrieve the data
features = data["data"]



# target - Integers representing the species of each observation

# 0 - Setosa, 1 - Vesicolor, 2 - viginica

labels = data["target"]


# Training Data - Fed into the model
# Test data - Used to evaluate performance and progress of algorithm's training

# 1 - Modify the training script so that only 80% of the data is used for training, and the remaining 20% is test data 
features_train,features_test,labels_train,labels_test= train_test_split(features, labels, test_size=0.2, random_state=45)

# Use training data on model
model = DecisionTreeClassifier()
model.fit(features_train, labels_train)

# Predict test data
labels_pred=model.predict(features_test)

# 2 - Output the accuracy score of the model on the test data
print(f"Accuracy score of the default model on the test data: {accuracy_score(labels_test, labels_pred)}")

# 3 - Implement a simple cross-validation step to find which of 1, 5, and 10 is the best max_depth for the classifier

# Max depth of 1
modelA = DecisionTreeClassifier(max_depth=1)
modelA.fit(features_train, labels_train)
labels_pred_A=modelA.predict(features_test)
print(f"Accuracy score of the model with max_depth 1 on the test data: {accuracy_score(labels_test, labels_pred_A)}")

# Max depth of 5
modelB = DecisionTreeClassifier(max_depth=5)
modelB.fit(features_train, labels_train)
labels_pred_B=modelB.predict(features_test)
print(f"Accuracy score of the model with max_depth 5 on the test data: {accuracy_score(labels_test, labels_pred_B)}")

# Max depth of 10
modelC = DecisionTreeClassifier(max_depth=10)
modelC.fit(features_train, labels_train)
labels_pred_C=modelC.predict(features_test)
print(f"Accuracy score of the model with max_depth 10 on the test data: {accuracy_score(labels_test, labels_pred_C)}")

# -- Based on the results, max depth of 10 and 5 are tied for best performing --

# 4 - Print the confusion matrix of the classifier that results from (3).
print(f"confusion matrix: {(confusion_matrix(labels_test, labels_pred_C))}")

# The most false positives are found in class C (viginica) where one is incorrectly identified as class B (Vesicolor)

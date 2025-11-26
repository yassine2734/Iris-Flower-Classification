# Import Packages
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

columns = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width', 'Class_labels']
df = pd.read_csv('iris.data', names = columns)

df.head()

df.describe()


# Visualizing The hole of the dataset
fig = sns.pairplot(df, hue='Class_labels')
# fig.savefig('pairplot.png')
'''
-- Iris-Setosa is the shortest, and it is well separated from Virginica and Versicolor
-- Iris-Virginica is the longest
'''

# Separating features and target
data = df.values
X = data[:, 0:4]
Y = data[:, 4]


# Calculating average of each features for all classes
feature_names = columns[:4]
classes = np.unique(Y) # ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
means_list = [X[Y == c].mean(axis=0) for c in classes] # Calculate means for each class
means = pd.DataFrame(means_list, index=classes, columns=feature_names) 
means_T = means.T
means_T.plot(kind='bar', figsize=(10, 6))
plt.title('Average of each feature for different classes')
plt.ylabel('Average value')
plt.xlabel('Class Labels')
plt.legend(title='Features')
# plt.show()
''' Here we can clearly confirm that Virginica is the longest and Setosa is the shortest '''

# Model training 
''' With train_test_split we will split the data into : - Training dataset
                                                        - Testing dataset
'''
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2) # I kept 20% of the dataset to the test and evaluation
svm = SVC()
svm.fit(X_train, Y_train)


# Model evaluation 
predictions = svm.predict(X_test) # Prediction from the test dataset

# Calculating the accuracy score of the predicted classes 
accuracy_score = accuracy_score(Y_test, predictions) 


# classification report 
print(classification_report(Y_test, predictions))



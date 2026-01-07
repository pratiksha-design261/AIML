#Import libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#loading dataset and display the first few rows
iris_data = pd.read_csv('Iris.csv')
iris_data.head()

#split data into feature (X) and label (Y)
X = iris_data.drop(columns=['id','Species'])
Y = iris_data['Species']

# split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# standardize the feature 
scaler = StandardScaler()
X_train_scale = scaler.fit_transform(X_train)
X_test_scale = scaler.transform(X_test))

# create a ml model
model = LogisticRegression()

# Train the model
model.fit(X_train_scale, Y_train)

# Evaluate the model on the testing set
Y_pred = model.predict(X_test_scale)
accuracy = accuracy_score(Y_test, Y_pred)
print("Accuracy:",accuracy)

# sample new data prediction
new_data = np.array([[5.1, 3.5, 1.4, 0.2],
                     [6.3, 2.9, 5.6, 1.8],
                     [4.9, 3.0, 1.4, 0.2]])

# standardize the new_data
new_data_scaled = scaler.transform(new_data)

# Prediction using train model 
prediction = model.predict(new_data_scaled)

#print prediction
print("Prdiction:",prediction)

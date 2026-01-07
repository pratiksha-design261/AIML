#Import libraries
import pandas as pd
from sklearn.leaner_model import LogesticRegression

#loading dataset and display the first few rows
iris_data = pd.read_csv('Iris.csv')
iris_data.head()

#split data into feature (X) and label (Y)
X = iris_data.drop(columns=['id','Species'])
Y = iris_data['Species']
X.head()

# create a ml model
model = LogisticRegression()

# Train the model
model.fit(X.values, Y)

# Prediction using train model 
prediction = model.predict([[4.6, 3.8, 1.5, 0.2]])

#print prediction
print(prediction)

# Implementation of Simple Linear Regression Model for Predicting the Marks Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Assign the points for representing in the graph.
5. Predict the regression for marks by using the representation of the graph.
6. Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:

Program to implement the simple linear regression model for predicting the marks scored.
Developed by : KEERTHIVASAN S 
RegisterNumber : 212223220046

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:

## Dataset
![dataset](https://github.com/user-attachments/assets/d45788b9-9596-407e-9bb9-1c6383b80c36)

## Head Values
![headvalues](https://github.com/user-attachments/assets/7a0d4c14-94f0-49f7-a1ba-12dbfa51c6f2)

## Tail Values
![tailvalues](https://github.com/user-attachments/assets/a7d066c0-9b65-45a6-b31a-6b087f9777e5)

## X and Y Values
![xandyvalues](https://github.com/user-attachments/assets/7467c907-28cd-41ac-97c7-56a0f4e5c0e8)

## Prediction Values of X and Y
![predictionvaluesofxandy](https://github.com/user-attachments/assets/db2fc4d4-820f-4955-b827-16251d41d6f7)

## MSE,MAE and RMSE
![mse,mae,rmse](https://github.com/user-attachments/assets/f4009ff7-a190-49b8-ba97-59b709581517)

## Training Set
![trainingset](https://github.com/user-attachments/assets/42343de0-36fb-40c0-8998-eef14f463399)

## Testing Set
![testingset](https://github.com/user-attachments/assets/b358cd17-df25-42a4-b7ad-1bc588066e54)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.

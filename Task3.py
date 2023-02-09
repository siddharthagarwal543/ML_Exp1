import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd

csv=pd.read_csv("Salary_Data.csv")
# Generating datasets from csv file
x=csv.iloc[:,:-1].values
y=csv.iloc[:,:-1].values

#Splitting dataset
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

#generating model
regressor=LinearRegression().fit(x_train,y_train)
#print(regressor.predict(x_test))
#Plotiing Train data
plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,regressor.predict(x_train),color="blue")
plt.title("Salary vs Experience(test set)")
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#Plotting test data
plt.scatter(x_test,y_test,color='red')
plt.plot(x_test,regressor.predict(x_test),color="blue")
plt.title("Salary vs Experience(train set)")
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#Finding parameters of model
print("intercept",regressor.intercept_)
print("coefficent of regression",regressor.score(x,y))
print("slope",regressor.coef_)

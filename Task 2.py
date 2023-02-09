import numpy as np
from sklearn.linear_model import LinearRegression
# Generating data values
x=[[0,1],[5,1],[15,2],[25,5],[35,11],[45,15],[55,34],[60,35]]
y=[4,5,20,14,32,22,38,43]
x,y=np.array(x),np.array(y)
#Generating model
model=LinearRegression().fit(x,y)
#Finding parameters of model
r=model.score(x,y)
print("coefficent of corelation",r)
print("intercept",model.intercept_)
print("score",model.score(x,y))
print("slope",model.coef_)
print("predict",model.predict(x))
print(model.intercept_+model.coef_*x)
print("y new",model.predict(np.arange(10).reshape(-1,2)))
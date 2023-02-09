import numpy as np
from sklearn.linear_model import LinearRegression
# declaring arrays with data values
a=np.array([5,15,25,35,45,55]).reshape(-1,1)
b=np.array([5,20,14,32,22,38])
# generating model
model=LinearRegression().fit(a,b)
# corelation coefficent
r=model.score(a,b)
print(r)
#other parameters of model
print("intercept",model.intercept_)
print("score",model.score(a,b))
print("coef",model.coef_)
print("predict",model.predict(a))
#generating model in another way
new_model=LinearRegression().fit(a,b.reshape(-1,1))
#printing parameters of model
print("new intercept",new_model.intercept_)
print("new score",new_model.score(a,b))
print("new coef",new_model.coef_)
print("new predict",new_model.predict(a))
print(model.intercept_+model.coef_*a)
print("y new",model.predict(np.arange(5).reshape(-1,1)))
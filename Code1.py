import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
csv=pd.read_csv("./prices.csv")
stock="GOOG"
df=pd.DataFrame(csv[csv.symbol==stock])
#print(df)
x=np.array(df['close']).reshape(-1,1)
y=np.array(df['volume']).reshape(-1,1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
regressor=LinearRegression().fit(x_train,y_train)
#print(regressor.predict(x_test))
plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,regressor.predict(x_train),color="blue")
plt.title("Closing Price vs Volume")
plt.xlabel('Closing Price')
plt.ylabel('Volume')
plt.show()
plt.scatter(x_test,y_test,color='red')
plt.plot(x_test,regressor.predict(x_test),color="blue")
plt.title("Closing Price vs Volume(test data)")
plt.xlabel('Closing Price')
plt.ylabel('Volume')
plt.show()
print("intercept",regressor.intercept_)
print("coefficent of regression",regressor.score(x,y))
print("slope",regressor.coef_)


#stock="GOOG"plt.xlabel('Years of Experience')
#plt.ylabel('Salary')
#df=pd.DataFrame(csv.symbol==stock)
#x=np.array(df['close'])
#y=np.array(df['close'].shift(-5))
#print(x)
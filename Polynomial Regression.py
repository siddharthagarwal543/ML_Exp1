import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

x=np.random.rand(100,1)
y=4+2*x+0.2*np.random.randn(100,1)+5*x**2
poly_features=PolynomialFeatures(degree=10,include_bias=False)
x_poly=poly_features.fit_transform(x)
reg=LinearRegression().fit(x_poly,y)
print(x_poly,y)
x_vals=np.linspace(-2,2,100).reshape(-1,1)
x_vals_poly=poly_features.transform(x_vals)

y_vals=reg.predict(x_vals_poly)


plt.scatter(x,y)
plt.plot(x_vals,y_vals,color="r")
plt.show()
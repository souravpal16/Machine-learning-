import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('Covid_India_Daily_Cases.csv')
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
poly_reg=PolynomialFeatures(degree=4)
x_poly=poly_reg.fit_transform(x)
line_reg1=LinearRegression()
line_reg1.fit(x_poly,y)

plt.scatter(x,y,color="red")
plt.plot(x,line_reg1.predict(x_poly),color="blue")
#plt.show()

a=line_reg1.predict(poly_reg.fit_transform([[103]]))
print(a)


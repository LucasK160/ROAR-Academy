import numpy as np

A= np.array([[6.,-9.,1.],[4.,24.,8.]])
CONSTANT=2
 
result=CONSTANT * A

B=np.array([[1.,0.],[0.,1.],[1.,1.]])

dot_product=np.dot(A,B)

print(result, dot_product)

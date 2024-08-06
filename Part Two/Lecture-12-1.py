import numpy as np

v = np.array([2., 2., 4.])



e0 = np.array([1., 0., 0.])
e1 = np.array([0., 1., 0.]) 
e2 = np.array([0., 0., 1.])

proj_e0 = np.dot(v, e0)
proj_e1 = np.dot(v, e1)
proj_e2 = np.dot(v, e2)

print(f"Projection onto e0 :", proj_e0)
print(f"Projection onto e0 :", proj_e1)
print(f"Projection onto e0 :", proj_e2)


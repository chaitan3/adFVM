import numpy as np
import matplotlib.pyplot as plt

viscosity = np.array([5e-4,1e-4,1e-3,6e-4,4e-4,5e-3,1e-2])
sens = np.array([-23.545240152,-757.17808868,99.2337272681,38.1457597971,-119.314335486,-37.6285100821,-31.1810471935])
perturb = -37.5296614557
error = 100*(sens-perturb)/abs(perturb)

plt.semilogx(viscosity, np.zeros_like(viscosity))
plt.semilogx(viscosity, error, 'b.',markersize=20)
plt.xlabel('viscosity scaling factor')
plt.ylabel('percent error in sensitivity')
plt.show()

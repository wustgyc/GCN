import numpy as np
from sklearn.impute import SimpleImputer

a=np.array(a)
print(a)
for i in range(3):
    for j,row in enumerate(a):
        if np.isnan(row[i]):
            print(row)

a=77
print(np.isnan(a))
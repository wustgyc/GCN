import numpy as np
from sklearn.impute import SimpleImputer
a=[1,1,3,2,4,1,1,33,1]
for i,row in enumerate(a):
    while i<len(a) and a[i]==1:
        del a[i]
print(a)
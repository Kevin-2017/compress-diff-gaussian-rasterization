import numpy as np
from icecream import ic 
data = np.load("imp_socre.npz")
lst = data.files
for item in lst:
        ic(item)
        ic(data[item].shape)
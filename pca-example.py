import numpy as np
from sklearn import decomposition


pca = decomposition.PCA(n_components=1)

x = np.array([
    [1,2, 3],
    [4,4,5],
])
c = pca.fit_transform(x)
print c
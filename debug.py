#%%
from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt
import seaborn
seaborn.set()

x = (np.random.randn(4096) * .5 + 1), (np.random.randn(4096) * 2 - 5)
x = np.concatenate(x, -1)
plt.hist(x, 100)
plt.show()

#%%

gmm = GaussianMixture(2).fit(x.reshape(-1, 1))
print(gmm.weights_.reshape(-1))
print(gmm.means_.reshape(-1))
print(gmm.covariances_.reshape(-1))

# %%

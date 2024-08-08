import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

header = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pd.read_csv('./data/pima-indians-diabetes.data.csv', names =header)

plt.clf()
# data.hist(figsize=(12,10), bins=5)
# plt.tight_layout()
# plt.savefig("./results/histogram.png")
# data.plot(kind="box", subplots=True, figsize=(12,18), layout=(3,3), sharex=False, sharey=False)
# data.plot(kind='density', figsize=(12,10), subplots=True, layout=(3,3), sharex=False)
# plt.savefig("./results/desity.png")

pd.plotting.scatter_matrix(data)
plt.savefig('./results/scatter.png')
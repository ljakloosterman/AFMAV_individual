import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import os

path = os.getcwd()
os.chdir(path)

files = os.listdir(path)
epoch_loss = []

for file in files:
  if file.endswith('.npy'):
    epoch_loss.append(file)
epoch_loss.sort()

matplotlib.rcParams['figure.figsize'] = [10, 5]

for file in epoch_loss:
  plt.plot(range(1,21), np.load(file))

plt.legend(['Batch size 1', 'Batch size 2', 'Batch size 4', 'Batch size 8'], fontsize=18)
plt.xlabel('Epochs', fontsize=18)
plt.ylabel('Loss', fontsize=18)
plt.show()
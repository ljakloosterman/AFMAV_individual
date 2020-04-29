import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import os


path = os.getcwd()
os.chdir(path)

files = os.listdir(path)

fpr = []
tpr = []

for file in files:
  if file.startswith('fpr'):
    fpr.append(file)
  elif file.startswith('tpr'):
    tpr.append(file)
fpr.sort()
tpr.sort()

matplotlib.rcParams['figure.figsize'] = [10, 5]

for i in range(len(fpr)):
  plt.plot(np.load(fpr[i]), np.load(tpr[i]))
plt.legend(['10 epochs', '1 epochs', '50 epochs', '5 epochs'], fontsize=18)
plt.xlabel('False Positive Ratio', fontsize=18)
plt.ylabel('True Positive Ratio', fontsize=18)
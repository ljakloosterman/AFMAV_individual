from Functions import load_img, load_testmask
import torchvision.models as models
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import numpy as np
import torch
import time
import os

# Obtain directory to all test images and append them to a sorted list
path = os.getcwd()
testimgpath = path + '/test_images/'
testmaskpath = path + '/test_masks/'

testimg = os.listdir(testimgpath)
testmask = os.listdir(testmaskpath)

# Selecting device on which testing will be done
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
  print("Testing on GPU")
else:
  print('Testing on CPU')
  
# Loading the untrained model
resnet101 = models.segmentation.fcn_resnet101(pretrained=False, progress=True, num_classes=2, aux_loss=None)
resnet101.to(device)

# Load desired model weights
model_name = 'resnet101_50_epochs'
path_model = path + '/saved_models/' + model_name
resnet101.load_state_dict(torch.load(path_model, map_location=torch.device(device)))

img_list = []
out_list = []
mask_list = []
fpr_list = [0]
tpr_list = [0]
auc_list = []
time_list = []

# Main testing loop
for i in range(len(testimg)):

  # Loading test image, tranform for display and append to list
  inp = load_img(testimgpath + testimg[i])
  img = inp.permute(1,2,0).cpu().numpy()
  img_list.append(img)

  # Calculating timed output of test image, transform for display and append to list
  t0 = time.time()
  out = resnet101(inp.unsqueeze(0))['out']
  elapsed_time = time.time() - t0
  out = 1 - torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
  out_list.append(out)

  # Loading mask image and append to list
  mask = load_testmask(testmaskpath + testmask[i])
  mask_list.append(mask)

  # Calculate individual fpr and tpr and append to lists
  fpr, tpr, thresholds = metrics.roc_curve(out.flatten(), mask.flatten())
  fpr_list.append(fpr[1])
  tpr_list.append(tpr[1])

  # Calculate individual Area Under Curve and append to list
  auc = metrics.roc_auc_score(out.flatten(), mask.flatten())
  auc_list.append(auc)

  # Append calculation time to list
  time_list.append(elapsed_time)

  # Print individual image results
  print('Figure ' + str(i+1) + ': Time = ' + str(elapsed_time)[:7] + ' seconds, AUC = ' + str(auc)[:7])

# Complete fpr and tpr lists
fpr_list.append(1)
tpr_list.append(1)

# Calculate total area under curve
total_auc = metrics.auc(sorted(fpr_list), sorted(tpr_list))

# Print and plot all results
print('Individual area under curve: Average = ' + str(np.mean(auc_list))[:7] + ' , Max = ' + str(np.amax(auc_list))[:7] + ' , Min = ' + str(np.amin(auc_list))[:7])
print('Total area under curve: ' + str(total_auc)[:7])
plt.figure(figsize=(5,5))
plt.plot(sorted(fpr_list), sorted(tpr_list))
plt.xlabel('False Positive Ratio', fontsize=18)
plt.ylabel('True Positive Ratio', fontsize=18)
plt.title('ROC Curve', fontsize=18)
plt.show()
print('Calculation time: Average = ' + str(np.mean(time_list))[:7] + ' seconds, Max = ' + str(np.amax(time_list))[:7] + ' seconds, Min = ' + str(np.amin(time_list))[:7] + ' seconds')

# Showing images and plotting individual ROC curves
for i in range(len(testimg[:5])): # Currently shows results of 4 images
  plt.rcParams.update({'figure.max_open_warning': 0})
  fig, axs = plt.subplots(nrows=1, ncols=4, figsize = (4*4,4))
  axs[0].imshow(img_list[int(i/4)])
  axs[0].set_title('Input image', fontsize=18)
  axs[1].imshow(out_list[int(i/4)], cmap='gray')
  axs[1].set_title('Output image', fontsize=18)
  axs[2].imshow(mask_list[int(i/4)], cmap='gray')
  axs[2].set_title('Mask image', fontsize=18)
  axs[3].plot([0., fpr_list[int(i/4) + 1], 1.], [0., tpr_list[int(i/4) + 1], 1.])
  axs[3].set_title('ROC Curve', fontsize=18)
  axs[3].set_xlabel('False Positive Ratio')
  axs[3].set_ylabel('True Positive Ratio')
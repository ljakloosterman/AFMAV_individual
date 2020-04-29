from Functions import load_img, load_mask
import torchvision.models as models
import matplotlib.pyplot as plt
from torch.utils import data
import torch.nn as nn
import numpy as np
import torch
import tqdm
import time
import os

# Obtain directory to all training images and append them to a sorted list
path = os.getcwd()
trainingimgpath = path + '/training_images/'
trainingmaskpath = path + '/training_masks/'
trainingimg = os.listdir(trainingimgpath)
trainingmask = os.listdir(trainingmaskpath)
trainingimg.sort()
trainingmask.sort()

# Selecting device on which training will be done
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
  print("Training on GPU")
else:
  print('Training on CPU')

# Loading the untrained model
resnet101 = models.segmentation.fcn_resnet101(pretrained=False, progress=True, num_classes=2, aux_loss=None)
resnet101.to(device)

trainingimg_list = []
trainingmask_list = []

# Loading all training images and masks
with tqdm.tqdm(total=len(trainingimg)) as pbar:
  for i in range(len(trainingimg)):
    img = load_img(trainingimgpath + trainingimg[i])
    mask = load_mask(trainingmaskpath + trainingmask[i])

    trainingimg_list.append(img)
    trainingmask_list.append(mask)

    pbar.update(1)

# Appending all images and mask to lists
trainingimg_list = torch.stack(trainingimg_list)
trainingmask_list = torch.stack(trainingmask_list)

# Construct a dataloader
trainingset = data.TensorDataset(trainingimg_list, trainingmask_list)
training_generator = data.DataLoader(trainingset, batch_size=1, shuffle='True')

# Training parameters
num_epoch = 50
criterion = nn.MSELoss()
optimizer = torch.optim.Adadelta(resnet101.parameters())

# Make folder to save models of it does not exist yet
savemodelpath = path + '/saved_models/' 
if not os.path.exists(savemodelpath):
    os.makedirs(savemodelpath)

epoch_loss_list = []
# Start timer
t0 = time.time()

# Main training loop 
for epoch in range(num_epoch):
  loss_list = []
  # Loop for each apoch
  with tqdm.tqdm(total=len(training_generator)) as pbar:
    for batch, (img, mask) in enumerate(training_generator):

      # Run the forward pass
      out = resnet101(img)['out']
      loss = criterion(out, mask)
      loss_list.append(loss.item())

      # Backprop and optimisation
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      pbar.update(1)

  # Calculate and append epoch loss to list
  epoch_loss = np.mean(loss_list)
  epoch_loss_list.append(epoch_loss)

  # Save every 10 epochs
  if epoch % 10 == 9:
    model_name = 'resnet101_' + str(epoch+1) + '_epochs'
    path_model = savemodelpath + model_name
    torch.save(resnet101.state_dict(), path_model)
  
  # Print results
  print("Epoch {}".format(epoch+1), "loss: {}".format(epoch_loss))
print('Training took {} seconds'.format(time.time() - t0))

plt.figure(figsize=(5,5))
plt.plot(range(1,num_epoch+1),epoch_loss_list)
plt.title('Epoch loss', fontsize = 18)
plt.xlabel('Epochs', fontsize=18)
plt.ylabel('Loss', fontsize=18)
plt.show
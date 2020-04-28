import torchvision.transforms as T
import numpy as np
import torch
import PIL

# Convert image to tensor and normalize
trf = T.Compose([T.ToTensor(), 
                 T.Normalize(mean = [0.485, 0.456, 0.406], 
                              std = [0.229, 0.224, 0.225])])

def load_img(path):
  'Loading image and resize'
  img = PIL.Image.open(path)
  img = T.functional.resize(img, size=(360,360))
  img = trf(img)
  return img

def load_mask(path):
  'Loading mask, resize is and tranform to desired form for training'
  mask = PIL.Image.open(path)
  gate = T.functional.resize(mask, size=(360,360))
  background = PIL.ImageOps.invert(gate)
  mask = torch.stack([T.functional.to_tensor(gate).squeeze(),T.functional.to_tensor(background).squeeze()])
  return mask

def load_testmask(path):
  'Loading mask, resize and tranform to desired form for testing'
  mask = PIL.Image.open(path)
  mask = T.functional.resize(mask, size=(360,360))
  mask = np.array(mask)
  return mask
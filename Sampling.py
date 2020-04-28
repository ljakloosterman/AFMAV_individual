from shutil import copyfile
import random as rnd
import tqdm
import os


# Locate dataset
path = os.getcwd()
dataset = os.listdir(path + '/WashingtonOBRace')

# Appending all images to a list
trainingimgtab = []
for i in dataset:
    if i[0:3] == 'img':
        trainingimgtab.append(i)
        
# Random sample 10% of the images for testing
rnd.seed(1)        
testimgtab = rnd.sample(trainingimgtab, k=int(len(trainingimgtab)/10))

# Deleting sampled test images from training images
for i in testimgtab:
    trainingimgtab.remove(i)
    
# Making training and test mask lists
trainingmasktab = []
testmasktab = []
for i in trainingimgtab:
    trainingmasktab.append(i.replace('img','mask'))
for i in testimgtab:
    testmasktab.append(i.replace('img','mask'))
    
# Making new folders
testimgpath = path + '/test_images/' 
if not os.path.exists(testimgpath):
    os.makedirs(testimgpath)
trainingimgpath = path + '/training_images/' 
if not os.path.exists(trainingimgpath):
    os.makedirs(trainingimgpath)    
trainingmaskpath = path + '/training_masks/' 
if not os.path.exists(trainingmaskpath):
    os.makedirs(trainingmaskpath)
testmaskpath = path + '/test_masks/' 
if not os.path.exists(testmaskpath):
    os.makedirs(testmaskpath)    

# Saving images and masks to correct folder
with tqdm.tqdm(total=len(testimgtab)) as pbar:
  for i in testimgtab:
    copyfile(str(path + '/WashingtonOBRace/' + i), str(path + '/test_images/' + i))
    pbar.update(1)
with tqdm.tqdm(total=len(trainingimgtab)) as pbar:
  for i in trainingimgtab:  
    copyfile(str(path + '/WashingtonOBRace/' + i), str(path + '/training_images/' + i))
    pbar.update(1)
with tqdm.tqdm(total=len(trainingmasktab)) as pbar:
  for i in trainingmasktab:
    copyfile(str(path + '/WashingtonOBRace/' + i), str(path + '/training_masks/' + i))
    pbar.update(1)
with tqdm.tqdm(total=len(testmasktab)) as pbar:
  for i in testmasktab:
    copyfile(str(path + '/WashingtonOBRace/' + i), str(path + '/test_masks/' + i))
    pbar.update(1)
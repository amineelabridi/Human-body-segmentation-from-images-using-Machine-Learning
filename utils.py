import numpy as np
import os
import matplotlib.pyplot as plt
from random import randrange

import skimage.io as io
from skimage.color import gray2rgb, rgb2gray
from skimage.transform import resize, rotate

#size of images
W, H, channels = 128, 128, 3
data_augmentation = 3

#reads the images and saves them into a numpy array

class Dataset:
    
    #construct a Dataset object with images, and true pixel-wise labels 
    #it has a number of images = train_size
    
    def __init__(self, path, data_augmentation, W, H, channels, train_size=None):
        
        self.path = path
        self.train_path = path + "/Training_Images"
        self.gt_path = path + "/Ground_Truth"
        self.dataset_size = len(os.listdir(self.train_path)) 
        
        
        if (train_size == None or train_size >= self.dataset_size):
            self.train_size = self.dataset_size
            print ("The train size chosen is superior to the dataset size. It has been set to", self.train_size, "(Data augmentation on", self.dataset_size, "images)")
        else:
            self.train_size = train_size
    
        self.X = np.zeros(( data_augmentation*self.train_size, )) #images as numpy array of size (Width, Height, channels=3 as RGB), where these paramerters are variable in each image
        self.labels = np.zeros(( data_augmentation*self.train_size, )) #groud truth masks
#         self.predicted_mask = np.zeros(( data_augmentation*self.train_size, )) #Predicted segmentation labels
        
    
    #function to load images from the data folder.
    #We perform data augmentation by flipping the image left right, up down and rotation of 45°
    
    def load_and_pepare_train_data(self):
        
        filenames =  os.listdir(self.train_path)
        List = []
        for index in range(1,  self.train_size+1):
            img = io.imread(os.path.join(self.train_path, str(index) + ".jpg"))
            img = gray2rgb(img)
            img = resize(img, (W, H, channels), anti_aliasing=True)
            
            lr_img = np.fliplr(img)
            ud_img = np.flipud(img)
#             rotated_img = rotate(img, 45)            
            
            if img is None:
                print ("Error. There has been a problem reading the image", index)
            else:
                List.append(img)
                List.append(lr_img)
                List.append(ud_img)
#                 List.append(rotated_img)
        self.X = np.array(List)
        
        return self.X

    
    
    
    def load_and_pepare_ground_truth(self):
        
        filenames =  os.listdir(self.gt_path)
        List = []
        
        for index in range(1,  self.train_size+1):
            img = io.imread(os.path.join(self.gt_path, str(index) + ".png"), as_gray= True)
            img = resize(img, (W, H), anti_aliasing=True)

            lr_img = np.fliplr(img)
            ud_img = np.flipud(img)
#             rotated_img = rotate(img, 45)  
            
            if img is None:
                print ("Error. There has been a problem reading the image", index)
            else:
                List.append(img)
                List.append(lr_img)
                List.append(ud_img)
#                 List.append(rotated_img)
        
        self.labels = np.array(List)
        
        return self.labels
    
    
    def subplot_segmentation(self, image_index, mask):
        
        if (image_index >= 4*self.train_size):
            image_index = 4*self.train_size - 1
        
        img = self.X[image_index-1]
        gt_mask = self.labels[image_index-1]
        
        # get segmented pixels coordinates to plot RGB segmented image
        segmentation_pixels = np.where((mask == 0))
        segmented_rgb = img.copy()
        
        for i in range(len(segmentation_pixels[0])):
            row, col = segmentation_pixels[0][i], segmentation_pixels[1][i]
            segmented_rgb[row, col] = 0

        fig, ax = plt.subplots(2, 2, figsize=(15,8))
        fig.subplots_adjust(wspace=0.6, hspace=0.6)
        
        ax[0,0].imshow(self.X[image_index-1])
        ax[0,0].set_title("Original image")
        ax[0,1].imshow(gt_mask)
        ax[0,1].set_title("Ground truth")
        ax[1,0].imshow(mask)
        ax[1,0].set_title("Segmentation result as Binary image")
        ax[1,1].imshow(segmented_rgb)
        ax[1,1].set_title("Segmentation result in RGB")
        
        
        
    def subplot_img_and_gt_mask(self, image_index):
        
        if (image_index >= 4*self.train_size):
            image_index = 4*self.train_size - 1
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,8))
        fig.subplots_adjust(wspace=0.6, hspace=0.6)
        
        ax1.imshow(self.X[image_index-1])
        ax1.set_title("Original image", size= 'x-large')
        ax1.axis('off')

        ax2.imshow(self.labels[image_index-1])
        ax2.set_title("Ground truth", size= 'x-large')
        ax2.axis('off')

def accuracy(pred_mask, true_mask):
    if (pred_mask.shape != true_mask.shape):
        print ("error. Size mismatch while computing accuracy",)
        return 0
    else:
        count1 = np.count_nonzero( (pred_mask - true_mask) == 0)
        count2 = np.count_nonzero( ( (1-pred_mask) - true_mask ) == 0)
        count = max(count1, count2)
        return count/(pred_mask.shape[0] * pred_mask.shape[1]) * 100
    
def precision(pred_mask, true_mask):
    
    #How accurate the positive predictions are
    
    if (pred_mask.shape != true_mask.shape):
        print ("error. Size mismatch while computing accuracy",)
        return 0
    else:
        count1 = np.count_nonzero( (pred_mask + true_mask) > 1) #how many True Positive (human body pixels)
        count2 = np.count_nonzero( ( (1-pred_mask) + true_mask ) > 1)
        count3 = np.count_nonzero( pred_mask > 0) # TP + FP (total positive predictions)
        count = max(count1, count2)
        return count/count3 * 100
    
def recall(pred_mask, true_mask):
    
    #Coverage of actual positive sample - Human body
    
    if (pred_mask.shape != true_mask.shape):
        print ("error. Size mismatch while computing accuracy",)
        return 0
    else:
        count1 = np.count_nonzero( (pred_mask + true_mask) > 1)
        count2 = np.count_nonzero( ( (1-pred_mask) + true_mask ) > 1)
        count3 = np.count_nonzero( true_mask > 0)
        count = max(count1, count2)
        return count/count3 * 100
    
def specificity(pred_mask, true_mask):
    
    #Coverage of actual negative sample - Background
    
    if (pred_mask.shape != true_mask.shape):
        print ("error. Size mismatch while computing accuracy",)
        return 0
    else:
        count1 = np.count_nonzero( (pred_mask + true_mask) == 0)
        count2 = np.count_nonzero( ( (1-pred_mask) + true_mask ) == 0)
        count3 = np.count_nonzero( true_mask == 0)
        count = max(count1, count2)
        return count/count3 * 100
    
    
def compute_metrics(pred_mask, true_mask):
    acc    =  accuracy(pred_mask, true_mask)
    prec   = precision(pred_mask, true_mask)
    rec    = recall(pred_mask, true_mask)
    spec   = specificity(pred_mask, true_mask)
    
    return (acc, prec, rec, spec)

def balanced_accuracy(pred_mask, true_mask):
    rec    = recall(pred_mask, true_mask)
    spec   = specificity(pred_mask, true_mask)
    return ((rec+spec)/2)

def print_metrics(pred_mask, true_mask):
    acc    =  accuracy(pred_mask, true_mask)
    prec   = precision(pred_mask, true_mask)
    rec    = recall(pred_mask, true_mask)
    spec   = specificity(pred_mask, true_mask)
    bal_acc = balanced_accuracy(pred_mask, true_mask)
    
    print ("percentage of background in the image:", 100*np.count_nonzero((true_mask == 0))/true_mask.size )
    print ("percentage of body in the image:", 100*np.count_nonzero((true_mask != 0))/true_mask.size )

    print("\naccuracy:",acc)
    print("precision:",prec)
    print("recall:",rec)
    print("specificity:",spec)
    print("balanced accuracy:", bal_acc)
    
    return (acc, prec, rec, spec, bal_acc)


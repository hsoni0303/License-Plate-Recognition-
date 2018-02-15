import os 
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray
import numpy as np
from skimage.io import imread

letters = [
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D',
            'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T',
            'U', 'V', 'W', 'X', 'Y', 'Z'
        ]

def read_training_data(train_dir):
    image_data = []
    target_data = []
    for each_letter in letters:
        for each in range(10):
            image_path = os.path.join(train_dir, each_letter, each_letter + "_" + str(each) + '.jpg')
            #read each image of each char
            image_details = imread(image_path)    
            # Converted into a gray scale
            image_details = rgb2gray(image_details)
            
            binary_image = image_details < threshold_otsu(image_details)
            # we need to convert 2d array to 1d becoz ml classifier require 
            # 1d array of each sample
            
            flat_bin_image = binary_image.reshape(-1)
            image_data.append(flat_bin_image)
            target_data.append(each_letter)
            
    return (np.array(image_data), np.array(target_data))

def cross_validation(model, num_of_fold, train_data, train_label):
    # this is to measure accuracy of model
    # num_of_fold determines the type of validation
    # e.g if num_of_fold is 4, then we are performing a 4-fold cross validation
    # it will divide the dataset into 4 and use 1/4 of it for testing
    # and the remaining 3/4 for the training
    
    accuracy_result = cross_val_score(model, train_data, train_label, cv=num_of_fold)

    print("cross validation result for ", str(num_of_fold), " -fold")
    
    print(accuracy_result * 100)
    
current_dir = os.getcwd()
training_dataset_dir = os.path.join(current_dir, 'training_data/train20X20')
image_data, target_data = read_training_data(training_dataset_dir)
svc_model = SVC(kernel='linear', probability=True)    # Probability is true to show how sure model is of its prediction
cross_validation(svc_model, 4, image_data, target_data)
svc_model.fit(image_data, target_data)   # train model on input data

# joblib will save trained model
save_dir = os.getcwd()
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
joblib.dump(svc_model, save_dir+'/svc.pkl')





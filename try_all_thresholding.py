from skimage.filters import try_all_threshold
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu      
import matplotlib.pyplot as plt 


vehicle_image = imread('image1.jpg')
gray_image = rgb2gray(vehicle_image)
    
# skimage change into gray using 0 to 1 scale 
# so for convinence we extend range back to 0 to 255
gray_car_image = gray_image * 255

fig, ax = try_all_threshold(gray_car_image, figsize=(10, 8), verbose=False)
plt.show()



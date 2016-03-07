__author__ = 'Shuran'
import matplotlib.pyplot as plt
from imutils import paths

from skimage.feature import hog
from skimage import data, color, exposure
import cv2

def inverte(imagem):
    imagem = (255-imagem)
    #cv2.imshow("new", imagem)
    #cv2.waitKey(0)
    return imagem

def showHOGpic(image):
    image = color.rgb2gray(image)

    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualise=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

    #original image
    ax1.axis('off')
    ax1.imshow(image, cmap=plt.cm.gray)
    ax1.set_title('Input image')
    ax1.set_adjustable('box-forced')

# Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))
    #HOG image
    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title('Histogram of Oriented Gradients')
    ax1.set_adjustable('box-forced')
    plt.show()



for trainimage in paths.list_images("neg"):
    img = cv2.imread(trainimage)
    showHOGpic(img)
    #inv_img = inverte(img)
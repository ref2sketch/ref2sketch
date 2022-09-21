import numpy as np
from PIL import Image
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import cv2
from matplotlib import pyplot as plt

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg",".PNG"])


def load_img(filepath):
    img = Image.open(filepath).convert('L')
    img = img.resize((256, 256), Image.BICUBIC)
    return img

def test_load_img(filepath):
    img = Image.open(filepath).convert('L')
    w,h = img.size
    img = img.resize((256, 256), Image.BICUBIC)
    return img,w,h

def save_img(image_tensor,w,h ,filename):
    image_numpy = image_tensor.float().numpy()
    image_numpy2 = image_tensor.float().numpy().squeeze()

    #plt.matshow(image_numpy2)
    #plt.colorbar()
    #plt.show()


    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_numpy = image_numpy.clip(0, 255)
    image_numpy = image_numpy.astype(np.uint8)
    
    image_numpy = np.squeeze(image_numpy, axis=2)  # axis=2 is channel dimension 

    #print(image_numpy.shape)
    image_pil = Image.fromarray(image_numpy)

    image_pil = image_pil.resize((int(w*(0.7)), int(h*(0.7))))
    image_pil.save(filename)
    print("Image saved as {}".format(filename))



def tensor_gauss_kernel(input):
    # Create gaussian kernels
    im_array = np.asarray(input)
    kernel1d  = cv2.getGaussianKernel(5, 3)
    kernel2d = np.outer(kernel1d, kernel1d.transpose())
    low_im_array = cv2.filter2D(im_array, -1, kernel2d) # convolve
    if np.sum(im_array) < np.sum(low_im_array):
        cal = np.linalg.norm(im_array-low_im_array)

        #cal = im_array,low_im_array
    else:
        cal = 0

    return cal



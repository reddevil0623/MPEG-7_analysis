import os
import glob
import skimage.morphology as skmorph
import skimage.io as skio
from tqdm import tqdm
import numpy as np
from scipy.ndimage import binary_fill_holes
from PIL import Image


rootfolder = '/home/adam/cellprofiler/MPEG-7_dataset/Data/MPEG7_CE-Shape-1_Part_B' 

imgfiles = glob.glob(os.path.join(rootfolder, '*.gif')) # 1402 images. 
print(imgfiles)
# from the filenames parse the image and the label 
n_contour_pts = 250 # the sampling may be the problem ? 
pad_img = 20

imglabels = []
imgcontours = []
imgfeats = []
imgpatches = []

"""
Iterate over the image, read, compute boundary contour and the geometrical features. 
"""
for ii in tqdm(range(len(imgfiles))[:]):
    imgfile = imgfiles[ii]
    imglabel = os.path.split(imgfile)[-1].split('-')[0]
    
    img = skio.imread(imgfile)
    img = img[0]
    print(img.shape)
    if len(img.shape) > 2:
        img = img[...,1]

  #  img = np.pad(img, [[pad_img, pad_img],
  #                      [pad_img, pad_img]]) # prevent contours breaking. 
    
    img = skmorph.binary_dilation(img>0, skmorph.disk(1))
    img = binary_fill_holes(img)
    print("/home/adam/cellprofiler/MPEG-7_filled/"+os.path.split(imgfile)[-1]+".gif")

    skio.imsave("/home/adam/cellprofiler/MPEG7_filled_new/"+os.path.split(imgfile)[-1],img)


## Motion artefacts generation


# !pip install torchio==0.18.90 
# !pip install pandas 
# !pip install matplotlib 
# !pip install seaborn 
# !pip install scikit-image 

# Importing modules
import os
import copy
import pprint
import torch
import torchio as tio
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()





# Base (taking the clean images) and output (putting the generated images) directories 
base_directory = '/Users/irismarmouset-delataille/Desktop/artefacts/'
output_directory = '/Users/irismarmouset-delataille/Desktop/GeneratedArtefacts'


# List of the artefacts created with the functions associated (cf end of the file for description of the functions)
add_bias = tio.RandomBiasField() # Magnetic field inhomogeneities in the MRI scanner produce low-frequency intensity distortions in the images
add_spike = tio.RandomSpike() # signal peak in  𝑘 -space,
add_ghosts = tio.RandomGhosting() # removing every  𝑛 th plane from the k-space
add_motion = tio.RandomMotion()  # filling the  𝑘 -space with random rigidly-transformed versions of the original images
random_flip = tio.RandomFlip(axes=['inferior-superior'], flip_probability=1) # flipping the images 
random_affine = tio.RandomAffine(scales=(1.5, 1.5)) # zooming in the images

functions = [add_bias, add_spike, add_ghosts, add_motion, random_flip, random_affine]


# Looping through all the subfolders in the base directory: 
for subfolder_name in os.listdir(base_directory):
    # Check if the subfolder starts with "sub-"
    if subfolder_name.startswith('sub-'):
        subfolder_path = os.path.join(base_directory, subfolder_name)

        # Check if "anat" folder exists in the subfolder
        anat_folder_path = os.path.join(subfolder_path, 'anat')
        if os.path.exists(anat_folder_path):
            # Find the clean image in the "anat" folder
            for file_name in os.listdir(anat_folder_path):
                if file_name.endswith('_acq-standard_T1w.nii.gz'): 
                    image_path = os.path.join(anat_folder_path, file_name)

                    # Create a subject with torchIO
                    subject = tio.Subject(mri=tio.ScalarImage(image_path))
                    
                    # Process the subjects with TorchIO functions + save them in the output directory 
                    for function in functions:
                        image = function(subject)
                        output_path = os.path.join(output_directory, f"{subfolder_name}_generated_{function}_T1W.nii.gz")
                        image.mri.save(output_path)


















##  Functions to use:

# NB: to look for the historic of an image: pprint.pprint(image.history)

# 1 - Random anisotropy:  --> not used yet
# Many medical images are acquired with anisotropic spacing, i.e. voxels are not cubes. 
# Researchers typically use anisotropic resampling for preprocessing before feeding the images into a neural network. 
# We can simulate this effect downsampling our image along a specific dimension and resampling back to an isotropic spacing.
# Function: Downsample an image along an axis and upsample to initial space
# Args:     axes: Axis or tuple of axes along which the image will be downsampled, = (0, 1, 2)
#           downsampling: Downsampling factor :math:`m \gt 1`. If a tuple
#              :math:`(a, b)` is provided then :math:`m \sim \mathcal{U}(a, b)`, = (1.5, 5)
#           image_interpolation: Image interpolation used to upsample the image back to its initial spacing, = 'linear'
#           scalars_only: Apply only to instances of :class:`torchio.ScalarImage`, = True
#random_anisotropy = tio.RandomAnisotropy()
#anisotropic_image = random_anisotropy(subject)


# 2 - Random affine:  --> used, zoom in scales(1.5, 1.5)
# Simulate different positions and size of the patient within the scanner.
# Function : Apply a random affine transformation and resample the image.
# Args:      scales: Tuple  defining the scaling ranges, if unique value x: scaling value ~ U(1-x, 1+x), = 0.1
#            degrees: Tuple  defining the rotation ranges in degrees, if unique value  θ: rotation value ~ U(-x, x), = 10
#            translation: Tuple  defining the translation ranges in mm,  if unique value  t: translation value ~ U(-x, x), = 0
#            isotropic: If True, the scaling factor along all dimensions is the same, = False
#            center: If 'image', rotations and scaling will be performed around the image center,
#                    If 'origin', rotations and scaling will be performed around the origin in world coordinates, = 'image'
#            default_pad_value: filling values near the borders after the image is rotated, 'minimum'
#            image_interpolation : = 'linear'
#            label_interpolation: = 'nearest'
#            check_shape: check or not if the images are in different physical spaces, = True
#random_affine = tio.RandomAffine(scales=(1.5, 1.5))
#slice_affine = random_affine(subject)


# 3 - Random flip: --> used with a systematic flipping infero-superior 
# Flipping images is a very cheap way to perform data augmentation. 
# In medical images, it's very common to flip the images horizontally --> will also do that 
# If we don't know the image orientation : use anatomical labels instead.
# Function: Reverse the order of elements in an image along the given axes.
# Args:     axes: Index or tuple of indices of the spatial dimensions along which the image might be flipped, = 0
#           flip_probability: Probability that the image will be flipped, = 0.5
#random_flip = tio.RandomFlip(axes=['inferior-superior'], flip_probability=1)
#flipped_image = random_flip(subject)


# 4 - Random elastic transormation:  --> not used yet
# Simulate anatomical variations in our images: non-linear deformation
# Function: Apply dense random elastic deformation.
# Args:     num_control_points: Number of control points along each dimension of the coarse grid, the smaller the smoother deformations. = 7
#           max_displacement: Maximum displacement along each dimension at each control point,  = 7.5
#           locked_borders: keeping of displacement vectors, 
#                           0:yes, 1: no displacement of control points at the border of the coarse grid,
#                           2: no  displacement of control points at the border of the image,
#                            = 2
#           image_interpolation: = 'linear'
#           label_interpolation: = 'nearest'
#random_elastic = tio.RandomElasticDeformation()
#slice_elastic = random_elastic(subject)


# 5 - Random blur:  --> not used yet
# Function: Blur an image using a random-sized Gaussian filter
# Args:     std: Tuple, the ranges (in mm) of the standard deviations of the Gaussian kernels used to blur the image, = (0, 2)
#blur = tio.RandomBlur()
#blurred = blur(subject)


# 6 - Random noise:  --> not used yet
# Gaussian noise can be simulated 
# This transform is easiest to use after ZNormalization (mean set to 0a,d std to 1)
# Noise in MRI is actually Rician, but it is nearly Gaussian for SNR > 2 (i.e. foreground).
# Function: Add Gaussian noise with random parameters.
# Args:     mean: Mean of the Gaussian distribution from which the noise is sampled, mu ~ U(-x, x),  = 0
#           std: Standard deviation  of the Gaussian distribution from which the noise is sampled, std ~ U(0, x), = (0, 0.25)
#standardize = tio.ZNormalization()
#add_noise = tio.RandomNoise(std=0.5)
#standard = standardize(subject)
#noisy = add_noise(standard)


# 7 - Random bias field --> used, important 
# MRI-specific transform
# Magnetic field inhomogeneities in the MRI scanner produce low-frequency intensity distortions in the images, 
# which are typically corrected using algorithms such as N4ITK.  To simulate this artifact, we can use RandomBiasField.
# Function: Add random MRI bias field artifact
# Args:     coefficients: Maximum magnitude of polynomial coefficients, related to  artifact strength, float = 0.5
#           order: Order of the basis polynomial functions, related to the artifact frequency, integer = 3
#add_bias = tio.RandomBiasField(coefficients=1)
#mni_bias = add_bias(subject)


# 8 - Random spike --> used, important 
# MRI-specific transform
# Sometimes, signal peaks can appear in  𝑘 -space, manifesting as stripes in image space.
# Function: Add random MRI spike artifacts, distorsion increasing with both arguments
# Args:     num_spikes: Number of spikes  present in k-space, tuple, n ~ U(0, a) or n ~ U(a, b), = 1
#           intensity: Ratio  between the spike intensity and the maximum of the spectrum, r ~ U(a, b) or r ~ U(-a, a), = (1, 3)
#add_spike = tio.RandomSpike()
#with_spike = add_spike(subject)


# 10 - Random ghosting --> used, important 
# Ghosting artifacts, caused by patient motion and can be simulated by removing every  𝑛 th plane from the k-space.
# Function: Add random MRI ghosting artifact
# Args:     num_ghosts: Number of ‘ghosts’ n in the image, n ~ U(0, a) or n ~ U(a, b), = (4, 10)
#           axes: Axis along which the ghosts will be created, anatomical labels may also be used, (0, 1, 2)
#           intensity: Positive number representing the artifact strength s with respect to the maximum of the k-space, (0.5, 1)
#           restore: Number between 0 and 1 indicating how much of the -space center should be restored after 
#                    removing the planes that generate the artifact, = 0.02
#add_ghosts = tio.RandomGhosting()
#with_ghosts = add_ghosts(subject)


# 11 - Random motion --> used, important 
# If the patient moves during the MRI acquisition, motion artifacts will be present
# Artifact is generated by filling the  𝑘 -space with random rigidly-transformed versions of the original images 
# and computing the inverse transform of the compound  𝑘 -space.
# Function: Add random MRI motion artifact, , distorsion increasing with the arguments
# Args:     degrees: Tuple defining the rotation range in degrees of the simulated movements, θ ~ U(1-a, a) or  θ ~ U(a, b), = 10
#           translation: Tuple  defining the translation in mm of the simulated movements, = 10
#           num_transforms: Number of simulated movements, = 2
#           image_interpolation: = 'linear'
#add_motion = tio.RandomMotion()
#with_motion = add_motion(subject)





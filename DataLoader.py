import os
import torch
import torchio as tio
import numpy as np



# Base (taking the clean images) and output (saving the generated images) directories 
base_directory = '/Users/irismarmouset-delataille/Desktop/artefacts/'
output_directory = '/Users/irismarmouset-delataille/Desktop/GeneratedArtefacts'


# TorchIO function to generate of augmented images by flipping them along the 3 dimensions
flip_axes_0 = tio.RandomFlip(axes=0, flip_probability=1) # flipping the images along the 1st dimension
flip_axes_1 = tio.RandomFlip(axes=1, flip_probability=1) # flipping the images along the 2nd dimension
flip_axes_2 = tio.RandomFlip(axes=2, flip_probability=1) # flipping the images along the 3rd dimension
augmentations = [flip_axes_0, flip_axes_1, flip_axes_2]


# TorchIO function to generate artefacted images
add_bias = tio.RandomBiasField() # Magnetic field inhomogeneities in the MRI scanner produce low-frequency intensity distortions in the images
add_spike = tio.RandomSpike() # signal peak in  𝑘 -space,
add_ghosts = tio.RandomGhosting() # removing every  𝑛 th plane from the k-space
add_motion = tio.RandomMotion()  # filling the  𝑘 -space with random rigidly-transformed versions of the original images
random_affine = tio.RandomAffine(scales=(1.5, 1.5)) # zooming in the images
artefacts = [add_bias, add_spike, add_ghosts, add_motion, random_affine]



def generation_artefacts(base_directory, output_directory, artefacts):
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
                        # Generate artefacts + save them in the output directory 
                        for artefact in artefacts:
                            artefacted_image = artefact(subject)
                            output_path = os.path.join(output_directory, f"{subfolder_name}_generated_{artefact}_T1W.nii.gz")
                            artefacted_image.mri.save(output_path)
                    


def augmentation_images(base_directory, output_directory, augmentations):
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
                        # Generate artefacts + save them in the output directory 
                        for augmentation in augmentations:
                            augmented_image = augmentation(subject)
                            output_path = os.path.join(output_directory, f"{subfolder_name}_generated_{augmentation}_T1W.nii.gz")
                            augmented_image.mri.save(output_path)
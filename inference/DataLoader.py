
import os, torch, torchio as tio, numpy as np, subprocess, pandas as pd


BASEDIR = '/Users/irismarmouset-delataille/Desktop/artefacts/' # images to transform
GROUND_TRUTH = '/Users/irismarmouset-delataille/Desktop/artefacts/GeneratedImages/scores.tsv' # ground truth of the generated images
RESULTS = '/Users/irismarmouset-delataille/Desktop/artefacts/GeneratedImages/results.csv' # result of the inference on the generated images
TemporaryDir = '/Users/irismarmouset-delataille/Desktop/artefacts/GeneratedImages' # temporary storage of generated image

ground_truth_labels = pd.DataFrame(columns=['id', 'gt_score'])


# TorchIO function to generate augmented images by flipping them along the 3 dimensions
flip_0 = tio.RandomFlip(axes=0, flip_probability=1) # 1st dimension
flip_1 = tio.RandomFlip(axes=1, flip_probability=1) # 2nd dimension
flip_2 = tio.RandomFlip(axes=2, flip_probability=1) # 3rd dimension
flip_0_1 = tio.Compose([flip_0, flip_1]) # 1st & 2nd dimension
flip_0_2 = tio.Compose([flip_0, flip_2]) # 1st & 3nd dimension
flip_1_2 = tio.Compose([flip_1, flip_2]) # 2nd & 3rd dimension
flip_0_1_2 = tio.Compose([flip_0, flip_1, flip_2]) # 1st & 2nd & 3rd dimension
augmentations = {"flip_0":flip_0, "flip_1":flip_1, "flip_2":flip_2,"flip_0_1":flip_0_1, 
                 "flip_0_2":flip_0_2, "flip_1_2":flip_1_2, "flip_0_1_2":flip_0_1_2}

# TorchIO function to generate artefacted images
add_bias = tio.RandomBiasField() # Magnetic field inhomogeneities in the MRI scanner produce low-frequency intensity distortions in the images
add_spike = tio.RandomSpike() # signal peak in  𝑘 -space,
add_ghosts = tio.RandomGhosting() # removing every  𝑛 th plane from the k-space
add_motion = tio.RandomMotion()  # filling the  𝑘 -space with random rigidly-transformed versions of the original images
random_affine = tio.RandomAffine(scales=(1.5, 1.5)) # zooming in the images
artefacts = {"add_bias":add_bias, "add_spike":add_spike, "add_ghosts":add_ghosts, 
             "add_motion":add_motion, "random_affine":random_affine}

modifications = {**augmentations, **artefacts}


# Looping through all the subfolders in the base directory: 
for subfolder_name in os.listdir(BASEDIR):
     # Check if the subfolder starts with "sub-"
    if subfolder_name.startswith('sub-'):
        subfolder_path = os.path.join(BASEDIR, subfolder_name)
        # Check if "anat" folder exists in the subfolder
        anat_folder_path = os.path.join(subfolder_path, 'anat')
        if os.path.exists(anat_folder_path):
            # Find the clean image in the "anat" folder
            for file_name in os.listdir(anat_folder_path):
                if file_name.endswith('_acq-standard_T1w.nii.gz'): 
                    image_path = os.path.join(anat_folder_path, file_name)

                    # Create a subject from this clean image with torchIO
                    subject = tio.Subject(mri=tio.ScalarImage(image_path))
                    # Generate modified (flipped clean, or artefacted) images 
                    for modification_name, modification_function in modifications.items():
                        modified_image = modification_function(subject)

                        # 1 - Save its ground truth in the dedicated file along with its id
                        temp_path = os.path.join(TemporaryDir, f"{subfolder_name}_{modification_name}_T1W.nii.gz")
                        id = temp_path
                        gt_score = 0 if modification_name.startswith("flip") else 1
                        ground_truth_labels.loc[len(ground_truth_labels)] = [id, gt_score] 

                        # 2 - Temporarly save the image so it can be re-loaded with nib library
                        modified_image.mri.save(temp_path)

                        # 3 - Run the inference on it
                        command = ["python", "raw_inference.py", "-m", "20", "-i", temp_path, "-c", RESULTS]
                        subprocess.run(command)

                        # 4 - Delete the image saved
                        os.remove(temp_path)


# Save the ground truth file                     
ground_truth_labels.to_csv(GROUND_TRUTH, sep='\t', index=False)                    
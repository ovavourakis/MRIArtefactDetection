
import os, torch, torchio as tio, numpy as np, subprocess

# The image path to run the inference) assumes that this Python script is in the same directory as the raw_inference.py script.


BASEDIR = '/Users/irismarmouset-delataille/Desktop/artefacts/'
GROUND_TRUTH = '/Users/irismarmouset-delataille/Desktop/artefacts/GeneratedImages/scores.tsv'


ground_truth_labels = pd.DataFrame(columns=['id', 'gt_score', 'GeneratFct'])






ground_truth_labels['bin_gt'] = ground_truth_labels['score'].replace({1:0, 2:1, 3:1})
ground_truth_labels.rename(columns={'bids_name': 'id', 'score':'gt_score'}, inplace=True)

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

modifications = augmentations + artefacts


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

                    # Create a subject with torchIO
                    subject = tio.Subject(mri=tio.ScalarImage(image_path))
                    # Generate artefacted images 
                    for modification in modifications:
                        modified_image = modification(subject)
                        # 1- Save its status/ ground truth 
                        id = f"{subfolder_name}_{modification}"
                        gt_score = 0 if modification.startswith("flip") else 1
                        ground_truth_labels.loc[len(ground_truth_labels)] = [id, gt_score, modification]

                        # 2- Running the inference on it
                        # - need to modify the way the function reads the path to the image
                        # - need to modify how the id is stored in the result.csv file
                        # command = ["python", "raw_inference.py", "-m", "20", "-i"] + artefacted_image ["-c", results.csv]
                        # subprocess.run(command)
                        # output_path = os.path.join(output_directory, f"{subfolder_name}_generated_{artefact}_T1W.nii.gz")
                        # artefacted_image.mri.save(output_path)



# Save the ground truth file                     
ground_truth_labels.to_csv(GROUND_TRUTH, sep='\t', index=False)                    
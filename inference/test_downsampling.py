
import os, torch, torchio as tio, numpy as np, subprocess, pandas as pd

# to do to use this script: 
# - in your directory of artefactsn create an empty directory named Downsampling
# - in raw_inference: 
#     if csv_filepath:
       # with open(csv_filepath, "w") as fd: MODIFY w by a otherwise overwritten each line
            #fd.write(os.linesep.join(csv_output) + os.linesep)
        #print("wrote results to {}".format(csv_filepath))
# use tio.Resample() ? couldn't make it work in my images (have a look at your imageu using image.plot())


BASEDIR = '/Users/irismarmouset-delataille/GE/artefacts_2/' # images to transform
GROUND_TRUTH = '/Users/irismarmouset-delataille/GE/artefacts_2/Downsampling/scores.tsv' # ground truth of the generated images
RESULTS = '/Users/irismarmouset-delataille/GE/artefacts_2/Downsampling/results.csv' # result of the inference on the generated images
TemporaryDir = '/Users/irismarmouset-delataille/GE/artefacts_2/Downsampling' # temporary storage of generated image

ground_truth_labels = pd.DataFrame()




# Looping through all the subfolders in the base directory: 
for subfolder_name in os.listdir(BASEDIR):
     # Check if the subfolder starts with "sub-"
    if subfolder_name.startswith('sub-'):
        subfolder_path = os.path.join(BASEDIR, subfolder_name)

        # Read the _scans.tsv file containing the ground truth labels
        scans_file_path = os.path.join(subfolder_path, f"{subfolder_name}_scans.tsv")
        scans_file = pd.read_csv(scans_file_path, sep='\t')

        # Store the content of the file to the ground truth file
        ground_truth_labels = pd.concat([ground_truth_labels, scans_file])

        # Check if "anat" folder exists in the subfolder
        anat_folder_path = os.path.join(subfolder_path, 'anat')
        if os.path.exists(anat_folder_path):
            # Find the images in the "anat" folder
            for file_name in os.listdir(anat_folder_path):
                if file_name.endswith('_T1w.nii.gz'): 
                    image_path = os.path.join(anat_folder_path, file_name)

                    # 1 - Create a subject from this image with torchIO
                    subject = tio.Subject(mri=tio.ScalarImage(image_path))

                    # 2 - Downsample to 256*256*64 the existing image 
                    # torchIO.resample or resize
                    resize = tio.Resize((256, 256, 64))
                    resized = resize(subject)

                    # 3 - Temporarly save the image so it can be re-loaded with nib library
                    temp_path = os.path.join(TemporaryDir, f"downsampled_{file_name}")
                    # ID in the gt file: anat/sub-000103_acq-standard_T1w.nii.gz
                    resized.mri.save(temp_path)

                    # 4 - Run the inference on it
                    command = ["python", "raw_inference.py", "-m", "20", "-i", temp_path, "-c", RESULTS]
                    subprocess.run(command)

                    # 5 - Delete the image saved
                    os.remove(temp_path)



# Save the ground truth file                     
ground_truth_labels.to_csv(GROUND_TRUTH, sep='\t', index=False)                    


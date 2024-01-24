
#!/usr/bin/env python

"""
This script runs MC inference on a set of images and ouputs the raw artefact-probability estimates
for each MC run for each image.

Model supposedly supports the following modalities:
"dwi", "flr", "mtOFF", "mtON", "pdw", "t1c", "t1g", "t1p", "t2w"
"""

import argparse, os, onnxruntime, multiprocessing, logging
from utils import get_subj_data, collate_inferences
from sys import stdout, argv
import os, torch, torchio as tio
import numpy as np
import csv
import matplotlib.pyplot as plt
import pandas as pd

WEIGHTS = os.path.join(os.path.dirname(__file__), 'weights/model.FINAL.onnx')

logger = logging.getLogger()

def parse_arguments(argv):
    parser = argparse.ArgumentParser(description="Image Inference Script")

    input_group = parser.add_mutually_exclusive_group(required=True)

    input_group.add_argument("-i", "--image_paths",
                             metavar="image_paths",
                             type=str,
                             nargs="+",
                             help="List of image paths to process")

    input_group.add_argument("-f", "--inputs_file",
                             metavar="inputs_file",
                             type=str,
                             help="Path to a file containing a list of image paths")

    parser.add_argument("-c", "--csv_filepath",
                        dest="csv_filepath",
                        type=str,
                        default='output.csv',
                        help="Optional path to an output CSV file")

    parser.add_argument("-m", "--mc_runs",
                        dest="mc_runs",
                        type=int,
                        default=20,
                        help="Number of Monte Carlo runs")

    parser.add_argument("-q", "--queue_max_items",
                        dest="queue_max_items",
                        type=int,
                        default=10,
                        help="Number of items in the queue")

    parser.add_argument("-s", "--seed",
                        dest="seed",
                        type=int,
                        default=1010,
                        help="random seed for Monte Carlo")

    parser.add_argument("-v", "--verbose",
                        dest="verbose",
                        action="store_true",
                        help="Verbose output if set")

    return parser.parse_args(argv)

def load_image_paths(image_queue, image_paths):
    for image_path in image_paths:
        # Add the original image to the queue:
        modif = "Original"
        ground_truth = 0
        X = get_subj_data(image_path)
        image_queue.put((image_path, X, ground_truth, modif))

        

        # Create a TorchIO subject
        subject = tio.Subject(mri=tio.ScalarImage(image_path)) 
        # Generate modified (flipped clean, or artefacted) images
        for modification_name, modification_function in modifications.items():
            modified_image = modification_function(subject)
            temp_path = image_path.rstrip(".nii.gz")+f"_{modification_name}.nii.gz"
            ground_truth = 0 if modification_name.startswith("flip") else 1
            modif = modification_name

            # Temporarly save the modified image so it can be re-loaded with nib library
            modified_image.mri.save(temp_path)

            # Re-open the modified image with nib library
            X = get_subj_data(temp_path)

            # Add the modified image along with its path to the image_queue so it can be then send to the model 
            image_queue.put((temp_path, X, ground_truth, modif))

            # Delete the temporary-stored modified image 
            os.remove(temp_path)

    image_queue.put(None)  # signal the end of the queue

def main(image_paths, csv_filepath=None, mc_runs=10, max_items=10, verbose=False):
    if verbose:
        logger.info('Loading model : {}'.format(WEIGHTS))

    image_queue = multiprocessing.Queue(max_items)
    producer = multiprocessing.Process(target=load_image_paths, args=(image_queue, image_paths))
    producer.start()

    # https://github.com/onnx/onnx/issues/3753
    sess = onnxruntime.InferenceSession(
        WEIGHTS, disabled_optimizers=["EliminateDropout"],
        #providers=["CUDAExecutionProvider"]
    )
    input_name = sess.get_inputs()[0].name

    csv_output = []

    # Creation of the output csv
    columns = ["ID", "Results", "Mean_preds", "Std_preds", "Distance_mean_preds_to_gt", "Ground_truth", "Modification done"]
    if csv_filepath:
        with open(csv_filepath, "w") as fd:
            csv_writer = csv.writer(fd)
            csv_writer.writerow(columns)

    while True:
        item = image_queue.get()
        if item is None:
            break
        image_path, X, ground_truth, modif = item

        predictions_for_img = []
        for mc in range(0, mc_runs):
            logger.info('Working on mc run {}'.format(mc))

            y = sess.run(None, {input_name: X})[0][0][1] # artefact-prob as float
            predictions_for_img.append(y)

        mean_preds = np.mean(predictions_for_img)
        std_preds = np.std(predictions_for_img)
        dist_mean_pred_to_gt = np.abs(ground_truth - mean_preds)

        if csv_filepath:
            with open(csv_filepath, "a") as fd:
                csv_writer = csv.writer(fd)
                csv_writer.writerow([image_path, predictions_for_img, mean_preds, std_preds, dist_mean_pred_to_gt, ground_truth, modif])

        csv_output.append("ID: " + str([image_path]+predictions_for_img).strip('[]') +
                          " Mean preds: " + str(mean_preds) + " Std preds: " + str(std_preds) +
                          " Dist_mean_preds_to_gt: " + str(dist_mean_pred_to_gt) +
                          " Ground_truth: "+ str(ground_truth) +
                          " Modification done: " + str(modif))
        
        print(csv_output[-1])
        


    #if csv_filepath:
       # with open(csv_filepath, "w") as fd: # open with w if overwrite, a if append
            #fd.write(os.linesep.join(csv_output) + os.linesep)
       # print("wrote results to {}".format(csv_filepath))

    producer.join()
    plot_output("output.csv")


def plot_output(csv_path):
    df = pd.read_csv(csv_path)
    plt.errorbar(df["Modification done"], df["Mean_preds"], yerr=df["Std_preds"], fmt='o', capsize=5)
    plt.xlabel("Modification done")
    plt.title("Absolute difference between ground truth and mean(predictions), with std")
    plt.ylabel("Abs(difference)")
    plt.xticks(df["Modification done"], rotation='vertical')
    plt.savefig("output_plot.png")



if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    args = parse_arguments(argv[1:])
    onnxruntime.set_seed(args.seed)

    if args.inputs_file:
        with open(args.inputs_file, 'r') as f:
            image_paths = [line.strip() for line in f.readlines()]
    else:
        image_paths = args.image_paths

    main(image_paths, args.csv_filepath, args.mc_runs, args.queue_max_items, args.verbose)
       


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

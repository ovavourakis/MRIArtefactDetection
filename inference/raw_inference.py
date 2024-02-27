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
        X = get_subj_data(image_path)
        image_queue.put((image_path, X))
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
    while True:
        item = image_queue.get()
        if item is None:
            break
        image_path, X = item

        predictions_for_img = []
        for mc in range(0, mc_runs):
            logger.info('Working on mc run {}'.format(mc))

            y = sess.run(None, {input_name: X})[0][0][1] # artefact-prob as float
            predictions_for_img.append(y)

        csv_output.append( str([image_path]+predictions_for_img).strip('[]'))
        print(csv_output[-1])

    if csv_filepath:
        with open(csv_filepath, "a") as fd:
            fd.write(os.linesep.join(csv_output) + os.linesep)
        print("wrote results to {}".format(csv_filepath))

    producer.join()


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

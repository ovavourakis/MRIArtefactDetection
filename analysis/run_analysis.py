"""
Example script to analyse raw model predictions produced by raw_inference.py.
Uses the functions defined in analysis.py.

NOTE: 
for new data load_predictions_and_ground_truth() will have to be re-implemented to produce:

ground_truth_labels of the form:
                                 id   bin_gt
0       sub-000103_acq-standard_T1w        0
1    sub-000103_acq-headmotion1_T1w        1
..                              ...      ...

where bin_gt is the binary/binarised ground truth and the id column contains image ids,
as well as

raw_model_preds of the form:
                                      1   ...            20
0                                         ...              
sub-926536_acq-headmotion1_T1w  0.433381  ...  5.049924e-01
sub-926536_acq-standard_T1w     0.003448  ...  6.611057e-02

where the columns are the raw model probability output from different MC samples
"""

from analysis_utils import *

GROUND_TRUTH = '/Users/odysseasvavourakis/Documents/2023-2024/Studium/SABS/GE Project/data.nosync/artefacts/derivatives/scores.tsv'
MODEL_PREDS = 'raw_mcmc20_out.csv'

GT2 = '/Users/odysseasvavourakis/Documents/2023-2024/Studium/SABS/GE Project/data.nosync/artefacts_2/scores.csv'
MODEL_PREDS2 = 'raw_artefacts_2_mcmc_20_out.csv'

# GE_DIR = 'GE_data'

OUTDIR_A = 'analysis_mean_class'
OUTDIR_B = 'analysis_mean_prob'

# load predictions and ground truth
ground_truth_labels, raw_model_preds = load_predictions_and_ground_truth(MODEL_PREDS, GROUND_TRUTH)
gt2, rmp2 = load_predictions_and_ground_truth_2(MODEL_PREDS2, GT2)

ground_truth_labels = pd.concat([ground_truth_labels, gt2])


# # ground_truth_labels, raw_model_preds, _, _ = load_predictions_GE(GE_DIR) # without FLAIR
# # _, _, ground_truth_labels, raw_model_preds = load_predictions_GE(GE_DIR) # with FLAIR


print('Running analysis for predictive probability := average predicted class')
run_analysis(raw_model_preds, ground_truth_labels, 
            MC=20, nbins=10, maxDFFMR=0.3, lattice_size=50, option='mean_class', OUTDIR=OUTDIR_A, init_thresh=0.5)

print('Running analysis for predictive probability := average probability')
run_analysis(raw_model_preds, ground_truth_labels, 
             MC=20, nbins=10, maxDFFMR=0.3, lattice_size=50, option='mean_prob', OUTDIR=OUTDIR_B)
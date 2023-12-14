from tqdm import tqdm
from analysis_utils import *

GROUND_TRUTH = '/Users/odysseasvavourakis/Documents/2023-2024/Studium/SABS/GE Project/data.nosync/artefacts/derivatives/scores.tsv'
MODEL_PREDS = 'raw_mcmc20_out.csv'
OUTDIR_A = 'analysis_mean_class'
OUTDIR_B = 'analysis_mean_prob'

# load predictions and ground truth
ground_truth_labels, raw_model_preds = load_predictions_and_ground_truth(MODEL_PREDS, GROUND_TRUTH)

print('Running analysis for predictive probability := average predicted class')
run_ternary_analysis(raw_model_preds, ground_truth_labels, MC=20, nbins=10,  lattice_size=10, 
                    option='mean_class', OUTDIR=OUTDIR_A, init_thresh=0.5, 
                    max_clean_impurity=0.0, min_dirty_impurity=0.95)

print('Running analysis for predictive probability := average probability')
run_ternary_analysis(raw_model_preds, ground_truth_labels, MC=20, nbins=10, lattice_size=10, 
                     option='mean_prob', OUTDIR=OUTDIR_B, 
                     max_clean_impurity=0.0, min_dirty_impurity=0.95)
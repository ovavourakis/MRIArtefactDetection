from tqdm import tqdm
from analysis_utils import *

GROUND_TRUTH = '/Users/odysseasvavourakis/Documents/2023-2024/Studium/SABS/GE Project/data.nosync/artefacts/derivatives/scores.tsv'
MODEL_PREDS = 'raw_mcmc20_out.csv'

GT2 = '/Users/odysseasvavourakis/Documents/2023-2024/Studium/SABS/GE Project/data.nosync/artefacts_2/scores.csv'
MODEL_PREDS2 = 'raw_artefacts2_mcmc_20_out.csv'

GT3 = '/Users/odysseasvavourakis/Documents/2023-2024/Studium/SABS/GE Project/data.nosync/artefacts_3/scores.csv'
MODEL_PREDS3 = 'raw_data_artefacts3_NOpmc_mcmc20_out.csv'

# GE_DIR = 'GE_data'
OUTDIR_A = '3ary_analysis_max_class_all_3_datasets'
OUTDIR_B = '3ary_analysis_max_prob_all_3_datsets'

# load predictions and ground truth
ground_truth_labels, raw_model_preds = load_predictions_and_ground_truth(MODEL_PREDS, GROUND_TRUTH)
gt2, rmp2 = load_predictions_and_ground_truth_2(MODEL_PREDS2, GT2)
gt3, rmp3 = load_predictions_and_ground_truth_3(MODEL_PREDS3, GT3)

ground_truth_labels = pd.concat([ground_truth_labels, gt2, gt3])
raw_model_preds = pd.concat([raw_model_preds, rmp2, rmp3])

# ground_truth_labels, raw_model_preds, _, _ = load_predictions_GE(GE_DIR) # without FLAIR
# _, _, ground_truth_labels, raw_model_preds = load_predictions_GE(GE_DIR) # with FLAIR

print('Running analysis for predictive probability := average predicted class')
run_ternary_analysis(raw_model_preds, ground_truth_labels, MC=20, nbins=10,  lattice_size=10, 
                    option='mean_class', OUTDIR=OUTDIR_A, init_thresh=0.5, 
                    max_clean_impurity=0.0, min_dirty_impurity=0.95)

print('Running analysis for predictive probability := average probability')
run_ternary_analysis(raw_model_preds, ground_truth_labels, MC=20, nbins=10, lattice_size=10, 
                     option='mean_prob', OUTDIR=OUTDIR_B, 
                     max_clean_impurity=0.0, min_dirty_impurity=0.95)
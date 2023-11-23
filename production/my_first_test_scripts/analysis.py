import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, roc_curve, precision_recall_curve
from sklearn.calibration import calibration_curve, CalibrationDisplay

GROUND_TRUTH = '/Users/odysseasvavourakis/Documents/2023-2024/Studium/SABS/GE Project/data.nosync/artefacts/derivatives/scores.tsv'

# correct the clean-probs for these
# MODEL_PREDS = '/Users/odysseasvavourakis/Documents/2023-2024/Studium/SABS/GE Project/codebase/pizarro/production/my_test_scripts/default_results.csv'
# MODEL_PREDS = '/Users/odysseasvavourakis/Documents/2023-2024/Studium/SABS/GE Project/codebase/pizarro/production/my_test_scripts/mcmc20_results.csv'

# don't correct clean-probs for this one
MODEL_PREDS = '/Users/odysseasvavourakis/Documents/2023-2024/Studium/SABS/GE Project/codebase/pizarro/production/my_test_scripts/new_collate_results.csv'

ground_truth_labels = pd.read_csv(GROUND_TRUTH, sep='\t')
ground_truth_labels['bin_gt'] = ground_truth_labels['score'].replace({1:0, 2:1, 3:1})
ground_truth_labels.rename(columns={'bids_name': 'id', 'score':'gt_score'}, inplace=True)

model_preds = pd.read_csv(MODEL_PREDS, sep=',')
model_preds['probability'] = model_preds['probability']/100
model_preds['image_path'] = model_preds['image_path'].str.split('/').str[-1].str.split('.').str[0]

# correct the clean-probs here
# model_preds.loc[model_preds['inferred_class'] == 'clean', 'probability'] = 1 - model_preds.loc[model_preds['inferred_class'] == 'clean', 'probability']

model_preds.rename(columns={'image_path': 'id', 'probability':'artfct_prob'}, inplace=True)

merged_df = pd.merge(model_preds, ground_truth_labels, on=['id'])


auroc = roc_auc_score(merged_df['bin_gt'], merged_df['artfct_prob'])
ap = average_precision_score(merged_df['bin_gt'], merged_df['artfct_prob'])

print('AUROC: {}'.format(auroc))
print('AP: {}'.format(ap))

fpr, tpr, thresholds = roc_curve(merged_df['bin_gt'], merged_df['artfct_prob'])
precision, recall, thresholds = precision_recall_curve(merged_df['bin_gt'], merged_df['artfct_prob'])

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.text(0, 1, 'AUROC: {:.3f}'.format(auroc), bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2'))

plt.subplot(1, 3, 2)
plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.text(0.8,1, 'AP: {:.3f}'.format(ap), bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2'))

# plt.subplot(1, 3, 3)
# plt.plot(thresholds, precision[:-1])
# plt.xlabel('Threshold')
# plt.ylabel('Precision')
# plt.title('Precision over Prediction Threshold')

# plt.subplot(2, 3, 4)
# plt.plot(thresholds, recall[:-1])
# plt.xlabel('Threshold')
# plt.ylabel('Recall')
# plt.title('Recall over Prediction Threshold')

plt.subplot(1, 3, 3)
thresholds = np.arange(0, 1.01, 0.01)

# f1_scores = [f1_score(merged_df['bin_gt'], merged_df['artfct_prob'] >= threshold) for threshold in thresholds]
# plt.plot(thresholds, f1_scores)
# plt.xlabel('Threshold')
# plt.ylabel('F1-Score')
# plt.title('F1-Score over Prediction Threshold')

accuracy = [np.mean(merged_df['bin_gt'] == (merged_df['artfct_prob'] >= threshold)) for threshold in thresholds]

# TODO: add an hvline for this
all_ones_acc = max([np.mean(merged_df['bin_gt'] == (np.ones(len(merged_df['bin_gt'])) >= threshold)) for threshold in thresholds])

plt.plot(thresholds, accuracy)
plt.xlabel('Threshold')
plt.ylabel('Accuracy')
plt.title('Accuracy over Prediction Threshold')

plt.show()

# plot a histogram of the model predicitons (artfct_prob)
plt.figure(figsize=(5,5))
plt.hist(merged_df['artfct_prob'], bins=50, density=True)
plt.xlabel('Predicted Artefact Probability')
plt.ylabel('Relative Frequency')
plt.show()

# plot a calibration curve
prob_true, prob_pred = calibration_curve(merged_df['bin_gt'], merged_df['artfct_prob'], n_bins=10)
disp = CalibrationDisplay(prob_true, prob_pred, merged_df['artfct_prob'])
disp.plot()
plt.title('Calibration Curve')
plt.show()


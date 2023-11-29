import os, pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
from tqdm import tqdm

def load_predictions_and_ground_truth(MODEL_PREDS, GROUND_TRUTH):
    # read ground-truth and binarise to 0: clean, 1: artefact
    ground_truth_labels = pd.read_csv(GROUND_TRUTH, sep='\t')
    ground_truth_labels['bin_gt'] = ground_truth_labels['score'].replace({1:0, 2:1, 3:1})
    ground_truth_labels.rename(columns={'bids_name': 'id', 'score':'gt_score'}, inplace=True)
    # read model predictions
    model_preds = pd.read_csv(MODEL_PREDS, sep=',', header=None)
    model_preds.loc[:,0] = model_preds.loc[:,0].str.split('/').str[-1].str.split('.').str[0]
    model_preds.set_index(0, inplace=True)

    return ground_truth_labels, model_preds

def assign_class(raw_prob, thresh=0.5):
    return 1 if raw_prob >= thresh else 0

def compute_mean_class_and_uncertainty(raw_model_preds, num_mc_runs=20, thresh=0.5):
    # this is for option (A): predictive probability = average predicted class
    model_assigned_classes = raw_model_preds.iloc[:,:num_mc_runs].map(lambda x: assign_class(x, thresh=thresh))
    return model_assigned_classes.mean(axis=1), model_assigned_classes.std(axis=1)

def compute_mean_prob_and_uncertainty(raw_model_preds, num_mc_runs=20):
    # this is for option (B): predictive probability = average predictive probability
    return raw_model_preds.iloc[:,:num_mc_runs].mean(axis=1), raw_model_preds.iloc[:,:num_mc_runs].std(axis=1)

def plot_mean_class_histograms(mean, std, num_mc_runs, nbins, outdir):
    # plot histogram of mean-class predictions
    plt.hist(mean, bins=nbins)
    plt.xlabel('Mean Class Prediction')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Mean Class Predictions for MC={num_mc_runs}')
    plt.savefig(outdir+f'/hist_mean_class_mc_{num_mc_runs}.png')
    plt.clf()

    # plot histogram of uncertainty in mean_class_predictions
    plt.hist(std, bins=nbins)
    plt.xlabel('Standard Deviation of Mean Class Prediction')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Mean Class Prediction Uncertainties for MC={num_mc_runs}')
    plt.savefig(outdir+f'/hist_std_mean_class_mc_{num_mc_runs}.png')
    plt.clf()

def plot_predictive_undertainty_per_bin(mean, std, num_mc_runs, nbins, outdir):
    # bin the mean-class predictions and calc. average std_dev per bin
    mean_bins = pd.cut(mean, bins=np.linspace(0, 1, nbins+1), include_lowest=True)
    average_std_per_bin = pd.concat([mean, mean_bins, std], axis=1).groupby(1, observed=False).agg({2: 'mean'})
    std_of_mean_std_per_bin = pd.concat([mean, mean_bins, std], axis=1).groupby(1, observed=False).agg({2: 'std'})
    # plot the average std_dev against the bin identifiers
    x = np.linspace(0, 10, nbins)*10
    plt.plot(x, average_std_per_bin)
    plt.fill_between(x, (average_std_per_bin - std_of_mean_std_per_bin).squeeze(), 
                        (average_std_per_bin + std_of_mean_std_per_bin).squeeze(), alpha=0.3)
    plt.ylim(0, 0.55)
    plt.xlabel('Mean Class Prediction [Percentile]')
    plt.ylabel('Average Prediction StdDev in Percentile')
    plt.title(f'Average StdDev of Mean Class Prediction for MC={num_mc_runs}')
    plt.savefig(outdir+f'/line_std_per_bin_mc_{num_mc_runs}.png')
    plt.clf()

def plot_calibration_plot(mean_preds, gt, num_mc_runs, nbins, outdir):
    mean_preds_bins = pd.cut(mean_preds, bins=np.linspace(0, 1, nbins+1), include_lowest=True)    
    preds_and_bins = pd.concat([mean_preds, mean_preds_bins], axis=1)
    merged = pd.merge(preds_and_bins,  gt.set_index('id')['bin_gt'], 
                        left_index=True, right_index=True)
    merged.columns = ['mean_pred', 'bin_pred', 'gt']
    average_gt_per_bin = merged.groupby('bin_pred', observed=False).agg({'gt': 'mean'})*100
    # plot the average ground_truth against the bin identifiers
    x = np.linspace(0, 10, nbins)*10
    plt.plot(x, average_gt_per_bin)
    # plot the identity line
    plt.plot([0, 100], [0, 100], 'k--')
    # add labels
    plt.xlabel('Mean Class Prediction [Percentile]')
    plt.ylabel('Ground-Truth Positive Class Frequency in Percentile')
    plt.title(f'Calibration Plot for MC={num_mc_runs}')
    plt.savefig(outdir+f'/calibration_plot_mc_{num_mc_runs}.png')
    plt.clf()

def merge_predictions_and_gt(mean, std, ground_truth_labels):
    # re-scale std to lie between 0 and 1
    scaled_std = (std - np.min(std)) / (np.max(std) - np.min(std))
    # merge mean-class predictions and uncertainties with ground-truth labels (left join)
    mean_and_std = pd.DataFrame([mean, std, scaled_std]).transpose()
    mean_and_std.columns = ['mean_pred', 'std_pred', 'scaled_std_pred']
    merged = pd.merge(mean_and_std, ground_truth_labels.set_index('id')['bin_gt'], 
                        left_index=True, right_index=True)
    return merged

def calculate_DFFMR_AP_AUROC(merged, eta):
    # fraction of Dataset Flagged For Manual Review (DFFMR)
    num_images = merged.shape[0]
    num_discarded = sum(merged['scaled_std_pred'] >= eta)
    DFFMR = num_discarded/num_images

    # average precision (AP) on the retained set
    if sum(merged['scaled_std_pred'] < eta) == 0:
        AP, AUC = np.nan, np.nan
    else:
        retained = merged[merged['scaled_std_pred'] < eta]   
        AP = average_precision_score(retained['bin_gt'], retained['mean_pred'])
        # area under the roc curve (AUC) on the retained set
        AUC = roc_auc_score(retained['bin_gt'], retained['mean_pred'])

    return DFFMR, AP, AUC

def plot_DFFMR_AP_AUROC(merged, num_mc_runs, maxDFFMR, OUTDIR):
    etas = np.linspace(0, 1, 100)
    DFFMR_AP = pd.DataFrame([calculate_DFFMR_AP_AUROC(merged, eta) for eta in etas])
    DFFMR_AP.columns = ['DFFMR', 'AP', 'AUROC']
    DFFMR_AP.index = etas
    DFFMR_AP.plot()

    plt.axhline(y=maxDFFMR, color='r', linestyle='--')  # horizontal line
    crossing_point = DFFMR_AP[DFFMR_AP['DFFMR'] >= maxDFFMR].index[-1]
    plt.axvline(x=crossing_point, color='r', linestyle='--')  # vertical line

    # Shade everything to the left of the vertical line in gray
    plt.fill_between(DFFMR_AP.index, 0, 1, where=DFFMR_AP.index <= crossing_point, color='gray', alpha=0.3)

    plt.xlabel('Uncertainty Threshold eta')
    plt.ylabel('Metric Value')
    plt.ylim(0, 1)
    plt.title('Pre-Screening Performance')
    plt.savefig(OUTDIR+f'/eta_preescreening_mc_{num_mc_runs}.png')
    plt.clf()

def pointwise_metrics(retained, theta):
    binarised_preds = retained['mean_pred'] > theta
    tn, fp, fn, tp = confusion_matrix(retained['bin_gt'], binarised_preds).ravel()
    accuracy = (tp+tn)/(tp+tn+fp+fn) if (tp+tn+fp+fn > 0) else np.nan
    specificity = tn / (tn+fp) if (tn+fp > 0) else np.nan
    F1 = f1_score(retained['bin_gt'], binarised_preds, zero_division=np.nan)
    precision = precision_score(retained['bin_gt'], binarised_preds, zero_division=np.nan)
    recall = recall_score(retained['bin_gt'], binarised_preds, zero_division=np.nan)
    return F1, precision, recall, accuracy, specificity

def calculate_gridpoint_metrics(merged, eta, theta):
    # fraction of Dataset Flagged For Manual Review (DFFMR)
    num_images = merged.shape[0]
    num_discarded = sum(merged['scaled_std_pred'] >= eta)
    DFFMR = num_discarded/num_images

    if DFFMR == 1.0:
        return np.nan, DFFMR, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    else: 
        # unusable dataset missed (UDM)
        retained = merged[merged['scaled_std_pred'] < eta]
        retained_labeled_clean = retained[retained['mean_pred'] < theta]
        FN = retained_labeled_clean['bin_gt'].sum()
        num_labelled_clean = retained_labeled_clean.shape[0]
        UDM = FN/num_labelled_clean if num_labelled_clean > 0 else np.nan

        # usable dataset discarded (UDD)
        all_discarded = merged[(merged['scaled_std_pred'] >= eta) | (merged['mean_pred'] >= theta)]
        UDD = all_discarded['bin_gt'].sum()/merged['bin_gt'].sum()

        # F1, precision, recall for the retained set
        F1, precision, recall, accuracy, specificity = pointwise_metrics(retained, theta)

        scores = [DFFMR, UDM, 1-UDD, 1-F1, 1-specificity]
        combined_score = sum(scores)
        
        return combined_score, DFFMR, UDM, F1, precision, recall, accuracy, specificity, UDD

def gridpoint_metrics_tensor(merged, maxDFFMR=0.3, lattice_size=50):
    etas = np.linspace(0, 1, lattice_size)
    thetas = np.linspace(0, 1, lattice_size)
    print('Running over eta grid:')
    gridpoint_metrics = pd.DataFrame([calculate_gridpoint_metrics(merged, eta, theta) 
                                        for eta in tqdm(etas) for theta in thetas])
    gridpoint_metrics.columns = ['combined', 'DFFMR', 'UDM', 'F1', 'precision', 'recall', 'accuracy', 'specificity', 'UDD']
    gridpoint_metrics.index = pd.MultiIndex.from_product([etas, thetas], names=['eta', 'theta'])

    # print the best gridpoint (in terms of the combined score)
    optimisation_tensor = gridpoint_metrics[gridpoint_metrics['DFFMR'] < maxDFFMR]
    print(optimisation_tensor.iloc[[optimisation_tensor['combined'].argmin()]])
    
    return gridpoint_metrics

def plot_gridpoint_metrics(gpm_tensor, maxDFFMR, num_mc_runs, OUTDIR):
    # os.makedirs(OUTDIR+'/grids', exist_ok=True)
    gridpoint_metrics = gpm_tensor[gpm_tensor['DFFMR'] < maxDFFMR].reset_index()

    for metric in gridpoint_metrics.columns[2:]:
        gpm = gridpoint_metrics.pivot(index='eta', columns='theta', values=metric)
        gpm = gpm.sort_index(ascending=False, axis=1).sort_index(ascending=True, axis=0)
        plt.figure(figsize=(10, 10))
        # round the axis ticks
        yticks = [f'{y:.2f}' for y in gpm.index]
        xticks = [f'{x:.2f}' for x in gpm.columns]
        # plot a 2D heatmap
        sns.heatmap(gpm, cmap='coolwarm', annot=False, fmt=".2f", linewidths=0.5,
                         xticklabels=xticks, yticklabels=yticks)
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        # add labels
        plt.xlabel('Decision Threshold theta')
        plt.ylabel('Uncertainty Threshold eta')
        plt.title(f'{metric} for MC={num_mc_runs}')
        plt.savefig(OUTDIR+f'gridpoint_{metric}_mc_{num_mc_runs}.png')
        plt.close()

def one_stage_screening(merged, OUTDIR='analysis_A_out'):
    # single-number stats
    DFFMR, AP, AUC = calculate_DFFMR_AP_AUROC(merged, eta=1.1) # don't exclude any images
    print(DFFMR)
    # threshold-dependent stats
    thetas = np.linspace(0, 1, 100)
    pwm = pd.DataFrame([calculate_gridpoint_metrics(merged, 1.1, theta) for theta in thetas])
    pwm.columns = ['combined', 'DFFMR', 'UDM (FN)', 'F1', 'precision', 'recall', 'accuracy', 'specificity', 'UDD (FP)']
    pwm['theta'] = thetas

    # plot the threshold-dependent stats over theta
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    # plot 'combined' score in the first subplot
    ax1.plot(pwm['theta'], pwm['combined'])
    ax1.set_ylabel('Combined Score')
    # add single-number stats to the first subplot
    ax1.text(0.05, 0.85, f'DFFMR: {DFFMR:.2f}\nAP: {AP:.2f}\nAUROC: {AUC:.2f}', transform=ax1.transAxes,
             bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
    # plot other metrics in the second subplot
    ax2.plot(pwm['theta'], pwm['UDM (FN)'], label='UDM (FN)')
    ax2.plot(pwm['theta'], pwm['UDD (FP)'], label='UDD (FP)')
    ax2.plot(pwm['theta'], pwm['F1'], label='F1')
    ax2.plot(pwm['theta'], pwm['accuracy'], label='Accuracy')
    ax2.plot(pwm['theta'], pwm['specificity'], label='Specificity')
    ax2.set_xlabel('Probability Threshold theta')
    ax2.set_ylabel('Metric Value')
    ax2.legend()

    plt.suptitle('Metrics for One-Stage Screening (All Data Included)')
    plt.tight_layout()
    plt.savefig(OUTDIR+f'/one_stage_screening.png')
    plt.clf()

def two_stage_screening(merged, MC=20, maxDFFMR=0.3, lattice_size=50, OUTDIR='analysis_A_out'):
    # eta-dependent stats
    plot_DFFMR_AP_AUROC(merged, MC, maxDFFMR, OUTDIR)
    # eta-and-theta-dependent stats
    gpm_tensor = gridpoint_metrics_tensor(merged, maxDFFMR=maxDFFMR, lattice_size=lattice_size)
    plot_gridpoint_metrics(gpm_tensor, maxDFFMR, MC, OUTDIR+'/two_stage_screening')

def run_analysis(raw_model_preds, ground_truth_labels, MC=20, nbins=10, maxDFFMR=0.3, lattice_size=50, option='mean_class', OUTDIR='analysis_A_out', init_thresh=0.5):
    # setup
    os.makedirs(OUTDIR+'/one_stage_screening', exist_ok=True)
    os.makedirs(OUTDIR+'/two_stage_screening', exist_ok=True)

    # analysis
    if option == 'mean_class':  # note: the init_thresh kwarg only affects option (A)
        mean, std = compute_mean_class_and_uncertainty(raw_model_preds, MC, thresh=init_thresh)
    elif option == 'mean_prob':
        mean, std = compute_mean_prob_and_uncertainty(raw_model_preds, MC)
    else:
        raise ValueError('analysis option must be either A or B')

    plot_mean_class_histograms(mean, std, MC, nbins, OUTDIR)
    plot_predictive_undertainty_per_bin(mean, std, MC, nbins, OUTDIR)
    plot_calibration_plot(mean, ground_truth_labels, MC, nbins, OUTDIR)

    merged = merge_predictions_and_gt(mean, std, ground_truth_labels)
    # one-stage screening (probability threshold only)
    one_stage_screening(merged, OUTDIR=OUTDIR+'/one_stage_screening')
    # two-stage screening (uncertainty and probability thresholds)
    two_stage_screening(merged, MC, maxDFFMR, lattice_size, OUTDIR=OUTDIR+'/two_stage_screening')

# BEGIN MAIN ====================================================================================

GROUND_TRUTH = '/Users/odysseasvavourakis/Documents/2023-2024/Studium/SABS/GE Project/data.nosync/artefacts/derivatives/scores.tsv'
MODEL_PREDS = 'raw_mcmc20_out.csv'
OUTDIR_A = 'analysis_mean_class'
OUTDIR_B = 'analysis_mean_prob'

ground_truth_labels, raw_model_preds = load_predictions_and_ground_truth(MODEL_PREDS, GROUND_TRUTH)
# option (A): predictive probability = average predicted class
run_analysis(raw_model_preds, ground_truth_labels, 
            MC=20, nbins=10, maxDFFMR=0.3, lattice_size=50, option='mean_class', OUTDIR=OUTDIR_A, init_thresh=0.5)
# option (B): predictive probability = average predictive probability
run_analysis(raw_model_preds, ground_truth_labels, 
             MC=20, nbins=10, maxDFFMR=0.3, lattice_size=50, option='mean_prob', OUTDIR=OUTDIR_B)
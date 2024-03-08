import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.metrics import AUC
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split, GroupShuffleSplit

from train_utils import DataCrawler, DataLoader, plot_train_metrics, split_by_patient, PrintModelWeightsNorm
from model import getConvNet

# Check for GPU availability and set TensorFlow to use GPU if available
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("Using GPU\n")
else:
    print("Using CPU\n")

SAVEDIR = './trainrun/' # make sure this has a trailing slash
DATADIR = '/Users/odysseasvavourakis/Documents/2023-2024/Studium/SABS/GE Project/data.nosync/struc_data'
DATASETS = ['artefacts'+str(i) for i in [1,2,3]]
CONTRASTS = ['T1wMPR']#, 'T1wTIR', 'T2w', 'T2starw', 'FLAIR']
QUALS = ['clean', 'exp_artefacts']

ARTEFACT_DISTRO = {
    'RandomZoom' : 1/15,
    #'RandomElasticDeformation' : ,     # not defined
    'RandomAnisotropy' : 1/15, 
    'Intensity' : 1/15, 
    'RandomMotion' : 2/15,              # more common
    'RandomGhosting' : 2/15,            # more common
    'RandomSpike' : 1/15,  
    'RandomBiasField' : 2/15,           # more common
    'RandomBlur' : 1/15, 
    #'RandomNoise' : ,                  # not defined
    'RandomSwap' : 1/15, 
    'RandomGamma' : 1/15,
    'RandomWrapAround' : 2/15,          # more common
}

MC_RUNS = 20  # number of Monte Carlo runs on test set

if __name__ == '__main__':
    
    os.makedirs(SAVEDIR, exist_ok=True)

    # get the paths and labels of the real images
    real_image_paths, pids, real_labels = DataCrawler(DATADIR, DATASETS, CONTRASTS, QUALS).crawl()

    # split by patient
    Xtrain, Xval, Xtest, ytrain, yval, ytest = split_by_patient(real_image_paths, pids, real_labels)

    # # train-val-test split (stratified to preserve class prevalence)
    # Xtrv, Xtest, ytrv, ytest = train_test_split(real_image_paths, real_labels, test_size=0.1, stratify=real_labels)
    # Xtrain, Xval, ytrain, yval = train_test_split(Xtrv, ytrv, test_size=0.2222, stratify=ytrv)

    for string, y in zip(['train', 'val', 'test'], [ytrain, yval, ytest]):
        print('number in ' + string + ' set:', len(y))
        print(string + ' class distribution: ', sum(y)/len(y), ' percent artefact')

    # instantiate DataLoaders
    trainloader = DataLoader(Xtrain, ytrain, train_mode=True, target_clean_ratio=0.5,
                                artef_distro=ARTEFACT_DISTRO, batch_size=4)
    valloader = DataLoader(Xval, yval, train_mode=False, batch_size=50)
    testloader = DataLoader(Xtest*MC_RUNS, np.array(ytest.tolist()*MC_RUNS), 
                            train_mode=False, batch_size=50)

    # write out ground truth for test set
    test_images = [file for sublist in testloader.batches for file in sublist]
    y_true_test = [y for sublist in testloader.labels for y in sublist]
    out = pd.DataFrame({'image': test_images, 'bin_gt': y_true_test}).groupby('image').agg({'bin_gt': 'first'})
    out.to_csv(SAVEDIR+'test_split_gt.tsv', sep='\t')

    # compile model
    model = getConvNet(out_classes=2, input_shape=(256,256,64,1))
    model.compile(loss='categorical_crossentropy',
                optimizer='nadam', 
                metrics=['accuracy', 
                        AUC(curve='ROC', name='auroc'), 
                        AUC(curve='PR', name='auprc')])
    print(model.summary())

    # prepare for training
    checkpoint_dir = SAVEDIR+"ckpts"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # define callbacks
    callbacks = [
        # keep an eye on model weights norm after every epoch
        PrintModelWeightsNorm(),
        # save model after each epoch
        ModelCheckpoint(filepath=checkpoint_dir + "/end_of_epoch_{epoch}.keras"), 
        # reduce learning rate if val_loss doesn't improve for 2 epochs
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, mode='auto',
                          min_delta=1e-2, cooldown=0, min_lr=0.0001),
        # stop training if val_loss doesn't improve for 5 epochs
        EarlyStopping(monitor="val_loss", min_delta=1e-2, patience=5, verbose=1)
    ]

    # train model
    history = model.fit(trainloader, 
                        validation_data=valloader,
                        steps_per_epoch=26,        # 26 batches per epoch
                        epochs=24,                 # 24 epochs = 624 gradient updates
                        callbacks=callbacks,
                        use_multiprocessing=True,
                        workers=2)
    
    # save metrics
    pd.DataFrame({
        'train_loss': history.history['loss'],
        'val_loss': history.history['val_loss'],
        'train_accuracy': history.history['accuracy'],
        'val_accuracy': history.history['val_accuracy'],
        'train_auc': history.history['auc'],
        'val_auc': history.history['val_auc'],
        'train_ap': history.history['auc_1'],
        'val_ap': history.history['val_auc_1']
    }).to_csv(SAVEDIR+'training_metrics.tsv', sep='\t')
    plot_train_metrics(history, SAVEDIR+'train_metrics_plot.png')

    # evaluate model (20 MC runs per test image)
    print("#"*30)
    print("Evaluate on Test Set")
    print("#"*30)
    # predict on test set, and write out results
    y_pred = model.predict(testloader, use_multiprocessing=True)
    df = pd.DataFrame({'image': test_images, 'bin_gt': y_true_test, 'y_pred': y_pred[:,1]})
    df = df.groupby('image').agg({'bin_gt': 'first', 'y_pred': list})
    df[[f'y_pred{i}' for i in range(MC_RUNS)]] = pd.DataFrame(df['y_pred'].tolist(), index=df.index)
    df = df.drop(columns='y_pred')
    df.to_csv(SAVEDIR+'raw_preds_test.tsv', sep='\t')

    # calculate AUROC, AP on each MC run individually
    aurocs, aps = [], []
    for i in range(MC_RUNS):
        aurocs.append(AUC(curve='ROC')(df['bin_gt'], df[f'y_pred{i}']))
        aps.append(AUC(curve='PR')(df['bin_gt'], df[f'y_pred{i}']))
    print('mean AUROC on test:', np.mean(aurocs))
    print('mean AP on test:', np.mean(aps))
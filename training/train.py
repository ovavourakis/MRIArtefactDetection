import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.metrics import AUC
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

from train_utils import DataCrawler, DataLoader, FullLossHistory
from model import getConvNet

# Check for GPU availability and set TensorFlow to use GPU if available
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("Using GPU")
else:
    print("Using CPU")

DATADIR = '/Users/odysseasvavourakis/Documents/2023-2024/Studium/SABS/GE Project/data.nosync/struc_data'
DATASETS = ['artefacts'+str(i) for i in [1,2,3]]
CONTRASTS = ['T1wMPR'] #, 'T1wTIR', 'T2w', 'T2starw', 'FLAIR']
QUALS = ['clean', 'exp_artefacts']

ARTEFACT_DISTRO = {
    'RandomAffine' : 1/12, 
    'RandomElasticDeformation' : 1/12, 
    'RandomAnisotropy' : 1/12, 
    'Intensity' : 1/12, 
    'RandomMotion' : 1/12, 
    'RandomGhosting' : 1/12, 
    'RandomSpike' : 1/12, 
    'RandomBiasField' : 1/12, 
    'RandomBlur' : 1/12, 
    'RandomNoise' : 1/12, 
    'RandomSwap' : 1/12, 
    'RandomGamma' : 1/12
}

if __name__ == '__main__':
    # get the paths and labels of the real images
    real_image_paths, real_labels = DataCrawler(DATADIR, DATASETS, CONTRASTS, QUALS).crawl()

    # train-val-test split (stratified to preserve class prevalence)
    Xtrv, Xtest, ytrv, ytest = train_test_split(real_image_paths, real_labels, test_size=0.3, stratify=real_labels)
    Xtrain, Xval, ytrain, yval = train_test_split(Xtrv, ytrv, test_size=0.3, stratify=ytrv)
    # write out real-image-paths in test set and labels to file for full evaluation later
    with open('test_split_gt.tsv', 'w') as f:
        for i, path in enumerate(Xtest):
            f.write(path + '\t' + str(ytest[i]) + '\n')

    # instantiate DataLoaders
    trainloader = DataLoader(Xtrain, ytrain, 
                            target_clean_ratio=0.5, artef_distro=ARTEFACT_DISTRO, batch_size=10)
    valloader = DataLoader(Xval, yval,
                            target_clean_ratio=0.5, artef_distro=ARTEFACT_DISTRO, batch_size=10)
    testloader = DataLoader(Xtest, ytest,
                            target_clean_ratio=0.5, artef_distro=ARTEFACT_DISTRO, batch_size=10)

    # compile model
    model = getConvNet(out_classes=2, input_shape=(256,256,64,1))
    model.compile(loss='categorical_crossentropy',
                  optimizer='nadam', 
                  metrics=['accuracy', AUC(curve='ROC'), AUC(curve='PR')])
    print(model.summary())

    # prepare for training
    # make directory to store model checkpoints
    checkpoint_dir = "./ckpts"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # define callbacks
    full_loss_history = FullLossHistory()
    callbacks = [
        # save model after each epoch
        ModelCheckpoint(filepath=checkpoint_dir + "/epoch_{epoch}.keras"), 
        # save model after 100 images, if train AUPRC improves   
        ModelCheckpoint(filepath=checkpoint_dir + "/batch_{auc_1}.keras",          
                        save_best_only=True, monitor='auc_1', save_freq=100),
        # save training loss for each batch
        full_loss_history,
        # adaptive learning rate at end of epoch (monitoring val_loss)
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, mode='auto',
                          min_delta=0.0001, cooldown=0, min_lr=0.0001)
    ]

    # train model
    # TODO: something is wrong with the validation data, probably incorrect usage, it shouldn't be exactly 0.5
    history = model.fit(trainloader, 
                        validation_data=valloader,
                        epochs=3,
                        callbacks=callbacks,
                        use_multiprocessing=True,
                        workers=2)
    training_losses = full_loss_history.per_batch_losses
    validation_losses = history.history['val_loss']

    # write out train and validation losses
    with open('train_losses.tsv', 'w') as f:
        for i, loss in enumerate(training_losses):
            f.write(str(i) + '\t' + str(loss) + '\n')
    with open('val_losses.tsv', 'w') as f:
        for i, loss in enumerate(validation_losses):
            f.write(str(i) + '\t' + str(loss) + '\n')
    # plot train losses
    plt.plot(training_losses)
    plt.xlabel('Batch')
    plt.ylabel('Training Loss')
    plt.savefig('training_losses.png')

    # evaluate model
    print("#"*30)
    print("Evaluate on Test Set")
    print("#"*30)
    result = model.evaluate(testloader)
    dict(zip(model.metrics_names, result))
    

# TODO:
# define distribution over artefact types
# split train-val-test by patient, not by image!

# check implementation with small toy model that is quick to train

# tinker with the model and input shapes - consider what's best for arbitrary datasets
# modidy image reader to coerce the input shape a different way

# train faster on GPU
# implement proper training loop
#      * train first on small batches with 50/50, then gradually increase batch size and clean ratio
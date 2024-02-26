import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.metrics import AUC
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split

from train_utils import DataCrawler, DataLoader
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

    print('train artefact percentage: ', sum(ytrain)/len(ytrain))
    print('val artefact percentage: ', sum(yval)/len(yval))
    print('test artefact percentage: ', sum(ytest)/len(ytest))

    # write out real-image-paths in test set and labels to file for full evaluation later
    with open('test_split_gt.tsv', 'w') as f:
        for i, path in enumerate(Xtest):
            f.write(path + '\t' + str(ytest[i]) + '\n')

    # instantiate DataLoaders
    trainloader = DataLoader(Xtrain, ytrain, train_mode=True,
                            target_clean_ratio=0.5, artef_distro=ARTEFACT_DISTRO, batch_size=4)
    valloader = DataLoader(Xval, yval, train_mode=False,
                            target_clean_ratio=0.5, artef_distro=ARTEFACT_DISTRO, batch_size=4)
    testloader = DataLoader(Xtest, ytest, train_mode=False,
                            target_clean_ratio=0.5, artef_distro=ARTEFACT_DISTRO, batch_size=4)

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
    callbacks = [
        # save model after each epoch
        ModelCheckpoint(filepath=checkpoint_dir + "/end_of_epoch_{epoch}.keras"), 
        # reduce learning rate if val_loss doesn't improve for 2 epochs
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, mode='auto',
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
    training_losses = history.history['loss']
    validation_losses = history.history['val_loss']

    # write out train and validation losses
    with open('train_losses.tsv', 'w') as f:
        for i, loss in enumerate(training_losses):
            f.write(str(i) + '\t' + str(loss) + '\n')
    with open('val_losses.tsv', 'w') as f:
        for i, loss in enumerate(validation_losses):
            f.write(str(i) + '\t' + str(loss) + '\n')
    # plot train losses
    plt.plot(training_losses, label='Training Loss')
    plt.plot(validation_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.savefig('training_losses.png')

    # evaluate model
    print("#"*30)
    print("Evaluate on Test Set")
    print("#"*30)
    result = model.evaluate(testloader)
    dict(zip(model.metrics_names, result))
    

# TODO:

# splitting by patient
# input size - rasampling or sub-volume sampling or registration
# make sure you save everything during training (straight to GDrive)
# tack on probabilsitic inference run on test at end of training
# evaluate probabilistic inference
# gradually shift training distribution to real distribution
# get set up on arc

# define artefact distribution
# code wrap-around and affine transformation for field gradient ingomoegenity
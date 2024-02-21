from sklearn.model_selection import train_test_split

from train_utils import DataCrawler, DataLoader
from model import getConvNet

DATADIR = '/Users/odysseasvavourakis/Documents/2023-2024/Studium/SABS/GE Project/data.nosync/struc_data'
DATASETS = ['artefacts'+str(i) for i in [1,2,3]]
CONTRASTS = ['T1wMPR'] #, 'T1wTIR', 'T2w', 'T2starw', 'FLAIR']
QUALS = ['clean', 'exp_artefacts']

ARTEFACT_DISTRO = {
    # TODO: define a categorical probability distribution over these
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

    # train-val-test split
    Xtrv, Xtest, ytrv, ytest = train_test_split(real_image_paths, real_labels, test_size=0.3, stratify=real_labels)
    Xtrain, Xval, ytrain, yval = train_test_split(Xtrv, ytrv, test_size=0.3, stratify=ytrv)

    # instantiate DataLoaders
    trainloader = DataLoader(Xtrain, ytrain, 
                            target_clean_ratio=0.5, artef_distro=ARTEFACT_DISTRO, batch_size=10)
    valloader = DataLoader(Xval, yval,
                            target_clean_ratio=0.5, artef_distro=ARTEFACT_DISTRO, batch_size=10)
    testloader = DataLoader(Xtest, ytest,
                            target_clean_ratio=0.5, artef_distro=ARTEFACT_DISTRO, batch_size=10)

    # compile model
    model = getConvNet(out_classes=2, input_shape=(256,256,64,1))

    # fit model
    model.fit(trainloader, 
            validation_data=valloader,
            use_multiprocessing=True,
            epochs=3) 
    
# TODO:
# tinker with the model and input shapes - consider what's best for arbitrary datasets
# train with model checkpoints
# implement proper training loop
#      * model checkpoints
#      * train first on small batches with 50/50
#      * gradually increase batch size and clean ratio
# evaluate on test set
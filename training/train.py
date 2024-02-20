from sklearn.model_selection import train_test_split

from train_utils import DataCrawler, DataLoader

DATADIR = '/Users/odysseasvavourakis/Documents/2023-2024/Studium/SABS/GE Project/data.nosync/struc_data'
DATASETS = ['artefacts'+i for i in [1,2,3]]
CONTRASTS = ['T1wMPR', 'T1wTIR', 'T2w', 'T2starw', 'FLAIR']
QUALS = ['clean', 'exp_artefacts']

ARTEFACT_DISTRO = {
    # TODO: define a categorical probability distribution over these
    'RandomAffine' : None, 
    'RandomElasticDeformation' : None, 
    'RandomAnisotropy' : None, 
    'Intensity' : None, 
    'RandomMotion' : None, 
    'RandomGhosting' : None, 
    'RandomSpike' : None, 
    'RandomBiasField' : None, 
    'RandomBlur' : None, 
    'RandomNoise' : None, 
    'RandomSwap' : None, 
    'RandomGamma' : None
}

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

# TODO:
# load and compile model
# train with model checkpoints
# evaluate on test set
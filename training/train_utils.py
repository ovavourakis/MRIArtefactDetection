import torch, os, random, math
import numpy as np
import tensorflow as tf
import torchio as tio
from keras.utils import Sequence
from keras.callbacks import Callback
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit

class PrintModelWeightsNorm(Callback):
    def on_epoch_end(self, epoch, logs=None):
        weights_norm = tf.sqrt(sum(tf.reduce_sum(tf.square(w)) for w in self.model.weights))
        print(f"Norm of weights after epoch {epoch+1}: {weights_norm.numpy()}")


def split_by_patient(real_image_paths, pids, real_labels):
    # remove patients with only one image
    pid_counts = {pid:0 for pid in pids}
    for pid in pids:
        pid_counts[pid] += 1
    pids_to_remove = [pid for pid in pid_counts if pid_counts[pid] < 3]
    
    real_image_paths = [path for path, pid in zip(real_image_paths, pids) if pid not in pids_to_remove]
    real_labels = [label for label, pid in zip(real_labels, pids) if pid not in pids_to_remove]
    pids = [pid for pid in pids if pid not in pids_to_remove]

    artf_frc = [1,0,0]
    while max(artf_frc)-min(artf_frc)>0.02:
        for trvidx, testidx in GroupShuffleSplit(n_splits=1, test_size=0.1).split(real_image_paths, real_labels, pids):
            Xtrval, Xtest = [real_image_paths[i] for i in trvidx], [real_image_paths[i] for i in testidx]
            ytrval, ytest = [real_labels[i] for i in trvidx], [real_labels[i] for i in testidx]
            pid_trval, pid_test = [pids[i] for i in trvidx], [pids[i] for i in testidx]

        for tridx, validx in GroupShuffleSplit(n_splits=1, test_size=0.2222).split(Xtrval, ytrval, pid_trval):
            Xtrain, Xval = [Xtrval[i] for i in tridx], [Xtrval[i] for i in validx]
            ytrain, yval = [ytrval[i] for i in tridx], [ytrval[i] for i in validx]
            pid_train, pid_val = [pid_trval[i] for i in tridx], [pid_trval[i] for i in validx]

        # assert there is not overlap between pid_train and pid_test, and pid_val
        assert len(set(pid_train).intersection(set(pid_test))) == 0
        assert len(set(pid_val).intersection(set(pid_test))) == 0
        assert len(set(pid_train).intersection(set(pid_val))) == 0

        artf_frc = [sum(y)/len(y) for y in [ytrain, yval, ytest]]

    return Xtrain, Xval, Xtest, np.array(ytrain), np.array(yval), np.array(ytest)

class ImageReader():
    '''
    Class to read and preprocess images from disk.
    Handles image loading, synthetic augmentation and 
    preprocessing (resizing, normalisation, padding etc.).
    '''

    def __init__(self, input_size=(256,256,64)):
        self.input_size = input_size
        self.Synths = {
            "RandomZoom": tio.RandomAffine(scales=(1.5, 1.5)),          # zooming in the images 
            # "RandomElasticDeformation": tio.RandomElasticDeformation(), # INCORRECT: parameters unchosen; elastic deformation of the images 
            "RandomAnisotropy": tio.RandomAnisotropy(),                 # anisotropic deformation of the images
            "RescaleIntensity": tio.RescaleIntensity((0.5, 1.5)),       # rescaling the intensity of the images
            "RandomMotion": tio.RandomMotion(degrees = (10,15), translation = (10,15)), # filling the  𝑘 -space with random rigidly-transformed versions of the original images
            "RandomGhosting": tio.RandomGhosting(),                     # removing every  𝑛 th plane from the k-space
            "RandomSpike": tio.RandomSpike(),                           # signal peak in  𝑘 -space,
            "RandomBiasField": tio.RandomBiasField(coefficients = (0.3,0.6)), # Magnetic field inhomogeneities in the MRI scanner produce low-frequency intensity distortions in the images
            "RandomBlur": tio.RandomBlur(std = (0.6,1.1)),              # blurring the images
            # "RandomNoise": tio.RandomNoise(),                         # INCORRECT: does not work (why ?); adding noise to the images
            "RandomSwap": tio.RandomSwap(),                             # swapping the phase and magnitude of the images
            "RandomGamma": tio.RandomGamma(log_gamma = (0.25,0.35)),    # intensity of the images
            "RandomWrapAround" : self.RandomWrapAround                     # wrapping around the images: Aliasing Artifact
        }
        # pre-processing
        self.orient = tio.transforms.ToCanonical()                  # RAS+ orientation
        self.normalise = tio.transforms.ZNormalization()            # TODO: consider histogram normalisation instead
        self.crop_pad = tio.transforms.CropOrPad(self.input_size)   # TODO: consider re-sampling instead, or in addition
        self.preprocess = tio.Compose([self.orient, self.normalise, self.crop_pad])

    def RandomWrapAround(self, subject):
        """ Wrapping around the images: Aliasing Artifact """
        modified = subject
        dim_value = np.random.choice([0, 1, 2]) # Pick the dimension to wrap on
        side = np.random.choice([0, 1]) # Pick the side we want to add the wrap-around
        nb_rows = random.randint(subject.shape[dim_value+1]//4, subject.shape[dim_value+1]//3) 
        if side == 0: # wrap the last rows to the firsts rows
                if dim_value == 0: modified.data[:, :nb_rows, :, :] += subject.data[:,-nb_rows:, :, :] 
                elif dim_value == 1: modified.data[:, :, :nb_rows, :] += subject.data[:, :,-nb_rows:, :] 
                elif dim_value == 2: modified.data[:, :, :, :nb_rows] += subject.data[:, :, :,-nb_rows:]
        elif side == 1: # wrap the firsts rows to the last rows
                if dim_value == 0: modified.data[:,-nb_rows:, :, :] += subject.data[:, :nb_rows, :, :] 
                elif dim_value == 1: modified.data[:, :,-nb_rows:, :] += subject.data[:, :, :nb_rows, :] 
                elif dim_value == 2: modified.data[:, :, :, -nb_rows:] += subject.data[:, :, :, :nb_rows]
        return modified
 

    def _apply_modifications(self, img_path):
        # 1 - Checking the extension on the img_path + storing it as extension_name (i.e the modification to apply)
        extensions = ["CAug", "RandomZoom", 
          'RandomAnisotropy', 'RescaleIntensity', 'RandomMotion', 'RandomGhosting', 'RandomSpike', 
          'RandomBiasField', 'RandomBlur', 'RandomSwap', 'RandomGamma', 'RandomWrapAround']
        extension_name = None 
        for extension in extensions:
            if extension in img_path:
                extension_name = extension
                break
        
         # 2 - Creating a torchIO image from the path
        if extension_name is None: # no modification to be applied
            img = tio.ScalarImage(img_path)
            return img
        stripped_img_path = img_path.replace(f"_{extension_name}.nii", ".nii")   
        img = tio.ScalarImage(stripped_img_path)
        
        # 3 - Defining the modification from extension_name
        if extension_name == "CAug":
            binary_flip = [np.random.choice([0, 1]) for _ in range(3)]
            idx_flip = [index for index, value in enumerate(binary_flip) if value == 1]
            flipping = tio.RandomFlip(axes=idx_flip, flip_probability=1)
            scaling = tio.RandomAffine(scales=(0.1)) # If only one value: U(1-x, 1+x)
            shifting = tio.RandomAffine(translation=(0.1)) # If only one value: U(-x, x)
            rotating = tio.RandomAffine(degrees=(10)) # If only one value: U(-x, x) 
            modification = tio.Compose([flipping, scaling, shifting, rotating])

        elif extension_name in ["RandomZoom", 
                                'RandomAnisotropy', 'RescaleIntensity', 
                                'RandomMotion', 'RandomGhosting', 'RandomSpike', 
                                'RandomBiasField', 'RandomBlur', 
                                'RandomSwap', 'RandomGamma', 'RandomWrapAround']:
            modification = self.Synths[extension_name]
        
        else:
            raise ValueError(f"Path extension (augmentation) {extension_name} not recognised")
        
         # 4 - Apply the modification on the modified image and return modified version
        modified_img = modification(img)   
        return modified_img

    def read_image(self, path):
        img = self._apply_modifications(path) # introduce artefacts, if specified in path_name
        img = self.preprocess(img) # re-orient, normalise, crop/pad
        return img # numpy array

class DataCrawler():
    """
    Crawls the dataset directories and returns a list of clean/artefact 
    image paths with labels.
    """

    def __init__(self, datadir, datasets, image_contrasts, image_qualities):
        self._check_inputs(datadir, datasets, image_contrasts, image_qualities)
        self.datadir = datadir
        self.datasets = datasets
        self.image_contrasts = image_contrasts
        self.image_qualities = image_qualities
       
    def _check_inputs(self, datadir, datasets, image_contrasts, image_qualities):
        # do paths to all requested types of images exist?
        assert(os.path.isdir(datadir))
        for d in datasets:
            assert(os.path.isdir(os.path.join(datadir, d)))
            for c in image_contrasts:
                assert(c in ['T1wMPR', 'T1wTIR', 'T2w', 'T2starw', 'FLAIR'])
                if os.path.isdir(os.path.join(datadir, c)):
                    for q in image_qualities:
                        assert(q in ['clean', 'exp_artefacts'])
                        assert(os.path.isdir(os.path.join(datadir, c, q)))

    def crawl(self):
        # get all the (non-synthetic) image paths
        clean_img_paths = []
        artefacts_img_paths = []
        for d in self.datasets:
            for c in self.image_contrasts:
                if os.path.isdir(os.path.join(self.datadir, d, c)):
                  for q in self.image_qualities:
                      path = os.path.join(self.datadir, d, c, q)
                      images = os.listdir(path)
                      img_paths = [os.path.join(path, i) for i in images]
                      if q == 'clean':
                          clean_img_paths.extend(img_paths)
                      else:
                          artefacts_img_paths.extend(img_paths)
        # parse out the subject IDs from the paths
        clean_ids = [p.split('sub-')[1].split('_')[0] for p in clean_img_paths]
        artefacts_ids = [p.split('sub-')[1].split('_')[0] for p in artefacts_img_paths]
        # define the appropriate labels
        num_clean = len(clean_img_paths)
        num_artefacts = len(artefacts_img_paths)
        y_true = np.array([0]*num_clean + [1]*num_artefacts)

        return clean_img_paths + artefacts_img_paths, clean_ids + artefacts_ids, y_true

class DataLoader(Sequence):
    """
    Dataloader for an image dataset.
    Pre-defines augmentations to be performed in order to reach a target clean-ratio.
    """

    def __init__(self, Xpaths, y_true, batch_size, 
                  train_mode=True, target_clean_ratio=None, artef_distro=None):
        self._check_inputs(Xpaths, y_true, artef_distro, target_clean_ratio, train_mode)

        self.batch_size = batch_size
        self.num_classes = len(np.unique(y_true))
        self.reader = ImageReader(input_size=(256,256,64))
        
        # paths to just the *real* images
        clean_idx, artefact_idx = np.where(y_true == 0)[0], np.where(y_true == 1)[0]
        self.Clean_img_paths = [Xpaths[i] for i in clean_idx]
        self.Artefacts_img_paths = [Xpaths[i] for i in artefact_idx]
        # self.clean_ratio = self._get_clean_ratio()

        self.train_mode = train_mode
        if self.train_mode:
            self.target_clean_ratio = target_clean_ratio
            self.target_artef_distro = artef_distro

        # paths to all images, including synthetic ones; split into batches
        self.clean_img_paths, self.artefacts_img_paths, self.batches, self.labels = self.on_epoch_end()

    def _check_inputs(self, Xpaths, y_true, artef_distro, target_clean_ratio, train_mode):
        assert(len(Xpaths) == len(y_true))
        assert(len(np.unique(y_true)) == 2)
        if train_mode:
            assert(target_clean_ratio >= 0 and target_clean_ratio <= 1)
            # artef_distro defines a categorical probability distribution over (synthetic) artefact types
            assert(isinstance(artef_distro, dict))
            assert(np.allclose(sum(artef_distro.values()), 1))
            for k, v in artef_distro.items():
                assert(v >= 0)
                assert(k in ['RandomZoom', 
                            'RandomAnisotropy', 'RescaleIntensity', 'RandomMotion', 'RandomGhosting', 
                            'RandomSpike', 'RandomBiasField', 'RandomBlur', 
                            'RandomSwap', 'RandomGamma', 'RandomWrapAround'])

    def _get_clean_ratio(self):
        # calculate fraction of clean images (among non-synthetic data)
        C = len(self.Clean_img_paths)
        T = C + len(self.Artefacts_img_paths)
        return C/T, C, T
    
    def _define_augmentations(self):
        '''Generates pathnames for synthetic images to be created in order to reach target clean-ratio.
        Synthetic clean images defined by random {flips, shifts, scales, rotations}.
        Synthetic artefact images defined by transforms drawn from a categorical distro over artefact types.'''
        
        def _pick_augment(path, aug_type='clean'):
            if aug_type=='clean':
                aug = 'CAug'
            elif aug_type=='artefact':
                augs = list(self.target_artef_distro.keys())
                probs = list(self.target_artef_distro.values())
                aug = np.random.choice(augs, size = 1, p = probs)[0]
            else:
                raise ValueError('aug_type must be either "clean" or "artefact"')
            dir, file = os.path.split(path)
            fname, ext = file.split('.', 1) # just the first dot, so we don't split extensions like .nii.gz
            print(dir, fname, aug, ext)
            return os.path.join(dir, fname) + '_' + aug + '.' + ext
        
        clean_ratio, cleans, total = self._get_clean_ratio()
        clean_img_paths = self.Clean_img_paths
        artefacts_img_paths = self.Artefacts_img_paths
    
        if clean_ratio < self.target_clean_ratio:
            # oversample clean images with random {flips, shifts, 10% scales, rotations} 
            # until desired clean-ratio is reached
            num_imgs_to_aug = int( (total*self.target_clean_ratio - cleans) / (1-self.target_clean_ratio))
            imgs_to_aug = random.choices(clean_img_paths, k=num_imgs_to_aug)
            augmented_paths = [_pick_augment(path, aug_type='clean') for path in imgs_to_aug]
            clean_img_paths.extend(augmented_paths)
        elif clean_ratio > self.target_clean_ratio:
            # create synthetic artefacts until desired clean-ratio is reached
            # pick aigmentations from categorcical distro over transform functions of TorchIO
            num_imgs_to_aug = cleans/self.target_clean_ratio - total
            imgs_to_aug = random.choices(clean_img_paths, k=num_imgs_to_aug)
            augmented_paths = [_pick_augment(path, aug_type='artefact') for path in imgs_to_aug]
            artefacts_img_paths.extend(augmented_paths)
        
        return clean_img_paths, artefacts_img_paths

    def __len__(self):
        return len(self.batches)

    def _def_batches(self, clean_img_paths, artefacts_img_paths):
        """ Determine batches at start of each epoch:
        Assign to batches the paths with the repartition we target
        Makes sure target clean-ratio is maintained in each batch
        Makes sure all the existing images are used in the epoch
        """
        
        if self.train_mode: 
            # 1 - Decide on number of clean and artefact images in each batch with the repartition we target
            num_clean = int(self.batch_size * self.target_clean_ratio )
            num_artefacts = self.batch_size - num_clean

            # 2 - Randomly assign the paths to the batches along with their labels
            nb_batches = len(clean_img_paths + artefacts_img_paths) // self.batch_size
            random.shuffle(self.clean_img_paths)
            random.shuffle(self.artefacts_img_paths)
            batches = [
                self.clean_img_paths[i * num_clean:(i + 1) * num_clean] + self.artefacts_img_paths[i * num_artefacts:(i + 1) * num_artefacts]
                for i in range(nb_batches)
                ]
            labels = [[0]*num_clean + [1]*num_artefacts for _ in range(nb_batches)]

            # 3 - Shuffle batch in batches ALONG with their labels
            for i in range(nb_batches):
                zipped = list(zip(batches[i], labels[i]))
                random.shuffle(zipped)
                batches[i], labels[i] = zip(*zipped)
                batches[i] = list(batches[i])
                labels[i] = list(labels[i])
        else:
            # in this case we don't care about the fraction of cleans in each batch, just chunk the data
            files = clean_img_paths + artefacts_img_paths
            flat_labels = [0]*len(clean_img_paths) + [1]*len(artefacts_img_paths)
            
            batches, labels = [], []
            while len(files) > 0:
                end_index = min(self.batch_size, len(files))
                batches.append(files[:end_index]); labels.append(flat_labels[:end_index])
                files = files[self.batch_size:]; flat_labels = flat_labels[self.batch_size:]

        return batches, labels

    def __getitem__(self,idx):
        '''Generates the batch, with associated labels.'''
        # read in the images, augment with artefacts as neccessary, then apply pre-processing
        # start_time = time.time()
        batch_images = [self.reader.read_image(path) for path in self.batches[idx]]  # TODO: this takes ~30s for 10-image batch
        # print(f"Batch image loading time: {time.time() - start_time} seconds")
        X = torch.stack([img.data.permute(1, 2, 3, 0) for img in batch_images])

        
        # also get appropriate labels
        y_true = self.labels[idx]
        y_one_hot = tf.one_hot(y_true, depth=self.num_classes)

        return X, y_one_hot
    
    def on_epoch_end(self):
         # get paths to all images, including synthetic ones, if requested
        if self.train_mode:
            # re-define augmentations to do for next epoch
            self.clean_img_paths, self.artefacts_img_paths = self._define_augmentations()
        else:
            self.clean_img_paths, self.artefacts_img_paths = self.Clean_img_paths, self.Artefacts_img_paths
        # split these new image paths into batches
        self.batches, self.labels = self._def_batches(self.clean_img_paths, self.artefacts_img_paths)

        return self.clean_img_paths, self.artefacts_img_paths, self.batches, self.labels
    
def plot_train_metrics(history, filename):
    '''Plot the training metrics'''

    # Plotting 4x4 metrics
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training and Validation Metrics')

    # Loss subplot
    axs[0, 0].plot(history.history['loss'], label='Train Loss')
    axs[0, 0].plot(history.history['val_loss'], label='Validation Loss')
    axs[0, 0].set_title('Loss')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].legend()

    # Accuracy subplot
    axs[0, 1].plot(history.history['accuracy'], label='Train Accuracy')
    axs[0, 1].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axs[0, 1].set_title('Accuracy')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Accuracy')
    axs[0, 1].legend()

    # AUROC subplot
    axs[1, 0].plot(history.history['auroc'], label='Train AUC')
    axs[1, 0].plot(history.history['val_auroc'], label='Validation AUC')
    axs[1, 0].set_title('AUC')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('AUC')
    axs[1, 0].legend()

    # AUPRC subplot
    axs[1, 1].plot(history.history['auprc'], label='Train AP')
    axs[1, 1].plot(history.history['val_auprc'], label='Validation AP')
    axs[1, 1].set_title('AP')
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('AP')
    axs[1, 1].legend()

    plt.tight_layout()
    plt.savefig(filename)
import os, random
import numpy as np
import torchio as tio
from keras.utils import Sequence
from keras.preprocessing.image import load_img, img_to_array
import math


class ImageReader():
    '''
    Class to read and preprocess images from disk.
    Handles image loading, synthetic augmentation and 
    preprocessing (resizing, normalisation, padding etc.).
    '''

    def __init__(self, input_size=(256,256,64)):
        self.input_size = input_size
        self.Synths = {
            "RandomAffine": tio.RandomAffine(scales=(1.5, 1.5)),        # zooming in the images 
            "RandomElasticDeformation": tio.RandomElasticDeformation(), # elastic deformation of the images, 
            "RandomAnisotropy": tio.RandomAnisotropy(),                 # anisotropic deformation of the images
            "RescaleIntensity": tio.RescaleIntensity((0.5, 1.5)),       # rescaling the intensity of the images
            "RandomMotion": tio.RandomMotion(),                         # filling the  𝑘 -space with random rigidly-transformed versions of the original images
            "RandomGhosting": tio.RandomGhosting(),                     # removing every  𝑛 th plane from the k-space
            "RandomSpike": tio.RandomSpike(),                           # signal peak in  𝑘 -space,
            "RandomBiasField": tio.RandomBiasField(),                   # Magnetic field inhomogeneities in the MRI scanner produce low-frequency intensity distortions in the images
            "RandomBlur": tio.RandomBlur(),                             # blurring the images
            "RandomNoise": tio.RandomNoise(),                           # adding noise to the images
            "RandomSwap": tio.RandomSwap(),                             # swapping the phase and magnitude of the images
            "RandomGamma": tio.RandomGamma()                            # intensity of the images
        }
        # pre-processing
        self.orient = tio.transforms.ToCanonical()                  # RAS+; TODO: might make pre-trained weights useless
        self.normalise = tio.transforms.ZNormalization()            # TODO: consider histogram normalisation instead
        self.crop_pad = tio.transforms.CropOrPad(self.input_size)   # TODO: consider re-sampling instead, or in addition
        self.preprocess = tio.Compose([self.orient, self.normalise, self.crop_pad])

    def _apply_modifications(self, img_path):
        # 1 - Checking the extension on the img_path + storing it as extension_name (i.e the modification to apply)
        extensions = ["CAug", "RandomAffine", 'RandomElasticDeformation' 
          'RandomAnisotropy', 'RescaleIntensity', 'RandomMotion', 'RandomGhosting', 'RandomSpike', 
          'RandomBiasField', 'RandomBlur', 'RandomNoise','RandomSwap', 'RandomGamma']
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

        elif extension_name in ["RandomAffine", 'RandomElasticDeformation' 'RandomAnisotropy', 'RescaleIntensity', 
                                'RandomMotion', 'RandomGhosting', 'RandomSpike', 'RandomBiasField', 'RandomBlur', 
                                'RandomNoise','RandomSwap', 'RandomGamma']:
            modification = self.Synths[extension_name]
        
         # 4 - Apply the modification on the modified image and return modified version
        modified_img = modification(img)   
        # modified_img.save(img_path)
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
        self._check_inputs(datadir, image_contrasts, image_qualities)
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
                for q in self.image_qualities:
                    path = os.path.join(d, c, q)
                    images = os.listdir(path)
                    img_paths = [os.path.join(path, i) for i in images]
                    if q == 'clean':
                        clean_img_paths.extend(img_paths)
                    else:
                        artefacts_img_paths.extend(img_paths)
        # define the appropriate labels
        num_clean = len(clean_img_paths)
        num_artefacts = len(artefacts_img_paths)
        y_true = np.array([1]*num_clean + [0]*num_artefacts)

        return clean_img_paths + artefacts_img_paths, y_true

class DataLoader(Sequence):
    """
    Dataloader for an image dataset.
    Pre-defines augmentations to be performed in order to reach a target clean-ratio.
    """

    def __init__(self, Xpaths, y_true, target_clean_ratio, artef_distro, batch_size):
        self._check_inputs(Xpaths, y_true, artef_distro, target_clean_ratio)

        self.batch_size = batch_size; 
        self.target_clean_ratio = target_clean_ratio
        self.target_artef_distro = artef_distro

        clean_idx, artefact_idx = np.where(y_true == 1)[0], np.where(y_true == 0)[0]
        # paths to just the *real* images
        self.Clean_img_paths, self.Artefacts_img_paths = Xpaths[clean_idx], Xpaths[artefact_idx]
        self.clean_ratio = self._get_clean_ratio()

        # paths to all images, including synthetic ones; split into batches
        self.clean_img_paths, self.artefacts_img_paths, self.batches, self.labels = self.on_epoch_end()

        self.reader = ImageReader(input_size=(256,256,64))

    def _check_inputs(self, Xpaths, y_true, artef_distro, target_clean_ratio):
        assert(len(Xpaths) == len(y_true))
        assert(target_clean_ratio >= 0 and target_clean_ratio <= 1)
        # artef_distro defines a categorical probability distribution over (synthetic) artefact types
        assert(isinstance(artef_distro, dict))
        assert(sum(artef_distro.values()) == 1)
        for k, v in artef_distro.items():
            assert(v >= 0)
            assert(k in ['RandomAffine', 'RandomElasticDeformation', 'RandomAnisotropy', 'Intensity', 
                         'RandomMotion', 'RandomGhosting', 'RandomSpike', 'RandomBiasField', 'RandomBlur', 
                         'RandomNoise', 'RandomSwap', 'RandomGamma'])

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
                aug = '_CAug'
            elif aug_type=='artefact':
                augs = list(self.target_artef_distro.keys())
                probs = list(self.target_artef_distro.values())
                aug = np.random.choice(augs, size = 1, p = probs)
            else:
                raise ValueError('aug_type must be either "clean" or "artefact"')
            name, ext = os.path.splitext(path)
            return name + aug + ext
        
        clean_ratio, cleans, total = self._get_clean_ratio()
        clean_img_paths = self.Clean_img_paths
        artefacts_img_paths = self.Artefacts_img_paths
    
        if clean_ratio < self.target_clean_ratio:
            # oversample clean images with random {flips, shifts, 10% scales, rotations} 
            # until desired clean-ratio is reached
            num_imgs_to_aug = int( (total*self.target_clean_ratio - cleans) / (1-self.target_clean_ratio))
            imgs_to_aug = random.sample(clean_img_paths, num_imgs_to_aug)
            augmented_paths = [_pick_augment(path, aug_type='clean') for path in imgs_to_aug]
            clean_img_paths.extend(augmented_paths)
        elif clean_ratio > self.target_clean_ratio:
            # create synthetic artefacts until desired clean-ratio is reached
            # pick aigmentations from categorcical distro over transform functions of TorchIO
            num_imgs_to_aug = cleans/self.target_clean_ratio - total
            imgs_to_aug = random.sample(clean_img_paths, num_imgs_to_aug)
            augmented_paths = [_pick_augment(path, aug_type='artefact') for path in imgs_to_aug]
            artefacts_img_paths.extend(augmented_paths)
        
        return clean_img_paths, artefacts_img_paths

    def __len__(self):
        return math.ceil((len(self.clean_img_paths)+len(self.artefacts_img_paths)) // self.batch_size)

    def _def_batches(self, clean_img_paths, artefacts_img_paths):
        """ Determine batches at start of each epoch:
        Assign to batches the paths with the repartition we target
        Makes sure target clean-ratio is maintained in each batch
        Makes sure all the existing images are used in the epoch
        """

        # 1 - Decide on number of clean and artefact images in each batch with the repartition we target
        num_clean = int(self.batch_size * self.target_clean_ratio)
        num_artefacts = self.batch_size - num_clean

        # 2 - Randomly assign the paths to the batches along with their labels
        nb_batches = len(clean_img_paths + artefacts_img_paths) // self.batch_size
        random.shuffle(self.clean_img_paths)
        random.shuffle(self.artefacts_img_paths)
        batches = [
            self.clean_img_paths[i * num_clean:(i + 1) * num_clean] + self.artefacts_img_paths[i * num_artefacts:(i + 1) * num_artefacts]
            for i in range(nb_batches)
            ]
        labels = [[1]*num_clean + [0]*num_artefacts for _ in range(nb_batches)]

        # 3 - Shuffle batch in batches ALONG with their labels
        for i in range(nb_batches):
            zipped = list(zip(batches[i], labels[i]))
            random.shuffle(zipped)
            batches[i], labels[i] = zip(*zipped)
            batches[i] = list(batches[i])
            labels[i] = list(labels[i])

        return batches, labels


    def __getitem__(self,idx):
        '''Generates the batch, with associated labels.'''
        # read in the images, augment with artefacts as neccessary, then apply pre-processing
        batch_images = [self.reader.read_image(path) for path in self.batches[idx]]
        X = np.array([img.data for img in batch_images])
        
        # also get appropriate labels
        y_true = self.labels[idx]

        return X, y_true
    
    def on_epoch_end(self):
        # re-define augmentations to do for next epoch
        # get paths to all images, including synthetic ones
        self.clean_img_paths, self.artefacts_img_paths = self._define_augmentations()
        # split these new image paths into new random batches
        self.batches, self.labels = self._def_batches(self.clean_img_paths, self.artefacts_img_paths)

        return self.clean_img_paths, self.artefacts_img_paths, self.batches, self.labels


# TODO:
# implement model
# implement training loop
#      * train first on small batches with 50/50
#      * gradually increase batch size and clean ratio
# evaluate 
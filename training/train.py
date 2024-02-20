import os, random
import numpy as np
import torchio as tio
from keras.utils import Sequence
from keras.preprocessing.image import load_img, img_to_array

class ImageReader():
    '''
    Class to read and preprocess images from disk.
    Handles image loading, resizing, normalisation, padding.
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
        img = self._apply_modifications(path) # introduce artefacts, if need be
        img = self.preprocess(img) # re-orient, normalise, crop/pad
        return img # numpy array

class DataLoader(Sequence):
    """
    Dataloader for training the model. 
    Handles data augmentation and synthetic artefact generation.
    """

    def __init__(self, datadir, datasets, image_contrasts, image_qualities, 
                 target_clean_ratio, artef_distro, batch_size):
        
        self._check_inputs(datadir, image_contrasts, image_qualities, artef_distro, target_clean_ratio)

        self.batch_size = batch_size
        self.target_clean_ratio = target_clean_ratio
        self.target_artef_distro = artef_distro

        self.Clean_img_paths, self.Artefacts_img_paths = self._crawl_paths(datadir, datasets, 
                                                                           image_contrasts, image_qualities)
        self._define_augmentations()
        [random.shuffle(paths) for paths in [self.clean_img_paths, self.artefacts_img_paths]]

        self.clean_ratio = self._get_clean_ratio()
        self.reader = ImageReader(input_size=(256,256,64))

    def _check_inputs(self, datadir, image_contrasts, image_qualities, artef_distro, target_clean_ratio):
        assert(target_clean_ratio >= 0 and target_clean_ratio <= 1)
        # do paths to all requested types of images exist?
        assert(os.path.isdir(datadir))
        for c in image_contrasts:
            assert(c in ['T1wMPR', 'T1wTIR', 'T2w', 'T2starw', 'FLAIR'])
            assert(os.path.isdir(os.path.join(datadir, c)))
            for q in image_qualities:
                assert(q in ['clean', 'exp_artefacts'])
                assert(os.path.isdir(os.path.join(datadir, c, q)))
        # artef_distro defines a categorical probability distribution over (synthetic) artefact types
        assert(isinstance(artef_distro, dict))
        assert(sum(artef_distro.values()) == 1)
        for k, v in artef_distro.items():
            assert(v >= 0)
            assert(k in ['RandomAffine', 'RandomElasticDeformation', 'RandomAnisotropy', 'Intensity', 
                         'RandomMotion', 'RandomGhosting', 'RandomSpike', 'RandomBiasField', 'RandomBlur', 
                         'RandomNoise', 'RandomSwap', 'RandomGamma'])

    def _crawl_paths(self, datadir, datasets, image_contrasts, image_qualities):
        # get all the (non-synthetic) image paths
        clean_img_paths = []
        artefacts_img_paths = []
        for d in datasets:
            for c in image_contrasts:
                for q in image_qualities:
                    path = os.path.join(d, c, q)
                    images = os.listdir(path)
                    img_paths = [os.path.join(path, i) for i in images]
                    if q == 'clean':
                        clean_img_paths.extend(img_paths)
                    else:
                        artefacts_img_paths.extend(img_paths)
        return clean_img_paths, artefacts_img_paths

    def _get_clean_ratio(self):
        # calculate fraction of clean images (among non-synthetic data)
        C = len(self.clean_img_paths)
        T = C + len(self.artefacts_img_paths)
        return C/T, C, T
    
    def _define_augmentations(self):
        '''Generates pathnames for synthetic images to be created in order to reach target clean-ratio.
        Synthetic clean images defined by random {flips, shifts, scales, rotations}.
        Synthetic artefact images defined by transforms drawn from a categorical distro over artefact types.'''
        
        self.clean_img_paths = self.Clean_img_paths
        self.clean_img_paths = self.Artefacts_img_paths
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

        if clean_ratio < self.target_clean_ratio:
            # oversample clean images with random {flips, shifts, 10% scales, rotations} 
            # until desired clean-ratio is reached
            num_imgs_to_aug = int( (total*self.target_clean_ratio - cleans) / (1-self.target_clean_ratio))
            imgs_to_aug = random.sample(self.clean_img_paths, num_imgs_to_aug)
            augmented_paths = [_pick_augment(path, aug_type='clean') for path in imgs_to_aug]
            self.clean_img_paths.extend(augmented_paths)
        elif clean_ratio > self.target_clean_ratio:
            # create synthetic artefacts until desired clean-ratio is reached
            # pick aigmentations from categorcical distro over transform functions of TorchIO
            num_imgs_to_aug = cleans/self.target_clean_ratio - total
            imgs_to_aug = random.sample(self.clean_img_paths, num_imgs_to_aug)
            augmented_paths = [_pick_augment(path, aug_type='artefact') for path in imgs_to_aug]
            self.artefacts_img_paths.extend(augmented_paths)

    def __def_batches(self):
        """ Determine batches at start of each epoch:
        - redefine augmentations 
        - assign to batches the paths with the repartition we target
        Makes sure target clean-ratio is maintained in each batch
        Makes sure all the existing images are used in the epoch
        """

        # 1 - Decide on number of clean and artefact images in each batch with the repartition we target
        num_clean = int(self.batch_size * self.target_clean_ratio)
        num_artefacts = self.batch_size - num_clean

        # 2 - Run _define_augmentations
        self._define_augmentations()

        # 3 - Assign the path to the batches
        nb_batches = len(self.clean_img_paths + self.artefacts_img_paths) // self.batch_size
        l = [[] for _ in range(self.nb_batches)]

            # for each batch in l: add to the batch randomly chosen num_clean images from the clean_img_paths without replacement 
            # same for artefacted images 


        # in each batch: assign num_clean clean images and num_artefacts artefact images randomly



        # creates l = [[paths1, paths2, ...], [pathX, pathX+1, ...]] list of batches
        # get_item
        # batches = l[idx]
        # return batch_paths
        pass



    def __getitem__(self,idx):
        '''Generates the batch, with associated labels.
        '''
        # decide on specific images to include in batch
        # TODO: assign the paths to the batches with the repartition we target
        clean_batch = random.sample(self.clean_img_paths, num_clean)
        artefact_batch = random.sample(self.artefacts_img_paths, num_artefacts)
        batch_paths = clean_batch + artefact_batch
        # read in the images and apply pre-processing
        batch_images = [self.reader.read_image(path) for path in batch_paths]
        X = np.array([img.data for img in batch_images])
        
        # generate labels: 1 for clean, 0 for artefact
        y_true = np.array([1]*num_clean + [0]*num_artefacts)

        return batch_paths, y_true
    
    


  

        



    # def _get_image_paths(self):
    #     num_exp_artefacts = 0
    #     np.random.shuffle(self.image_paths)
    #     return num_exp_artefacts / len(self.image_paths)

    # def __len__(self):
    #     return len(self.image_paths) // self.batch_size

    # def __getitem__(self, idx):
    #     batch_paths = self.image_paths[idx*self.batch_size:(idx+1)*self.batch_size]
    #     batch_images = np.array([img_to_array(load_img(path)) for path in batch_paths])
    #     return batch_images

    # def on_epoch_end(self):
    #     np.random.shuffle(self.image_paths)


# TODO:
# pre-determine batches at start of epoch
# re-define augmentations at start of each epoch

# implement model
# implement training loop
#      * train first on small batches with 50/50
#      * gradually increase batch size and clean ratio
# evaluate 
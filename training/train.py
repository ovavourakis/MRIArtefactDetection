import os, random
import numpy as np
from keras.utils import Sequence
from keras.preprocessing.image import load_img, img_to_array
import torchio as tio
import pandas as pd


# pick random image from clean images
# pick random augmentation from categorical distribution
# apply augmentation to image
# write image to temporary file
# add path to temporary file to self.tmp_img_paths

# also give option to augment by flipping or rotating

class CustomDatasetGenerator(Sequence):

    def __init__(self, datadir, datasets, image_contrasts, image_qualities, 
                 target_clean_ratio, artef_distro, batch_size):
        
        self._check_inputs(datadir, image_contrasts, image_qualities, artef_distro, target_clean_ratio)

        self.batch_size = batch_size
        self.target_clean_ratio = target_clean_ratio
        self.target_artef_distro = artef_distro
        self.Synths = {
            "RandomAffine": tio.RandomAffine(scales=(1.5, 1.5)), # zooming in the images 
            "RandomElasticDeformation": tio.RandomElasticDeformation(), # elastic deformation of the images, 
            "RandomAnisotropy": tio.RandomAnisotropy(), # anisotropic deformation of the images
            "RescaleIntensity": tio.RescaleIntensity((0.5, 1.5)), # rescaling the intensity of the images
            "RandomMotion": tio.RandomMotion(), # filling the  𝑘 -space with random rigidly-transformed versions of the original images
            "RandomGhosting": tio.RandomGhosting(), # removing every  𝑛 th plane from the k-space
            "RandomSpike": tio.RandomSpike(), # signal peak in  𝑘 -space,
            "RandomBiasField": tio.RandomBiasField(), # Magnetic field inhomogeneities in the MRI scanner produce low-frequency intensity distortions in the images
            "RandomBlur": tio.RandomBlur(), # blurring the images
            "RandomNoise": tio.RandomNoise(), # adding noise to the images
            "RandomSwap": tio.RandomSwap(), # swapping the phase and magnitude of the images
            "RandomGamma": tio.RandomGamma() # intensity of the images
            }

        self.clean_img_paths, self.artefacts_img_paths = self._crawl_paths(datadir, datasets, 
                                                                           image_contrasts, image_qualities)
        self._define_augmentations()
        [random.shuffle(paths) for paths in [self.clean_img_paths, self.artefacts_img_paths]]

        self.clean_ratio = self._get_clean_ratio()

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

    def _define_augmentations(self):
         # calculate fraction of clean images (among non-synthetic data)
        C = len(self.clean_img_paths)
        T = C + len(self.artefacts_img_paths)
        clean_ratio = C / T

        if clean_ratio < self.target_clean_ratio:
            # oversample clean images with random flips until desired clean-ratio is reached
            numer = C*(1-self.target_clean_ratio)-self.target_clean_ratio*(T-C)
            denom = self.target_clean_ratio-1
            num_imgs_to_flip = int(numer / denom)
            imgs_to_flip = [p+'_flip' for p in random.sample(self.clean_img_paths, num_imgs_to_flip)]
            self.clean_img_paths.extend(imgs_to_flip)
        elif clean_ratio > self.target_clean_ratio:
            # create synthetic artefacts until desired clean-ratio is reached
            num_imgs_to_augment = C/self.target_clean_ratio - T
            imgs_to_augment = [p+'_synth' for p in random.sample(self.clean_img_paths, num_imgs_to_augment)]
            self.artefacts_img_paths.extend(imgs_to_augment)


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

        if extension_name is None:
            return

        # 2 - Creating a torchIO image from the stripped path
        stripped_img_path = img_path.replace(f"_{extension_name}.nii", ".nii")        
        img = tio.ScalarImage(stripped_img_path)
        modified_img = img

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
            modification = Synths[extension_name]
        
         # 4 - Apply the modification on the modified image and finaly save it to the provided img_path
        modified_img = modification(modified_img)   
        modified_img.save(img_path)
            
                
    
    def _get_clean_ratio(self):
        C = len(self.clean_img_paths)
        T = C + len(self.artefacts_img_paths)
        return C / T

    def _batch_generator(self):
        pass # TODO

    def __getitem__(self, idx):
        'Generates one batch of data'
        pass # TODO


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
# read/create dataset
# implement augmentations in DataLoader
# implement model
# implement training loop
# evaluate 


Synths = {
    "RandomAffine": tio.RandomAffine(scales=(1.5, 1.5)), # zooming in the images 
    "RandomElasticDeformation": tio.RandomElasticDeformation(), # elastic deformation of the images, 
    "RandomAnisotropy": tio.RandomAnisotropy(), # anisotropic deformation of the images
    "RescaleIntensity": tio.RescaleIntensity((0.5, 1.5)), # rescaling the intensity of the images
    "RandomMotion": tio.RandomMotion(), # filling the  𝑘 -space with random rigidly-transformed versions of the original images
    "RandomGhosting": tio.RandomGhosting(), # removing every  𝑛 th plane from the k-space
    "RandomSpike": tio.RandomSpike(), # signal peak in  𝑘 -space,
    "RandomBiasField": tio.RandomBiasField(), # Magnetic field inhomogeneities in the MRI scanner produce low-frequency intensity distortions in the images
    "RandomBlur": tio.RandomBlur(), # blurring the images
    "RandomNoise": tio.RandomNoise(), # adding noise to the images
    "RandomSwap": tio.RandomSwap(), # swapping the phase and magnitude of the images
    "RandomGamma": tio.RandomGamma() # intensity of the images
    }
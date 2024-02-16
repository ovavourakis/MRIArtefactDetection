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

    def _apply_modifications(self, img_paths, modifications, artef_distro):
        modified_paths = []
        ground_truth_labels = pd.DataFrame(columns=['id', 'gt_score'])

        for path in img_paths: # going through all the images
            img = tio.ScalarImage(path) # creating a torchIO image from the path
            modified_img = img

            if modifications == flips: # augmentation of image: no augmentation distribution and groud truth score is 0
                modification_key = random.choice(list(modifications.keys()))
                modification = modifications[modification_key]
                gt_score = 0 
            elif synths == modifications: # corruption of image: modification distribution and groud truth score is 1
                modification_key = random.choice(list(artef_distro.keys()), p=list(artef_distro.values()))
                modification = modifications[modification_key]
                gt_score = 1 

            modified_img = modification(modified_img)
            # check if needed: modified_img.plot()
            modified_path = path.replace('.nii', f'_{modification_key}.nii')
            modified_img.save(modified_path)
            modified_paths.append(modified_path) 

            ground_truth_labels.loc[len(ground_truth_labels)] = [modified_path, gt_score]

        return modified_paths, ground_truth_labels
    
    # Then use it this way later in the process: 
    # flipped_clean_paths = self._apply_modifications(self.clean_img_paths, flips)
    # synth_artefacts_paths = self._apply_modifications(self.artefacts_img_paths, synths)
    
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
    

# TorchIO function to generate augmented images by flipping them along the 3 dimensions
flip_0 = tio.RandomFlip(axes=0, flip_probability=1) # 1st dimension
flip_1 = tio.RandomFlip(axes=1, flip_probability=1) # 2nd dimension
flip_2 = tio.RandomFlip(axes=2, flip_probability=1) # 3rd dimension
flip_0_1 = tio.Compose([flip_0, flip_1]) # 1st & 2nd dimension
flip_0_2 = tio.Compose([flip_0, flip_2]) # 1st & 3nd dimension
flip_1_2 = tio.Compose([flip_1, flip_2]) # 2nd & 3rd dimension
flip_0_1_2 = tio.Compose([flip_0, flip_1, flip_2]) # 1st & 2nd & 3rd dimension
flips = {"flip_0":flip_0, "flip_1":flip_1, "flip_2":flip_2,"flip_0_1":flip_0_1,
          "flip_0_2":flip_0_2, "flip_1_2":flip_1_2, "flip_0_1_2":flip_0_1_2}


# TorchIO function to generate synthetic images
RandomAffine = tio.RandomAffine(scales=(1.5, 1.5)) # zooming in the images
RandomElasticDeformation = tio.RandomElasticDeformation() # elastic deformation of the images
RandomAnisotropy = tio.RandomAnisotropy() # anisotropic deformation of the images
RescaleIntensity = tio.RescaleIntensity((0.5, 1.5)) # rescaling the intensity of the images
RandomMotion = tio.RandomMotion()  # filling the  𝑘 -space with random rigidly-transformed versions of the original images
RandomGhosting = tio.RandomGhosting() # removing every  𝑛 th plane from the k-space
RandomSpike = tio.RandomSpike() # signal peak in  𝑘 -space,
RandomBiasField = tio.RandomBiasField() # Magnetic field inhomogeneities in the MRI scanner produce low-frequency intensity distortions in the images
RandomBlur = tio.RandomBlur() # blurring the images
RandomNoise = tio.RandomNoise() # adding noise to the images
RandomSwap = tio.RandomSwap() # swapping the phase and magnitude of the images
RandomGamma = tio.RandomGamma() # intensity of the images

synths = {'RandomAffine': RandomAffine, 'RandomElasticDeformation': RandomElasticDeformation, 
          'RandomAnisotropy': RandomAnisotropy, 'RescaleIntensity': RescaleIntensity, 
          'RandomMotion': RandomMotion, 'RandomGhosting': RandomGhosting, 'RandomSpike': RandomSpike, 
          'RandomBiasField': RandomBiasField, 'RandomBlur': RandomBlur, 'RandomNoise': RandomNoise, 
          'RandomSwap': RandomSwap, 'RandomGamma': RandomGamma}

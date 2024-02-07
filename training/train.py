import os, random
import numpy as np
from keras.utils import Sequence
from keras.preprocessing.image import load_img, img_to_array


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
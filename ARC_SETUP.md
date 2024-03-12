# How to get set up with GPUs on ARC

1. [Login to ARC](https://arc-user-guide.readthedocs.io/en/latest/connecting-to-arc.html), then start an interactive session on CPU:
```
srun -p interactive --pty /bin/bash
```
2. Change to the $DATA directory, run `pwd` and note down the path.
```
cd $DATA && pwd
```
3. [Install `mamba`](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html), which we'll be using to set up our virtual environment. 
    * While navigating the installation dialogue, make sure you install it into the `$DATA` directory, e.g. at `$DATA/miniforge3`, otherwise you'll blow past the disk quota on your `$HOME` directory. You can use the path you copied earlier.
```
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
```
4. Load the below CUDA drivers, and also add the same line to your `~/.bashrc`. Newer driver versions may work, but have not been tested–you can load the current defaults by running `module load CUDA cuDNN`, instead.
```
module load CUDA/12.0.0 cuDNN/8.8.0.121-CUDA-12.0.0
module list
```
5. Create a new `Python 3.10.4` virtual environment using `mamba`. Newer versions of `Python` may work but have not been tested.
```
mamba create -n mri python=3.10.4 pip
mamba activate mri
```
6. Install `tensorflow` and `pytorch` with GPU acceleration into the environment, as well as `torchio` for eading MRI volumes, along with the usual suspects. Each of these can take some time.
```
mamba install tensorflow-gpu
mamba install pytorch torchvision torchaudio pytorch-cuda -c pytorch -c nvidia
mamba install torchio
mamba install numpy matplotlib scikit-learn keras pandas seaborn tqdm
```
7. `exit` out of the interactive session. Verify the installation in a new interactive session on a GPU-enabled node:
```
srun --gres=gpu:1 -p interactive --pty /bin/bash
module load CUDA/12.0.0 cuDNN/8.8.0.121-CUDA-12.0.0
mamba activate mri
python
>>> import tensorflow
>>> import torch
>>> import torchio
```
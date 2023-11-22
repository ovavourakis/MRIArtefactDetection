# Getting Started

## Setting Up

0. Make sure you have `git` and `Docker` installed on your system, and that Docker is running.
1. Clone this repository to your preferred location.
   ``` bash
   git clone git@github.com:ovavourakis/MRIArtefactDetection.git
   ```
2. Build the docker image.
   ``` bash
   docker build -t artefacts MRIArtefactDetection/setup/ 
   ```

## Running Inference

Now, to run inferences on your own data, you have to mount tha data, as well as the model code + inference scripts (this repository) into the container like so:

```
docker container run --mount type=bind,source="./MRIArtefactDetection",target=/root/artefacts_detection \
                      --mount type=bind,source="PATH/TO/DATA",target=/root/artefacts_detection/production/data \
                      -w /root/artefacts_detection/production \
                      -it artefacts
```

Be sure to replace `PATH/TO/DATA` with wherever you keep your data on your system (the top-level folder).

Because this is a bit much too write every time, you might want to consider setting up an `alias` for this long command in your shell. For example:

```
echo "alias artefact='docker container run --mount type=bind,source="./MRIArtefactDetection",target=/root/artefacts_detection \
                      --mount type=bind,source="PATH/TO/DATA",target=/root/artefacts_detection/production/data \
                      -w /root/artefacts_detection/production \
                      -it artefacts'" >> ~/.bashrc \
&& source ~/.bashrc
```

You can then launch the container using the command `artefact`.

Once the container is running, you're in the `production` directory of the code, from where you should be able to see your data and any local changes you've made to the code. Run your inference script as:

```
python my_inference_script.py
```

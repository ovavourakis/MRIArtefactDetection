# Getting Started

## Setting Up

0. Make sure you have `git` and `Docker` installed on your system.
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
docker container run --mount type=bind,source="MRIArtefactDetection",target=/root/artefacts_detection
                      --mount type=bind,source="PATH/TO/DATA",target=/root/artefacts_detection/production
                      -w /root/artefacts_detection/production
                      -it artefacts
```

This will start up the container. Switch to to `production` directory and run your inference script on your data.

```
cd ~/artefacts_detection/production && python my_inference_script.py
```
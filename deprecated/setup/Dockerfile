# Use an official Ubuntu base image
FROM ubuntu:latest

# Update and install necessary packages
RUN apt-get update && apt-get install -y wget && apt-get clean

# Set environment variables for Miniconda installation
ENV PATH=/root/miniconda3/bin:$PATH

# Download and install Miniconda
RUN mkdir -p ~/miniconda3 \
    && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh \
    && bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3 \
    && rm -rf ~/miniconda3/miniconda.sh

# Initialize Miniconda
RUN conda init bash

# Set up the Conda environment for the model
COPY artefact_env.yml /root/artefact_env.yml
RUN conda env create -f ~/artefact_env.yml \
    && conda clean -a \
    && rm ~/artefact_env.yml
RUN echo "conda activate artefact" >> ~/.bashrc

CMD ["/bin/bash"]

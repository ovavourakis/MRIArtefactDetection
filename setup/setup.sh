docker build -t artefacts .

CWD=$(readlink -f ..)

echo "Docker Image built successfully."
echo "To run the image, use the following command:"
echo " "
echo "docker container run --mount type=bind,source="$CWD",target=/root/artefacts_detection \\
                           --mount type=bind,source="PATH/TO/LOCAL/DATA",target=/root/artefacts_detection/production \\
                           -w /root/artefacts_detection/production  \\
                           -it artefacts"
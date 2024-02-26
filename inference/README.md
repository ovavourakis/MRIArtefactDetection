# Running Inference

Once you have set up and launched the container as described in the `setup` directory (including mounting any data and the code), you should be inside this `inference` directory, from where you should be able to see your data and any local changes you've made to the code. The model's conda environment `artefact` should be activated by default, otherwise activate it using

```
conda activate artefact
```

You're ready to run inference. Here, we provide an example inference script `raw_inference.py` that takes several CLI arguments (notably the input image volumes to classify and the number of MC runs per volume). From within the running container, you could run

```
python raw_inference.py -m 20 -i image1 image2 
```

If there are a a lot of image volumes to classify, you can also specify a list of them in a file and pass that file to `raw_inference.py` using the `-f` (or equivalently `--inputs_file`) flag, rather than writing them out one-by-one on the command line.

This script will output the raw predictions as a `csv`, where each row contains all predictions for the same image volume.


**GE: It would be helpful to have this output for a realistic dataset, along with the binary ground-truth (artefact/no) for each image volume.**

# Running Inference

This example script 

* reads in the test-set (written out during training)
* performs inference using the pre-trained model weights,
* saves the raw inference output to file (required for model evaluation; see `evaluation` directory)
* outputs some simple performance statistics on the test set.

To predict labels for new images (of unknown artefact status, i.e. not a test set), this script, as well as the DataLoader class will have to be modified.

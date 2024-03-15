# Analysing Results

## Theory

The raw model output is two softmaxed logits that may be interpreted as two different kinds of predictions:

1. as the probability that the image contains an artefact (or does not)
2. as a prediction that the image contains an artefact (or does not), depending on which logit is larger

These two interpretations give rise to two different ways to derive an aggregate predictive class probability from the raw model outputs across multiple MC inference-runs on the same image volume:

1. **pred_prob = mean prob** across MC runs (average over probabilities)
2. **pred_prob = mean predicted class** across MC runs (average over 0s and 1s)

These are in turn associated with different predictive uncertainties, and in general behave differently. The functions defined in this module analyse raw model predictions both ways and can be used to compare the result. 

Given either definition of the predictive probability, we also compare two different analyses modes:

1. **single-stage screening**:
	* every image volume above a particular threshold *probability* theta is classified as containing an artefact
2. **two-stage screening**: 
	* every image volume above a particular threshold *uncertainty* eta (of the probability estimate) is flagged for manual review and quarantined away; the model is unsure of what the correct probability is
	* of the remaining volumes, every image above a particular threshold *probability* theta is classified as containing an artefact

These considerations give rise to several possible evaluation metrics:

TODO ...

![Standard Classification Metrics](https://sinyi-chou.github.io/images/classification/metric_definition.png)

We also define a combination score that trades off some of these competing considerations appropriately:

TODO ...

## Practice

The functions in `analysis_utils.py` provide a scaffold for an analysis as sketched out above. Right now, the implementation of some of the individual metrics shouldn't be trusted / is in flux and the definition of the combined score is also not yet agreed upon (best to ignore anything to do with the combined score for now).

`run_analysis.py` gives an example of how to use these functions to analyse the toy results obtained from 20 MC runs on the small existing test set. Note the comments in that file.

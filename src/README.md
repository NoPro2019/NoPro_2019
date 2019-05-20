# README - Source Code

This diretory includes source code used to train LSTM models and infer summaries. 

## Training

For basic usage, simply running 

```python3 train.py```

should train the default BiLSTM model found in the models folder. This can be changed using the --model flag

## Inference

Inference can be run on the test set (consisting of unified expert-annotated gold labels) using 

```python3 inference.py [model]```

where model points to the previously trained model. ```inference_remaining.py``` can be used to generate summaries for the remaining unique comments from the C function database.

### Notes
Tested with tensorflow 1.13.1 and CUDA 9.0. The files assume that there is a "data/" folder in the current directory containing appropriately pickled training/testing data. The inference files assume that any models will be found in the automatically generated "trained/{modelname}/{date}" folders
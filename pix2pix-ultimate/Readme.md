# Ultimate

We train and test using a seperate file for every axis. The main implementation can be found in shared.py. This file exports functions that can be used from each axis module. These functions take a class as a parameter that contains every customizable function.

To train the model with 1 GPU use `CUDA_VISIBLE_DEVICES=1 python my_script.py`

# Files

  - metrics.py contains an implementation of the BRATS metrics
  - ops.py contains a wrapper for commonly used tensorflow functions
  - utils.py contains image manipulation functions
  - shared.py contains the model definition and program execution code
  - train-z.py is configuration file for shared.py that controls common runtime parameters

# Software Requierements

  - python 2.7
  - tensorflow 0.11

# Getting startet

Edit the train-z.py to adjust the used dataset. Most of the interestiung model properties can be changed there for example how many input images should be used or how deps each convolution should be. Larger changes like the model architecture shoud be shanged in the shared.py.

# Folder structure

Running the train-z.py creates a few folders.

- checkpoint-z contains the trained model parameters
- sample-z contains validation output
- test-z contains testing output
- log contains logging informations and runtime errors


# Start python server

### Requirements

`pip install flask`  
`export FLASK_APP=app.py`

### Usage
`flask run`

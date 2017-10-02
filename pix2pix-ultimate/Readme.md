# Ultimate

We train and test using a seperate file for every axis. The main implementation can be found in shared.py. This file exports functions that can be used from each axis module. These functions take a class as a parameter that contains every customizable function.

To train the model with 1 GPU use `CUDA_VISIBLE_DEVICES=1 python my_script.py`

# Files

  - metrics.py contains an implementation of the BRATS metrics
  - ops.py contains wraper for commonly used tensorflow functions
  - utils.py contains image manipulation functions
  - shared.py contains the model definition and program execution code
  - train-z.py is  configuration file for shared.py that controls common runtime parameters

# Software Requierements

  - python 2.7
  - tensorflow 0.11

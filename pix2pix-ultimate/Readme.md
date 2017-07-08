# Ultimate

We train and test using a seperate file for every axis. The main implementation can be found in shared.py. This file exports functions that can be used from each axis module. These functions take a class as a parameter that contains every customizable function.

To train the model with 1 GPU use `CUDA_VISIBLE_DEVICES=1 python my_script.py`

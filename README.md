# Starting code for course project of Machine Learning and Deep Learning (MLDL) 2023

## Getting started

You can play around with the code on your local machine, and use Google Colab for training on GPUs. 
In all the cases it is necessary to have the reduced version of the dataset where you are running the code. For simplicity, we inserted just the necessary frames at [this link](https://politoit-my.sharepoint.com/:f:/g/personal/simone_peirone_polito_it/EqCmaEAi2oBEqEqzBZ-pIA0Bke4EGNkUEhqwApEhUp9tDw?e=DtSbMP).

Before starting to implement your own code, make sure to:
1. read and study the material provided
2. understand how all the scripts are working and interacting
3. get familiar with the structure of the [EPIC-KITCHENS dataset](https://epic-kitchens.github.io/2022), what a sample, a clip and a frame are
4. play around with the code in the template to familiarize with all the tools.

Some scripts do not need to be run (i.e., [train_classifier_scratch.py](./train_classifier_scratch.py)) but are still inserted in the template in order to make the students understand how the baseline models are obtained.

### 1. Local

You can work on your local machine directly, the code which needs to be run does not require heavy computations. 
In order to do so a file with all the requirements for the python environment is provided [here](requirements.yaml), it contains even more packages than the strictly needed ones so if you want to install everything step-by-step just be careful to use pytorch 1.12 and torchvision 0.13. 

### 2. Google Colab

You can also run the code on [Google Colab](https://colab.research.google.com/).

- Upload all the scripts in this repo.
- Prepare a proper notebook structured as the `train_classifier.py` script.

As a reference, `colab_runner.ipynb` provides an example of how to set up a working environment in Google Colab.

NOTE: you need to stay connected to the Google Colab interface at all times for your python scripts to keep training.

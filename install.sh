#!/usr/bin/env bash

conda create --name simclr python=3.7
conda activate simclr

conda install -c pytorch pytorch
conda install -c pytorch torchvision
conda install -c conda-forge scikit-image

pip install torchsat
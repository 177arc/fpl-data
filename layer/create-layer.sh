#!/bin/bash

# This script generates a layer.zip file that can be used as an AWS lambda layer (see https://docs.aws.amazon.com/lambda/latest/dg/configuration-layers.html)
# The zip file has to be installed manually. This script has to be executed on a linux platform since AWS lambdas run on Linux.

# Remove target dir
rm -rf python

# Create directory
mkdir python

# Download python packages
pip3 install -r ../requirements.txt --ignore-installed --target python

# Remove tests to reduce size
find python -type d -name "__pycache__" -exec rm -rf {} +
find python -type d -name "tests" -exec rm -rf {} +
find python -type d -name "datasets" -exec rm -rf {} +

# Create zip file in the root directory
rm layer.zip || true
zip -r layer.zip python

#!/bin/bash

# Create and activate conda environment
conda create -n gpt2_finetune python=3.8 -y
conda activate gpt2_finetune

# Install dependencies from requirements.txt
pip install -r requirements.txt

echo "Environment setup complete!"
echo "To activate the environment, run: conda activate gpt2_finetune" 
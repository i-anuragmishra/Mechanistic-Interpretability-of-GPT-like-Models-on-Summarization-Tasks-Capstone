#!/bin/bash

# Activate the conda environment
source ~/miniconda3/bin/activate
conda activate gpt2_finetune

# Set environment variables for better GPU utilization
export CUDA_VISIBLE_DEVICES=0

# Run the experiment
echo "Starting GPT-2 fine-tuning experiment..."
python gpt2_summarization.py

# Monitor with TensorBoard (uncomment to run in background)
# tensorboard --logdir=logs &

echo "Experiment completed. Results saved in model_outputs/ and visualizations/" 
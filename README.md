# GPT-2 Fine-tuning for Text Summarization with Interpretability Analysis

## Project Title and Author Information
**Project Title**: GPT-2 Fine-tuning for Text Summarization with Mechanistic Interpretability Analysis  
**Author**: Anurag Mishra  
**Institution**: Rochester Institute of Technology  
**Contact**: am2552@rit.edu

## Cover Figure or Framework
![GPT-2 Attention Head Evolution](visualizations/attention_head_evolution.gif)

## Code Structure Overview
```
.
├── gpt2_summarization.py          # Main training and evaluation script
├── visualize_metrics.py           # Visualization generation script
├── analyze_latent_space.py        # Latent space analysis utilities
├── setup_environment.sh           # Environment setup script
├── run_experiment.sh              # Experiment execution script
├── requirements.txt               # Python dependencies
│
├── attention_analysis/            # Attention mechanism analysis
├── latent_space_evolution/        # Latent space evolution tracking
├── interpretability_metrics/      # Stored interpretability metrics
├── visualizations/                # Generated visualization files
├── logs/                          # Training and experiment logs
├── model_outputs/                 # Model outputs and predictions
└── checkpoints/                   # Model checkpoints (gitignored)
```

## Installation and Environment
### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- Conda package manager

### Setup
1. Create and activate the conda environment:
```bash
conda create -n gpt2_finetune python=3.8 -y
conda activate gpt2_finetune
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the setup script:
```bash
./setup_environment.sh
```

## Usage: Training and Testing
### Training
To run the full experiment:
```bash
./run_experiment.sh
```

This will:
1. Fine-tune GPT-2 on the CNN/Daily Mail dataset
2. Generate interpretability metrics
3. Create visualizations
4. Save model outputs

### Monitoring
Monitor training progress using TensorBoard:
```bash
tensorboard --logdir=logs
```

## Demo Example
### Sample Input (Article)
```
The president announced a new policy on climate change that aims to reduce carbon emissions by 50% by 2030. The announcement was made during a press conference where he outlined several initiatives to combat global warming and promote renewable energy. Environmental experts have praised the ambitious target, though some industry leaders have expressed concerns about implementation costs.
```

### Sample Output (Generated Summary)
```
The president announced a new policy on climate change that aims to reduce carbon emissions by 50% by 2030. Environmental experts have praised the ambitious target, though some industry leaders have expressed concerns about implementation costs.
```

## Results and Visualizations
The project generates several types of visualizations stored in the `visualizations/` directory:

1. **Attention Head Evolution**
   - GIF showing attention head metrics evolution
   - Interactive HTML visualizations
   - Heatmaps of attention patterns

2. **Latent Space Analysis**
   - 3D UMAP visualizations
   - Interactive HTML plots
   - Evolution GIFs

3. **Model Metrics**
   - Temporal metrics evolution
   - Weight changes visualization
   - Gradient norm analysis
   - Neuron activation patterns

All visualizations are accessible through the generated HTML report at `visualizations/interpretability_report.html`.

## Model Card Information
### Model Details
- **Model Type**: GPT-2 (fine-tuned)
- **Base Model**: GPT-2 (124M parameters)
- **Task**: Text Summarization
- **Dataset**: CNN/Daily Mail v3.0.0

### Training Details
- **Optimizer**: AdamW
- **Learning Rate**: Starting at 5e-5 with linear decay
- **Batch Size**: 2 per device (effective batch size 16 with gradient accumulation)
- **Gradient Accumulation Steps**: 8
- **Training Steps**: 23,400+
- **Epochs**: 3 (full dataset)
- **Hardware**: CUDA-compatible GPU

### Performance Metrics
#### ROUGE Scores Comparison
| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE-Lsum |
|-------|---------|---------|---------|------------|
| GPT-2 (zero-shot) | 18.75% | 1.14% | 13.56% | 16.32% |
| GPT-2 (fine-tuned) | 23.36% | 1.87% | 14.28% | 20.59% |

## Citation
If you use this work in your research, please cite:
```
@misc{mishra2023gpt2finetuning,
  author = {Mishra, Anurag},
  title = {GPT-2 Fine-tuning for Text Summarization with Mechanistic Interpretability Analysis},
  year = {2023},
  publisher = {GitHub},
  institution = {Rochester Institute of Technology},
  howpublished = {\url{https://github.com/anuragmishra/gpt2-summarization}}
}
```

## License
MIT License

## Notes
- The `checkpoints/` directory is gitignored to prevent large model files from being pushed to GitHub
- All metrics and logs are stored in their respective directories
- Visualizations are generated automatically during training and can be regenerated using `visualize_metrics.py` 
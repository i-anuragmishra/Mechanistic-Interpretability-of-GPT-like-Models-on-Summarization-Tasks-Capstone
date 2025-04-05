import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from transformers import (
    GPT2Tokenizer, 
    GPT2LMHeadModel, 
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling,
    TrainerCallback
)
from transformers import pipeline
from evaluate import load
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import hdbscan
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from torch.utils.tensorboard import SummaryWriter
import imageio.v2 as imageio  # Use imageio v2 to avoid deprecation warnings
from tqdm import tqdm
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import copy
import json
from datetime import datetime
from bertviz import model_view, head_view
import re

# Set seed for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

# Create directories for outputs
os.makedirs("model_outputs", exist_ok=True)
os.makedirs("visualizations", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("attention_analysis", exist_ok=True)
os.makedirs("latent_space_evolution", exist_ok=True)
os.makedirs("interpretability_metrics", exist_ok=True)

# Initialize tensorboard writer
writer = SummaryWriter(log_dir="logs")

# Load models and tokenizer
print("Loading GPT-2 model and tokenizer...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
off_the_shelf_model = GPT2LMHeadModel.from_pretrained("gpt2")

# Initialize the fine-tuned model from the pre-trained one
fine_tuned_model = GPT2LMHeadModel.from_pretrained("gpt2")

# GPT-2 doesn't have a padding token, so we'll set it to the EOS token
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"  # For generation tasks

# Load dataset
print("Loading CNN/Daily Mail dataset...")
dataset = load_dataset("cnn_dailymail", "3.0.0")
print(f"Dataset loaded: {dataset}")

# Prepare the dataset for fine-tuning
# Format: article + [sep token] + summary
def process_data_to_model_inputs(examples):
    # Combine article and summary with a separator
    inputs = [f"Article: {article}\nSummary: {summary}" 
              for article, summary in zip(examples["article"], examples["highlights"])]
    
    # Reduce max_length to save memory
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    
    # Create labels (same as inputs for language modeling)
    model_inputs["labels"] = model_inputs["input_ids"].copy()
    
    return model_inputs

# Process the datasets
print("Processing dataset...")
tokenized_datasets = dataset.map(
    process_data_to_model_inputs,
    batched=True,
    remove_columns=["article", "highlights", "id"],
)

# Create smaller datasets for testing
small_train_dataset = tokenized_datasets["train"].select(range(500))  # Small dataset for testing
small_val_dataset = tokenized_datasets["validation"].select(range(50))

# Medium dataset - larger than small but not as large as full
medium_train_dataset = tokenized_datasets["train"].select(range(5000))  # Medium dataset with 5,000 examples
medium_val_dataset = tokenized_datasets["validation"].select(range(500))  # Medium validation set

# For full fine-tuning use the complete dataset (commented out to save resources)
full_train_dataset = tokenized_datasets["train"]
full_val_dataset = tokenized_datasets["validation"]

# Data collator for language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # We're not doing masked language modeling
)

# Use this to switch between dataset sizes (small, medium, full)
# Choose which dataset to use: small, medium or full
use_dataset_size = "full"  # Changed from "medium" to "full"

if use_dataset_size == "small":
    train_dataset = small_train_dataset
    val_dataset = small_val_dataset
    epochs = 5
elif use_dataset_size == "medium":
    train_dataset = medium_train_dataset
    val_dataset = medium_val_dataset
    epochs = 5
else:  # "full"
    train_dataset = full_train_dataset
    val_dataset = full_val_dataset
    epochs = 3  # Fewer epochs for full dataset to save time

# Define sample text for attention analysis
SAMPLE_TEXTS = [
    "The president announced a new policy on climate change that aims to reduce carbon emissions by 50% by 2030.",
    "Scientists have discovered a new exoplanet that might have conditions suitable for life.",
    "The company reported record profits for the fourth quarter, exceeding analyst expectations."
]

# Define training arguments based on dataset size
if use_dataset_size == "full":
    # Use smaller batch size for full dataset to avoid memory issues
    per_device_batch_size = 2
    grad_accum_steps = 8  # Larger accumulation to compensate for smaller batch
else:
    per_device_batch_size = 4
    grad_accum_steps = 4

# Define training arguments
training_args = TrainingArguments(
    output_dir="model_outputs",
    overwrite_output_dir=True,
    num_train_epochs=epochs,
    per_device_train_batch_size=per_device_batch_size,
    per_device_eval_batch_size=per_device_batch_size,
    gradient_accumulation_steps=grad_accum_steps,  # Effective batch size remains similar
    evaluation_strategy="steps",
    eval_steps=200,  # More frequent evaluation
    save_steps=200,  # More frequent checkpoints to capture more data points
    logging_dir="logs",
    logging_steps=50,  # More frequent logging
    save_total_limit=20,  # Keep more checkpoints
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    greater_is_better=False,
    fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
    dataloader_pin_memory=False,
    optim="adamw_torch",
)

# Custom callback to capture metrics during training
class InterpretabilityCallback(TrainerCallback):
    def __init__(self, tokenizer, eval_dataset, sample_texts):
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.sample_texts = sample_texts
        self.metrics_history = {
            "attention_entropy": [],
            "attention_sparsity": [],
            "head_importance": [],
            "latent_representations": [],
            "gradient_norm": [],  # Track gradient norms
            "weight_change": [],  # Track weight changes
            "neuron_activations": [],  # Track activation patterns
            "steps": []
        }
        # Store initial weights for comparison
        self.initial_weights = {}
        
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """Capture initial weights at the beginning of training"""
        if model is None:
            return
        
        print("Capturing initial model weights...")
        # Store initial weights for key layers
        for name, param in model.named_parameters():
            if 'attn' in name and 'weight' in name:  # Focus on attention weights
                self.initial_weights[name] = param.detach().cpu().clone()
        
    def on_save(self, args, state, control, **kwargs):
        """Capture interpretability metrics when model is saved"""
        model = kwargs.get('model')
        if model is None:
            return
        
        print(f"\nCapturing interpretability metrics at step {state.global_step}...")
        
        # Save a copy of the current model state
        checkpoint_dir = f"checkpoints/checkpoint-{state.global_step}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        model.save_pretrained(checkpoint_dir)
        
        # Extract and analyze attention patterns
        attention_metrics = self.analyze_attention_patterns(model)
        
        # Extract latent representations
        latent_reps = self.extract_latent_representations(model)
        
        # Capture gradient norm
        grad_norm = 0.0
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm += param.grad.norm().item() ** 2
        grad_norm = grad_norm ** 0.5
        
        # Measure weight changes from initial state
        weight_changes = {}
        for name, param in model.named_parameters():
            if name in self.initial_weights:
                # Calculate L2 norm of weight changes
                weight_diff = param.detach().cpu() - self.initial_weights[name]
                weight_changes[name] = weight_diff.norm().item()
        
        # Track neuron activations on sample texts
        neuron_activations = self.analyze_neuron_activations(model)
        
        # Save metrics
        self.metrics_history["steps"].append(state.global_step)
        self.metrics_history["attention_entropy"].append(attention_metrics["entropy"])
        self.metrics_history["attention_sparsity"].append(attention_metrics["sparsity"])
        self.metrics_history["head_importance"].append(attention_metrics["head_importance"])
        self.metrics_history["latent_representations"].append(latent_reps)
        self.metrics_history["gradient_norm"].append(grad_norm)
        self.metrics_history["weight_change"].append(weight_changes)
        self.metrics_history["neuron_activations"].append(neuron_activations)
        
        # Helper function to recursively convert numpy arrays to lists
        def convert_numpy_to_lists(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_to_lists(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_to_lists(item) for item in obj]
            else:
                return obj
        
        # Save metrics to file
        with open(f"interpretability_metrics/metrics_step_{state.global_step}.json", "w") as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_metrics = copy.deepcopy(self.metrics_history)
            serializable_metrics = convert_numpy_to_lists(serializable_metrics)
            
            json.dump(serializable_metrics, f)
        
        return control
    
    def analyze_attention_patterns(self, model):
        """Analyze attention patterns for interpretability metrics"""
        model.eval()
        all_entropy = []
        all_sparsity = []
        all_head_importance = []
        
        with torch.no_grad():
            for text in self.sample_texts:
                inputs = self.tokenizer(text, return_tensors="pt").to(model.device)
                
                # Get attention weights
                outputs = model(**inputs, output_attentions=True)
                attentions = outputs.attentions  # Tuple of attention tensors
                
                # Calculate entropy and sparsity for each attention head
                for layer_idx, layer_attention in enumerate(attentions):
                    # layer_attention shape: [batch_size, num_heads, seq_len, seq_len]
                    for head_idx in range(layer_attention.size(1)):
                        # Get attention weights for this head
                        head_attention = layer_attention[0, head_idx].cpu().numpy()
                        
                        # Calculate entropy
                        # Add small epsilon to avoid log(0)
                        epsilon = 1e-10
                        log_attn = np.log2(head_attention + epsilon)
                        entropy = -np.sum(head_attention * log_attn, axis=-1).mean()
                        all_entropy.append(entropy)
                        
                        # Calculate sparsity (Gini coefficient)
                        sorted_attn = np.sort(head_attention.flatten())
                        n = sorted_attn.size
                        index = np.arange(1, n+1)
                        sparsity = 1 - 2 * np.sum((n + 1 - index) * sorted_attn) / (n * np.sum(sorted_attn))
                        all_sparsity.append(sparsity)
                        
                        # Simple head importance metric (based on attention diversity)
                        importance = entropy * (1 - sparsity)
                        all_head_importance.append(importance)
        
        return {
            "entropy": np.array(all_entropy).reshape(-1, len(self.sample_texts)),
            "sparsity": np.array(all_sparsity).reshape(-1, len(self.sample_texts)),
            "head_importance": np.array(all_head_importance).reshape(-1, len(self.sample_texts))
        }
    
    def analyze_neuron_activations(self, model):
        """Analyze activation patterns of neurons on sample inputs"""
        model.eval()
        activations = []
        
        with torch.no_grad():
            for text in self.sample_texts:
                inputs = self.tokenizer(text, return_tensors="pt").to(model.device)
                
                # Get hidden states
                outputs = model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states
                
                # For simplicity, focus on the output of the last layer
                last_layer = hidden_states[-1][0].cpu().numpy()  # [seq_len, hidden_dim]
                
                # Calculate statistics about neuron activations
                mean_activation = np.mean(last_layer, axis=0)
                max_activation = np.max(last_layer, axis=0)
                activation_std = np.std(last_layer, axis=0)
                
                activations.append({
                    "mean": mean_activation,
                    "max": max_activation,
                    "std": activation_std
                })
        
        return activations
    
    def extract_latent_representations(self, model):
        """Extract latent representations for visualization"""
        model.eval()
        # Take a small subset for visualization
        viz_dataset = self.eval_dataset.select(range(min(30, len(self.eval_dataset))))  # Increased from 20 to 30
        
        hidden_states = []
        with torch.no_grad():
            for sample in viz_dataset:
                # Process one sample at a time to save memory
                inputs = torch.tensor([sample["input_ids"]]).to(model.device)
                outputs = model(inputs, output_hidden_states=True)
                
                # Get the last hidden state
                last_hidden_state = outputs.hidden_states[-1]
                
                # Average over sequence length
                avg_hidden_state = last_hidden_state.mean(dim=1).cpu().numpy()
                hidden_states.append(avg_hidden_state[0])
                
                # Clear cache to save memory
                del outputs, last_hidden_state
                torch.cuda.empty_cache()
        
        return np.array(hidden_states)

# Initialize the callback
interpretability_callback = InterpretabilityCallback(tokenizer, val_dataset, SAMPLE_TEXTS)

# Initialize trainer with the callback
trainer = Trainer(
    model=fine_tuned_model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    callbacks=[interpretability_callback],
)

# Train the model
print("Fine-tuning the model...")
trainer.train()

# Save the fine-tuned model
model_save_path = "model_outputs/gpt2_cnn_dailymail"
fine_tuned_model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)
print(f"Model saved to {model_save_path}")

# Create visualizations of the metrics evolution
def create_attention_head_evolution_gif():
    """Create a GIF showing the evolution of attention head metrics"""
    print("Creating attention head evolution visualization...")
    
    # Load the saved metrics
    metrics_files = sorted([f for f in os.listdir("interpretability_metrics") if f.startswith("metrics_step_")])
    
    if not metrics_files:
        print("No metrics files found. Skipping attention head evolution visualization.")
        return
    
    steps = []
    entropy_values = []
    sparsity_values = []
    importance_values = []
    
    for metrics_file in metrics_files:
        with open(os.path.join("interpretability_metrics", metrics_file), "r") as f:
            metrics = json.load(f)
            steps.append(metrics["steps"][-1])
            entropy_values.append(np.array(metrics["attention_entropy"][-1]))
            sparsity_values.append(np.array(metrics["attention_sparsity"][-1]))
            importance_values.append(np.array(metrics["head_importance"][-1]))
    
    # Number of heads and layers
    num_layers = 12  # GPT-2 has 12 layers
    num_heads = 12   # GPT-2 has 12 heads per layer
    
    # Create a figure with subplots for each metric
    fig = plt.figure(figsize=(18, 12))
    
    frames = []
    for i in range(len(steps)):
        # Clear the figure
        plt.clf()
        
        # Reshape values for visualization
        entropy = entropy_values[i].reshape(num_layers * num_heads, -1).mean(axis=1).reshape(num_layers, num_heads)
        sparsity = sparsity_values[i].reshape(num_layers * num_heads, -1).mean(axis=1).reshape(num_layers, num_heads)
        importance = importance_values[i].reshape(num_layers * num_heads, -1).mean(axis=1).reshape(num_layers, num_heads)
        
        # Create subplots
        ax1 = plt.subplot(131)
        sns.heatmap(entropy, annot=False, fmt=".2f", cmap="viridis", ax=ax1)
        ax1.set_title(f"Attention Entropy (Step {steps[i]})")
        ax1.set_xlabel("Head")
        ax1.set_ylabel("Layer")
        
        ax2 = plt.subplot(132)
        sns.heatmap(sparsity, annot=False, fmt=".2f", cmap="Reds", ax=ax2)
        ax2.set_title(f"Attention Sparsity (Step {steps[i]})")
        ax2.set_xlabel("Head")
        ax2.set_ylabel("Layer")
        
        ax3 = plt.subplot(133)
        sns.heatmap(importance, annot=False, fmt=".2f", cmap="plasma", ax=ax3)
        ax3.set_title(f"Head Importance (Step {steps[i]})")
        ax3.set_xlabel("Head")
        ax3.set_ylabel("Layer")
        
        plt.suptitle(f"Attention Head Evolution - Step {steps[i]}", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save the frame
        frame_path = f"visualizations/attention_frame_{i}.png"
        plt.savefig(frame_path, dpi=100)
        frames.append(imageio.imread(frame_path))
    
    # Create a GIF
    imageio.mimsave("visualizations/attention_head_evolution.gif", frames, duration=1, loop=0)
    print("Attention head evolution GIF saved to visualizations/attention_head_evolution.gif")
    
    # Clean up individual frames
    for i in range(len(steps)):
        frame_path = f"visualizations/attention_frame_{i}.png"
        if os.path.exists(frame_path):
            os.remove(frame_path)

def create_3d_latent_space_evolution():
    """Create 3D visualization of latent space evolution"""
    print("Creating 3D latent space evolution visualization...")
    
    # Load the saved metrics
    metrics_files = sorted([f for f in os.listdir("interpretability_metrics") if f.startswith("metrics_step_")])
    
    if not metrics_files:
        print("No metrics files found. Skipping latent space evolution visualization.")
        return
    
    # Load all latent representations
    latent_reps_by_step = []
    steps = []
    
    for metrics_file in metrics_files:
        with open(os.path.join("interpretability_metrics", metrics_file), "r") as f:
            metrics = json.load(f)
            steps.append(metrics["steps"][-1])
            latent_reps_by_step.append(np.array(metrics["latent_representations"][-1]))
    
    # Apply UMAP for dimensionality reduction to 3D
    reducer = umap.UMAP(n_components=3, n_neighbors=5, min_dist=0.1, metric='cosine')
    
    # Concatenate all representations to fit the UMAP model
    all_reps = np.vstack(latent_reps_by_step)
    reduced_data = reducer.fit_transform(all_reps)
    
    # Split back by step
    reduced_by_step = []
    start_idx = 0
    for reps in latent_reps_by_step:
        end_idx = start_idx + reps.shape[0]
        reduced_by_step.append(reduced_data[start_idx:end_idx])
        start_idx = end_idx
    
    # Create an interactive 3D plot
    fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scatter3d'}]])
    
    colors = px.colors.qualitative.Plotly[:len(steps)]
    
    for i, (step, reduced) in enumerate(zip(steps, reduced_by_step)):
        fig.add_trace(
            go.Scatter3d(
                x=reduced[:, 0],
                y=reduced[:, 1],
                z=reduced[:, 2],
                mode='markers',
                marker=dict(
                    size=5,
                    color=colors[i],
                    opacity=0.8
                ),
                name=f"Step {step}"
            )
        )
    
    # Update layout
    fig.update_layout(
        title="3D Latent Space Evolution During Fine-tuning",
        scene=dict(
            xaxis_title="UMAP Dimension 1",
            yaxis_title="UMAP Dimension 2",
            zaxis_title="UMAP Dimension 3"
        ),
        width=1000,
        height=800
    )
    
    # Save as HTML for interactive visualization
    fig.write_html("visualizations/latent_space_3d_evolution.html")
    print("3D latent space evolution HTML saved to visualizations/latent_space_3d_evolution.html")
    
    # Create a GIF of rotating 3D plot
    frames = []
    for angle in range(0, 360, 5):
        camera = dict(
            eye=dict(
                x=1.25 * np.cos(np.radians(angle)),
                y=1.25 * np.sin(np.radians(angle)),
                z=0.5
            )
        )
        fig.update_layout(scene_camera=camera)
        frame_path = f"visualizations/latent_3d_frame_{angle}.png"
        fig.write_image(frame_path, width=800, height=600)
        frames.append(imageio.imread(frame_path))
    
    # Create a GIF
    imageio.mimsave("visualizations/latent_space_3d_rotation.gif", frames, duration=0.1, loop=0)
    print("3D latent space rotation GIF saved to visualizations/latent_space_3d_rotation.gif")
    
    # Clean up individual frames
    for angle in range(0, 360, 5):
        frame_path = f"visualizations/latent_3d_frame_{angle}.png"
        if os.path.exists(frame_path):
            os.remove(frame_path)

def visualize_attention_head_changes():
    """Create visualization showing how specific attention head weights change"""
    print("Creating attention head weights visualization...")
    
    # Load checkpoints
    checkpoint_dirs = sorted([d for d in os.listdir("checkpoints") if d.startswith("checkpoint-")])
    
    if not checkpoint_dirs:
        print("No checkpoint directories found. Skipping attention head weights visualization.")
        return
    
    steps = []
    head_weights_changes = []
    
    # Reference text for visualization
    viz_text = SAMPLE_TEXTS[0]
    inputs = tokenizer(viz_text, return_tensors="pt")
    
    # Extract weights from each checkpoint
    for checkpoint_dir in checkpoint_dirs:
        # Extract step number
        step = int(re.search(r'checkpoint-(\d+)', checkpoint_dir).group(1))
        steps.append(step)
        
        # Load model from checkpoint
        model_path = os.path.join("checkpoints", checkpoint_dir)
        model = GPT2LMHeadModel.from_pretrained(model_path)
        model.eval()
        
        # Get attention weights
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)
            attention = outputs.attentions
            
            # Extract weights for a specific layer and head (using layer 5, head 8 as an example)
            layer_idx, head_idx = 5, 8
            head_weights = attention[layer_idx][0, head_idx].cpu().numpy()
            head_weights_changes.append(head_weights)
    
    # Create a heatmap animation of the changing attention patterns
    tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
    
    # Create frames for GIF
    frames = []
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for i, (step, weights) in enumerate(zip(steps, head_weights_changes)):
        ax.clear()
        sns.heatmap(weights, cmap="YlOrRd", ax=ax)
        ax.set_title(f"Attention Weights Evolution - Layer 5, Head 8 (Step {step})")
        ax.set_xlabel("Token Position (Target)")
        ax.set_ylabel("Token Position (Source)")
        
        # Add token labels if sequence is not too long
        if len(tokens) <= 30:
            ax.set_xticks(np.arange(len(tokens)) + 0.5)
            ax.set_yticks(np.arange(len(tokens)) + 0.5)
            ax.set_xticklabels(tokens, rotation=90)
            ax.set_yticklabels(tokens)
        
        plt.tight_layout()
        
        # Save the frame
        frame_path = f"visualizations/attention_weights_frame_{i}.png"
        plt.savefig(frame_path, dpi=100)
        frames.append(imageio.imread(frame_path))
    
    # Create a GIF
    imageio.mimsave("visualizations/attention_weights_evolution.gif", frames, duration=0.5, loop=0)
    print("Attention weights evolution GIF saved to visualizations/attention_weights_evolution.gif")
    
    # Clean up individual frames
    for i in range(len(steps)):
        frame_path = f"visualizations/attention_weights_frame_{i}.png"
        if os.path.exists(frame_path):
            os.remove(frame_path)

def create_temporal_metrics_visualization():
    """Create a line chart showing how different metrics change over time"""
    print("Creating temporal metrics visualization...")
    
    # Load the saved metrics
    metrics_files = sorted([f for f in os.listdir("interpretability_metrics") if f.startswith("metrics_step_")])
    
    if not metrics_files:
        print("No metrics files found. Skipping temporal metrics visualization.")
        return
    
    steps = []
    avg_entropy = []
    avg_sparsity = []
    avg_importance = []
    
    for metrics_file in metrics_files:
        with open(os.path.join("interpretability_metrics", metrics_file), "r") as f:
            metrics = json.load(f)
            steps.append(metrics["steps"][-1])
            # Calculate average metrics across all heads
            avg_entropy.append(np.mean(np.array(metrics["attention_entropy"][-1])))
            avg_sparsity.append(np.mean(np.array(metrics["attention_sparsity"][-1])))
            avg_importance.append(np.mean(np.array(metrics["head_importance"][-1])))
    
    # Create an interactive plot
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=steps,
        y=avg_entropy,
        mode='lines+markers',
        name='Average Entropy',
        line=dict(color='blue', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=steps,
        y=avg_sparsity,
        mode='lines+markers',
        name='Average Sparsity',
        line=dict(color='red', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=steps,
        y=avg_importance,
        mode='lines+markers',
        name='Average Importance',
        line=dict(color='green', width=2)
    ))
    
    # Update layout
    fig.update_layout(
        title="Evolution of Attention Metrics During Fine-tuning",
        xaxis_title="Training Steps",
        yaxis_title="Metric Value",
        hovermode="x unified",
        template="plotly_white"
    )
    
    # Save as HTML
    fig.write_html("visualizations/temporal_metrics_evolution.html")
    print("Temporal metrics evolution HTML saved to visualizations/temporal_metrics_evolution.html")
    
    # Also create a static image
    fig.write_image("visualizations/temporal_metrics_evolution.png", width=1000, height=600)
    print("Temporal metrics evolution PNG saved to visualizations/temporal_metrics_evolution.png")

# Run all visualization functions
create_attention_head_evolution_gif()
create_3d_latent_space_evolution()
visualize_attention_head_changes()
create_temporal_metrics_visualization()

# Add new visualization functions to display weight changes and neuron activations
def visualize_weight_changes():
    """Create visualization of weight changes over time"""
    print("Creating weight changes visualization...")
    
    # Load the saved metrics
    metrics_files = sorted([f for f in os.listdir("interpretability_metrics") if f.startswith("metrics_step_")])
    
    if not metrics_files:
        print("No metrics files found. Skipping weight changes visualization.")
        return
    
    steps = []
    weight_changes = {}
    
    for metrics_file in metrics_files:
        with open(os.path.join("interpretability_metrics", metrics_file), "r") as f:
            metrics = json.load(f)
            step = metrics["steps"][-1]
            steps.append(step)
            
            # Process weight changes - check if key exists
            if "weight_change" in metrics and len(metrics["weight_change"]) > 0:
                changes = metrics["weight_change"][-1]
                for weight_name, change_val in changes.items():
                    if weight_name not in weight_changes:
                        weight_changes[weight_name] = []
                    weight_changes[weight_name].append(change_val)
    
    # Skip if no weight changes data found
    if not weight_changes:
        print("No weight changes data found. Skipping weight changes visualization.")
        return
        
    # Create an interactive plot
    fig = go.Figure()
    
    # Plot weight changes for each attention layer
    for weight_name, changes in weight_changes.items():
        # Skip if we don't have enough data points
        if len(changes) != len(steps):
            continue
            
        # Create a shorter display name
        display_name = weight_name.split('.')[-3:]
        display_name = '.'.join(display_name)
        
        fig.add_trace(go.Scatter(
            x=steps,
            y=changes,
            mode='lines+markers',
            name=display_name,
            line=dict(width=2)
        ))
    
    # Update layout
    fig.update_layout(
        title="Weight Changes During Fine-tuning",
        xaxis_title="Training Steps",
        yaxis_title="L2 Norm of Weight Change",
        hovermode="x unified",
        template="plotly_white"
    )
    
    # Save as HTML
    fig.write_html("visualizations/weight_changes_evolution.html")
    print("Weight changes evolution HTML saved to visualizations/weight_changes_evolution.html")
    
    # Also create a static image
    fig.write_image("visualizations/weight_changes_evolution.png", width=1000, height=600)
    print("Weight changes evolution PNG saved to visualizations/weight_changes_evolution.png")

def visualize_gradient_norm():
    """Create visualization of gradient norm changes over time"""
    print("Creating gradient norm visualization...")
    
    # Load the saved metrics
    metrics_files = sorted([f for f in os.listdir("interpretability_metrics") if f.startswith("metrics_step_")])
    
    if not metrics_files:
        print("No metrics files found. Skipping gradient norm visualization.")
        return
    
    steps = []
    grad_norms = []
    
    for metrics_file in metrics_files:
        with open(os.path.join("interpretability_metrics", metrics_file), "r") as f:
            metrics = json.load(f)
            steps.append(metrics["steps"][-1])
            # Check if gradient_norm key exists
            if "gradient_norm" in metrics and len(metrics["gradient_norm"]) > 0:
                grad_norms.append(metrics["gradient_norm"][-1])
    
    # Skip if no gradient data
    if not grad_norms or len(grad_norms) != len(steps):
        print("No gradient norm data found or incomplete data. Skipping gradient norm visualization.")
        return
        
    # Create an interactive plot
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=steps,
        y=grad_norms,
        mode='lines+markers',
        name='Gradient Norm',
        line=dict(color='purple', width=2)
    ))
    
    # Update layout
    fig.update_layout(
        title="Gradient Norm Evolution During Fine-tuning",
        xaxis_title="Training Steps",
        yaxis_title="Gradient L2 Norm",
        template="plotly_white"
    )
    
    # Save as HTML
    fig.write_html("visualizations/gradient_norm_evolution.html")
    print("Gradient norm evolution HTML saved to visualizations/gradient_norm_evolution.html")
    
    # Also create a static image
    fig.write_image("visualizations/gradient_norm_evolution.png", width=1000, height=600)
    print("Gradient norm evolution PNG saved to visualizations/gradient_norm_evolution.png")

def visualize_neuron_activations():
    """Create visualization of neuron activations over time"""
    print("Creating neuron activations visualization...")
    
    # Load the saved metrics
    metrics_files = sorted([f for f in os.listdir("interpretability_metrics") if f.startswith("metrics_step_")])
    
    if not metrics_files:
        print("No metrics files found. Skipping neuron activations visualization.")
        return
    
    # Load activation data
    steps = []
    mean_activations = []
    max_activations = []
    std_activations = []
    
    for metrics_file in metrics_files:
        with open(os.path.join("interpretability_metrics", metrics_file), "r") as f:
            metrics = json.load(f)
            step = metrics["steps"][-1]
            
            # Check if neuron_activations key exists and has data
            if "neuron_activations" in metrics and len(metrics["neuron_activations"]) > 0 and len(metrics["neuron_activations"][-1]) > 0:
                steps.append(step)
                # Get neuron activation stats for first sample text
                activations = metrics["neuron_activations"][-1][0]  # First sample text
                mean_activations.append(np.array(activations["mean"]))
                max_activations.append(np.array(activations["max"]))
                std_activations.append(np.array(activations["std"]))
    
    # Skip if we don't have enough data
    if not mean_activations:
        print("No activation data found. Skipping neuron activations visualization.")
        return
    
    # Create heatmap animation showing neuron activations changing over time
    num_neurons = 20  # Just visualize first 20 neurons to keep it manageable
    
    # Function to create a heatmap for a specific step
    def create_heatmap(step_idx):
        mean_act = mean_activations[step_idx][:num_neurons]
        max_act = max_activations[step_idx][:num_neurons]
        std_act = std_activations[step_idx][:num_neurons]
        
        # Stack them for visualization
        data = np.vstack([mean_act, max_act, std_act])
        
        fig, ax = plt.subplots(figsize=(12, 6))
        im = ax.imshow(data, cmap='viridis', aspect='auto')
        
        # Add labels
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(['Mean', 'Max', 'Std'])
        ax.set_xlabel('Neuron Index')
        ax.set_title(f'Neuron Activation Patterns (Step {steps[step_idx]})')
        
        plt.colorbar(im, ax=ax, label='Activation Value')
        plt.tight_layout()
        
        return fig
    
    # Create frames for GIF
    frames = []
    for i in range(len(steps)):
        fig = create_heatmap(i)
        
        # Save the frame
        frame_path = f"visualizations/neuron_activations_frame_{i}.png"
        plt.savefig(frame_path, dpi=100)
        plt.close(fig)
        frames.append(imageio.imread(frame_path))
    
    # Create a GIF
    imageio.mimsave("visualizations/neuron_activations_evolution.gif", frames, duration=1, loop=0)
    print("Neuron activations evolution GIF saved to visualizations/neuron_activations_evolution.gif")
    
    # Clean up individual frames
    for i in range(len(steps)):
        frame_path = f"visualizations/neuron_activations_frame_{i}.png"
        if os.path.exists(frame_path):
            os.remove(frame_path)

# Run the new visualization functions
visualize_weight_changes()
visualize_gradient_norm()
visualize_neuron_activations()

# Evaluation functions
def generate_summary(article, model, max_new_tokens=150):
    # Truncate the article if it's too long to avoid position embedding issues
    # GPT-2's max position embeddings is 1024, so we'll limit to 800 to be safe
    inputs = tokenizer("Article: " + article + "\nSummary:", truncation=True, max_length=800, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate summary
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        num_beams=2,  # Reduce beam size to save memory
        no_repeat_ngram_size=2,
        early_stopping=True
    )
    
    # Decode and return the summary
    summary = tokenizer.decode(output[0], skip_special_tokens=True)
    # Extract only the summary part
    if "Summary:" in summary:
        summary = summary.split("Summary:")[1].strip()
    return summary

# Rouge metric
rouge = load("rouge")

def evaluate_model(model, eval_dataset, num_samples=5):  # Reduced samples to save time
    model.eval()
    
    # Select random samples for evaluation
    indices = np.random.choice(len(eval_dataset), num_samples, replace=False)
    samples = eval_dataset.select(indices)
    
    generated_summaries = []
    reference_summaries = []
    
    for sample in samples:
        article = sample["article"]
        reference = sample["highlights"]
        
        generated_summary = generate_summary(article, model)
        
        generated_summaries.append(generated_summary)
        reference_summaries.append(reference)
    
    # Calculate ROUGE scores
    results = rouge.compute(predictions=generated_summaries, references=reference_summaries)
    
    return {
        "samples": list(zip(generated_summaries, reference_summaries)),
        "rouge_scores": results
    }

# Compare models
print("Comparing models...")
test_samples = dataset["test"].select(range(10))  # Increased from 5 to 10

print("Evaluating off-the-shelf GPT-2...")
off_the_shelf_results = evaluate_model(off_the_shelf_model, test_samples)

print("Evaluating fine-tuned GPT-2...")
fine_tuned_results = evaluate_model(fine_tuned_model, test_samples)

print("\nOff-the-shelf GPT-2 ROUGE scores:")
print(off_the_shelf_results["rouge_scores"])

print("\nFine-tuned GPT-2 ROUGE scores:")
print(fine_tuned_results["rouge_scores"])

# Create a report HTML file summarizing all findings
def create_html_report():
    print("Creating HTML report...")
    
    # Check if we have ROUGE scores
    try:
        rouge1_off = off_the_shelf_results["rouge_scores"]["rouge1"]
        rouge2_off = off_the_shelf_results["rouge_scores"]["rouge2"]
        rougeL_off = off_the_shelf_results["rouge_scores"]["rougeL"]
        
        rouge1_ft = fine_tuned_results["rouge_scores"]["rouge1"]
        rouge2_ft = fine_tuned_results["rouge_scores"]["rouge2"]
        rougeL_ft = fine_tuned_results["rouge_scores"]["rougeL"]
    except:
        # If ROUGE scores are not available
        rouge1_off = rouge2_off = rougeL_off = 0
        rouge1_ft = rouge2_ft = rougeL_ft = 0
    
    # Check which visualizations were created
    has_attn_head_gif = os.path.exists("visualizations/attention_head_evolution.gif")
    has_latent_3d_gif = os.path.exists("visualizations/latent_space_3d_rotation.gif")
    has_latent_3d_html = os.path.exists("visualizations/latent_space_3d_evolution.html")
    has_attn_weights_gif = os.path.exists("visualizations/attention_weights_evolution.gif")
    has_temporal_metrics_png = os.path.exists("visualizations/temporal_metrics_evolution.png")
    has_temporal_metrics_html = os.path.exists("visualizations/temporal_metrics_evolution.html")
    has_weight_changes_png = os.path.exists("visualizations/weight_changes_evolution.png")
    has_weight_changes_html = os.path.exists("visualizations/weight_changes_evolution.html")
    has_gradient_norm_png = os.path.exists("visualizations/gradient_norm_evolution.png")
    has_gradient_norm_html = os.path.exists("visualizations/gradient_norm_evolution.html")
    has_neuron_activations_gif = os.path.exists("visualizations/neuron_activations_evolution.gif")
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>GPT-2 Fine-tuning Interpretability Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
            h1, h2, h3 {{ color: #444; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            .visualization {{ margin: 20px 0; text-align: center; }}
            .metrics {{ margin: 20px 0; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .footer {{ margin-top: 30px; font-size: 0.8em; color: #777; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>GPT-2 Fine-tuning Interpretability Report</h1>
            <p>Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            
            """
    
    # Add sections only if visualizations exist
    if has_attn_head_gif:
        html_content += """
            <h2>Attention Head Evolution</h2>
            <div class="visualization">
                <img src="attention_head_evolution.gif" alt="Attention Head Evolution" width="800">
                <p>Visualization showing how attention head metrics (entropy, sparsity, importance) evolve during training.</p>
            </div>
            """
    
    if has_latent_3d_gif:
        html_content += f"""
            <h2>Latent Space Evolution</h2>
            <div class="visualization">
                <img src="latent_space_3d_rotation.gif" alt="3D Latent Space Evolution" width="800">
                <p>3D visualization of how the latent space evolves during fine-tuning. Each color represents a different training step.</p>
                {"<p><a href='latent_space_3d_evolution.html' target='_blank'>Interactive 3D Visualization</a></p>" if has_latent_3d_html else ""}
            </div>
            """
    
    if has_attn_weights_gif:
        html_content += """
            <h2>Attention Weights Evolution</h2>
            <div class="visualization">
                <img src="attention_weights_evolution.gif" alt="Attention Weights Evolution" width="800">
                <p>Visualization showing how attention weights for Layer 5, Head 8 change during fine-tuning.</p>
            </div>
            """
    
    if has_temporal_metrics_png:
        html_content += f"""
            <h2>Temporal Metrics Evolution</h2>
            <div class="visualization">
                <img src="temporal_metrics_evolution.png" alt="Temporal Metrics Evolution" width="800">
                <p>Line chart showing how different attention metrics change over training steps.</p>
                {"<p><a href='temporal_metrics_evolution.html' target='_blank'>Interactive Metrics Visualization</a></p>" if has_temporal_metrics_html else ""}
            </div>
            """
    
    if has_weight_changes_png:
        html_content += f"""
            <h2>Weight Changes</h2>
            <div class="visualization">
                <img src="weight_changes_evolution.png" alt="Weight Changes Evolution" width="800">
                <p>Visualization of how weights in attention layers change during fine-tuning (measured as L2 norm of difference from initial weights).</p>
                {"<p><a href='weight_changes_evolution.html' target='_blank'>Interactive Weight Changes Visualization</a></p>" if has_weight_changes_html else ""}
            </div>
            """
    
    if has_gradient_norm_png:
        html_content += f"""
            <h2>Gradient Norm Evolution</h2>
            <div class="visualization">
                <img src="gradient_norm_evolution.png" alt="Gradient Norm Evolution" width="800">
                <p>Visualization of how the gradient norm changes during training, showing the magnitude of weight updates.</p>
                {"<p><a href='gradient_norm_evolution.html' target='_blank'>Interactive Gradient Norm Visualization</a></p>" if has_gradient_norm_html else ""}
            </div>
            """
    
    if has_neuron_activations_gif:
        html_content += """
            <h2>Neuron Activation Patterns</h2>
            <div class="visualization">
                <img src="neuron_activations_evolution.gif" alt="Neuron Activations Evolution" width="800">
                <p>Visualization showing how neuron activation patterns change during fine-tuning.</p>
            </div>
            """
    
    html_content += f"""
            <h2>Model Performance Comparison</h2>
            <div class="metrics">
                <h3>Off-the-shelf GPT-2 ROUGE Scores</h3>
                <p>ROUGE-1: {rouge1_off:.4f}</p>
                <p>ROUGE-2: {rouge2_off:.4f}</p>
                <p>ROUGE-L: {rougeL_off:.4f}</p>
                
                <h3>Fine-tuned GPT-2 ROUGE Scores</h3>
                <p>ROUGE-1: {rouge1_ft:.4f}</p>
                <p>ROUGE-2: {rouge2_ft:.4f}</p>
                <p>ROUGE-L: {rougeL_ft:.4f}</p>
            </div>
            
            <div class="footer">
                <p>Generated using the GPT-2 Fine-tuning Interpretability Script</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    with open("visualizations/interpretability_report.html", "w") as f:
        f.write(html_content)
    
    print("HTML report saved to visualizations/interpretability_report.html")

# Create the HTML report
create_html_report()

print("All visualizations and analyses completed!")

# Final message
print("""
Mechanistic Interpretability Analysis Complete!

The following artifacts have been generated:
1. Attention head evolution GIF: visualizations/attention_head_evolution.gif
2. 3D latent space rotation GIF: visualizations/latent_space_3d_rotation.gif
3. 3D latent space interactive HTML: visualizations/latent_space_3d_evolution.html
4. Attention weights evolution GIF: visualizations/attention_weights_evolution.gif
5. Temporal metrics evolution HTML: visualizations/temporal_metrics_evolution.html
6. Weight changes evolution HTML: visualizations/weight_changes_evolution.html
7. Gradient norm evolution HTML: visualizations/gradient_norm_evolution.html
8. Neuron activations evolution GIF: visualizations/neuron_activations_evolution.gif
9. Complete HTML report: visualizations/interpretability_report.html

Model checkpoints are saved in the 'checkpoints' directory, and all metrics
are stored in 'interpretability_metrics' for further analysis.
""") 
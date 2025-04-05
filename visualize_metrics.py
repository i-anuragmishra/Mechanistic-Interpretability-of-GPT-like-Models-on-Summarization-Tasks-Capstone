import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import imageio.v2 as imageio
from tqdm import tqdm
import json
from datetime import datetime
import re
import umap
import hdbscan

print("Generating visualizations from recorded metrics...")

# Load tokenizer for any token-level visualizations
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Sample texts that were used during training
SAMPLE_TEXTS = [
    "The president announced a new policy on climate change that aims to reduce carbon emissions by 50% by 2030.",
    "Scientists have discovered a new exoplanet that might have conditions suitable for life.",
    "The company reported record profits for the fourth quarter, exceeding analyst expectations."
]

# Ensure output directories exist
os.makedirs("visualizations", exist_ok=True)

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
    try:
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
                        color=colors[i % len(colors)],  # Use modulo to avoid index errors
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
    except Exception as e:
        print(f"Error creating 3D latent space visualization: {e}")

def visualize_attention_head_changes():
    """Create visualization showing how specific attention head weights change"""
    print("Creating attention head weights visualization...")
    
    # Load checkpoints
    checkpoint_dirs = sorted([d for d in os.listdir("checkpoints") if d.startswith("checkpoint-")])
    
    if not checkpoint_dirs:
        print("No checkpoint directories found. Skipping attention head weights visualization.")
        return
    
    # Load model to device based on availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        steps = []
        head_weights_changes = []
        
        # Reference text for visualization
        viz_text = SAMPLE_TEXTS[0]
        inputs = tokenizer(viz_text, return_tensors="pt").to(device)
        
        # Extract weights from each checkpoint
        for checkpoint_dir in checkpoint_dirs:
            # Extract step number
            step = int(re.search(r'checkpoint-(\d+)', checkpoint_dir).group(1))
            steps.append(step)
            
            # Load model from checkpoint
            model_path = os.path.join("checkpoints", checkpoint_dir)
            model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
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
    except Exception as e:
        print(f"Error creating attention head changes visualization: {e}")

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

def create_html_report():
    print("Creating HTML report...")
    
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
            <p>These visualizations have been created from metrics captured during the fine-tuning process before the server connection was closed.</p>
            
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
            <div class="footer">
                <p>Generated using the GPT-2 Fine-tuning Interpretability Visualization Script</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    with open("visualizations/interpretability_report.html", "w") as f:
        f.write(html_content)
    
    print("HTML report saved to visualizations/interpretability_report.html")

# Run all visualization functions
try:
    print("Creating visualizations from recorded metrics...")
    create_attention_head_evolution_gif()
    create_3d_latent_space_evolution()
    visualize_attention_head_changes()
    create_temporal_metrics_visualization()
    visualize_weight_changes()
    visualize_gradient_norm()
    visualize_neuron_activations()
    create_html_report()
    print("All visualizations and analyses completed!")
except Exception as e:
    print(f"Error during visualization creation: {e}")

# Final message
print("""
Visualizations from Recorded Metrics Complete!

The following artifacts have been generated (if data was available):
1. Attention head evolution GIF: visualizations/attention_head_evolution.gif
2. 3D latent space rotation GIF: visualizations/latent_space_3d_rotation.gif
3. 3D latent space interactive HTML: visualizations/latent_space_3d_evolution.html
4. Attention weights evolution GIF: visualizations/attention_weights_evolution.gif
5. Temporal metrics evolution HTML: visualizations/temporal_metrics_evolution.html
6. Weight changes evolution HTML: visualizations/weight_changes_evolution.html
7. Gradient norm evolution HTML: visualizations/gradient_norm_evolution.html
8. Neuron activations evolution GIF: visualizations/neuron_activations_evolution.gif
9. Complete HTML report: visualizations/interpretability_report.html
""") 
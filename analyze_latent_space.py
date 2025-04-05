import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import hdbscan
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import argparse

# Create directories for outputs
os.makedirs("visualizations", exist_ok=True)

def load_model_and_tokenizer(model_path):
    """Load the fine-tuned model and tokenizer"""
    if os.path.exists(model_path):
        print(f"Loading fine-tuned model from {model_path}...")
        model = GPT2LMHeadModel.from_pretrained(model_path)
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    else:
        print(f"Model path {model_path} not found. Using base GPT-2...")
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    # Set padding token
    tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def load_test_data(num_samples=100):
    """Load test data from CNN/Daily Mail dataset"""
    dataset = load_dataset("cnn_dailymail", "3.0.0")
    test_data = dataset["test"]
    
    if len(test_data) > num_samples:
        indices = np.random.choice(len(test_data), num_samples, replace=False)
        test_data = test_data.select(indices)
    
    return test_data

def prepare_inputs(examples, tokenizer):
    """Prepare inputs for the model"""
    inputs = [f"Article: {article}\nSummary: {summary}" 
             for article, summary in zip(examples["article"], examples["highlights"])]
    
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True, padding="max_length", return_tensors="pt")
    
    return model_inputs

def extract_hidden_states(model, tokenized_inputs, layers=None):
    """Extract hidden states from specified layers"""
    model.eval()
    
    if layers is None:
        # Use all layers by default
        layers = list(range(model.config.n_layer))
    
    # Dict to store hidden states for each layer
    hidden_states_by_layer = {layer: [] for layer in layers}
    
    with torch.no_grad():
        for i in range(0, len(tokenized_inputs["input_ids"]), 4):  # Process in small batches
            batch = {k: v[i:i+4] for k, v in tokenized_inputs.items()}
            outputs = model(**batch, output_hidden_states=True)
            
            # Extract hidden states from specified layers
            for layer in layers:
                layer_hidden_states = outputs.hidden_states[layer + 1]  # +1 because 0 is embeddings
                # Average over sequence length to get a single vector per example
                avg_hidden_states = layer_hidden_states.mean(dim=1).cpu().numpy()
                hidden_states_by_layer[layer].extend(avg_hidden_states)
    
    return hidden_states_by_layer

def reduce_dimensionality(hidden_states, method="pca", n_components=2):
    """Apply dimensionality reduction"""
    if method == "pca":
        reducer = PCA(n_components=n_components)
    elif method == "tsne":
        reducer = TSNE(n_components=n_components, perplexity=30, n_iter=1000)
    elif method == "umap":
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=n_components)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    reduced_data = reducer.fit_transform(hidden_states)
    return reduced_data

def find_optimal_clusters(data, max_clusters=10):
    """Find optimal number of clusters using silhouette score"""
    silhouette_scores = []
    
    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(data)
        
        score = silhouette_score(data, cluster_labels)
        silhouette_scores.append(score)
        print(f"Clusters: {n_clusters}, Silhouette Score: {score:.4f}")
    
    best_n_clusters = np.argmax(silhouette_scores) + 2  # +2 because we start from 2 clusters
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score for Different Cluster Numbers')
    plt.savefig("visualizations/optimal_clusters.png", dpi=300, bbox_inches="tight")
    
    return best_n_clusters

def cluster_data(data, method="kmeans", n_clusters=None):
    """Cluster the data"""
    if method == "kmeans":
        if n_clusters is None:
            n_clusters = find_optimal_clusters(data)
        
        clusterer = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = clusterer.fit_predict(data)
    elif method == "hdbscan":
        clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=3)
        cluster_labels = clusterer.fit_predict(data)
    else:
        raise ValueError(f"Unknown clustering method: {method}")
    
    return cluster_labels

def visualize_2d(data_2d, cluster_labels, method="pca", layer=None, save_path=None):
    """Create 2D visualization of the data"""
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(data_2d[:, 0], data_2d[:, 1], 
                         c=cluster_labels, cmap="viridis", alpha=0.7, s=50)
    
    plt.colorbar(scatter, label="Cluster")
    title = f"Latent Space Visualization using {method.upper()}"
    if layer is not None:
        title += f" - Layer {layer}"
    plt.title(title)
    plt.xlabel(f"{method.upper()} Dimension 1")
    plt.ylabel(f"{method.upper()} Dimension 2")
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    # Interactive visualization with Plotly
    fig = px.scatter(
        x=data_2d[:, 0], 
        y=data_2d[:, 1],
        color=cluster_labels,
        title=title,
        labels={"x": f"{method.upper()} Dimension 1", "y": f"{method.upper()} Dimension 2"}
    )
    
    if save_path:
        interactive_path = save_path.replace(".png", "_interactive.html")
        fig.write_html(interactive_path)

def visualize_3d(data_3d, cluster_labels, method="pca", layer=None, save_path=None):
    """Create 3D visualization of the data"""
    fig = px.scatter_3d(
        x=data_3d[:, 0], 
        y=data_3d[:, 1], 
        z=data_3d[:, 2],
        color=cluster_labels,
        title=f"3D Latent Space Visualization using {method.upper()}" + (f" - Layer {layer}" if layer is not None else ""),
        labels={"x": f"{method.upper()} Dimension 1", 
                "y": f"{method.upper()} Dimension 2", 
                "z": f"{method.upper()} Dimension 3"}
    )
    
    if save_path:
        fig.write_html(save_path)

def analyze_layer_differences(hidden_states_by_layer, reduction_method="pca"):
    """Analyze and visualize differences between layers"""
    # Reduce dimensionality for each layer
    reduced_data_by_layer = {}
    for layer, states in hidden_states_by_layer.items():
        reduced_data_by_layer[layer] = reduce_dimensionality(states, method=reduction_method)
    
    # Calculate distances between consecutive layers
    distances = []
    layers = sorted(hidden_states_by_layer.keys())
    
    for i in range(len(layers) - 1):
        layer1 = layers[i]
        layer2 = layers[i + 1]
        
        # Calculate average Euclidean distance
        dist = np.mean([np.linalg.norm(reduced_data_by_layer[layer1][j] - reduced_data_by_layer[layer2][j]) 
                       for j in range(len(reduced_data_by_layer[layer1]))])
        distances.append((layer1, layer2, dist))
    
    # Visualize distances
    plt.figure(figsize=(12, 6))
    plt.bar([f"{l1}-{l2}" for l1, l2, _ in distances], [d for _, _, d in distances])
    plt.xlabel("Layer Transition")
    plt.ylabel("Average Euclidean Distance")
    plt.title("Distances Between Consecutive Layers' Representations")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"visualizations/layer_distances_{reduction_method}.png", dpi=300, bbox_inches="tight")
    
    return distances

def main():
    parser = argparse.ArgumentParser(description="Analyze latent space of GPT-2 for summarization")
    parser.add_argument("--model_path", type=str, default="model_outputs/gpt2_cnn_dailymail",
                        help="Path to fine-tuned model")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Number of samples to analyze")
    parser.add_argument("--reduction_method", type=str, default="pca", choices=["pca", "tsne", "umap"],
                        help="Dimensionality reduction method")
    parser.add_argument("--clustering_method", type=str, default="kmeans", choices=["kmeans", "hdbscan"],
                        help="Clustering method")
    parser.add_argument("--selected_layers", type=str, default=None,
                        help="Comma-separated list of layers to analyze, e.g., '0,5,11'")
    args = parser.parse_args()
    
    # Load model and data
    model, tokenizer = load_model_and_tokenizer(args.model_path)
    test_data = load_test_data(args.num_samples)
    tokenized_inputs = prepare_inputs(test_data, tokenizer)
    
    # Determine which layers to analyze
    if args.selected_layers:
        selected_layers = [int(l) for l in args.selected_layers.split(",")]
    else:
        selected_layers = [0, 5, 11]  # Default: first, middle, last layers
    
    # Extract hidden states
    hidden_states_by_layer = extract_hidden_states(model, tokenized_inputs, layers=selected_layers)
    
    # Analyze each layer
    for layer, states in hidden_states_by_layer.items():
        print(f"Analyzing layer {layer}...")
        
        # 2D visualization
        data_2d = reduce_dimensionality(states, method=args.reduction_method, n_components=2)
        cluster_labels = cluster_data(data_2d, method=args.clustering_method)
        visualize_2d(data_2d, cluster_labels, method=args.reduction_method, layer=layer,
                    save_path=f"visualizations/layer_{layer}_{args.reduction_method}_2d.png")
        
        # 3D visualization
        data_3d = reduce_dimensionality(states, method=args.reduction_method, n_components=3)
        visualize_3d(data_3d, cluster_labels, method=args.reduction_method, layer=layer,
                    save_path=f"visualizations/layer_{layer}_{args.reduction_method}_3d.html")
    
    # Analyze differences between layers
    if len(hidden_states_by_layer) > 1:
        print("Analyzing differences between layers...")
        distances = analyze_layer_differences(hidden_states_by_layer, reduction_method=args.reduction_method)
        
        # Print layer differences
        print("\nDistances between consecutive layers:")
        for l1, l2, dist in distances:
            print(f"Layers {l1} to {l2}: {dist:.4f}")
    
    print("\nLatent space analysis completed. Visualizations saved to 'visualizations/' directory.")

if __name__ == "__main__":
    main() 
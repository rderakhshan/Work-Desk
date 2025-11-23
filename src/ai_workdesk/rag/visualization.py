"""Visualization module for embedding analysis and projection."""

from typing import List, Tuple, Optional
import numpy as np
from loguru import logger

class EmbeddingVisualizer:
    """Visualize embeddings in 2D/3D space."""
    
    def __init__(self):
        """Initialize the visualizer."""
        pass
    
    def project_embeddings_2d(
        self, 
        embeddings: np.ndarray,
        labels: Optional[List[str]] = None,
        method: str = "umap"
    ) -> Tuple[np.ndarray, str]:
        """
        Project high-dimensional embeddings to 2D.
        
        Args:
            embeddings: Array of shape (n_samples, n_features)
            labels: Optional labels for each embedding
            method: Projection method ("umap" or "tsne")
            
        Returns:
            Tuple of (projected_embeddings, html_plot)
        """
        try:
            if method == "umap":
                from umap import UMAP
                reducer = UMAP(n_components=2, random_state=42)
            else:  # tsne
                from sklearn.manifold import TSNE
                reducer = TSNE(n_components=2, random_state=42)
            
            projected = reducer.fit_transform(embeddings)
            
            # Create interactive plot
            import plotly.graph_objects as go
            
            fig = go.Figure(data=[go.Scatter(
                x=projected[:, 0],
                y=projected[:, 1],
                mode='markers',
                marker=dict(
                    size=8,
                    color=list(range(len(projected))),
                    colorscale='Viridis',
                    showscale=True
                ),
                text=labels if labels else [f"Doc {i}" for i in range(len(projected))],
                hovertemplate='<b>%{text}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>'
            )])
            
            fig.update_layout(
                title=f"Embedding Projection ({method.upper()})",
                xaxis_title="Dimension 1",
                yaxis_title="Dimension 2",
                hovermode='closest',
                template="plotly_white",
                height=600,
                width=900
            )
            
            html_plot = fig.to_html(include_plotlyjs='cdn', full_html=False, div_id="plotly-viz")
            # Wrap in div for proper rendering in Gradio
            wrapped_html = f'<div style="width:100%; height:600px; overflow:auto;">{html_plot}</div>'
            logger.info(f"Projected {len(embeddings)} embeddings to 2D using {method}")
            return projected, wrapped_html
            
        except Exception as e:
            logger.error(f"Error projecting embeddings: {e}")
            return np.array([]), "<p>Error generating visualization</p>"
    
    def project_embeddings_3d(
        self, 
        embeddings: np.ndarray,
        labels: Optional[List[str]] = None,
        method: str = "umap"
    ) -> Tuple[np.ndarray, str]:
        """
        Project high-dimensional embeddings to 3D.
        
        Args:
            embeddings: Array of shape (n_samples, n_features)
            labels: Optional labels for each embedding
            method: Projection method ("umap" or "tsne")
            
        Returns:
            Tuple of (projected_embeddings, html_plot)
        """
        try:
            if method == "umap":
                from umap import UMAP
                reducer = UMAP(n_components=3, random_state=42)
            else:  # tsne
                from sklearn.manifold import TSNE
                reducer = TSNE(n_components=3, random_state=42)
            
            projected = reducer.fit_transform(embeddings)
            
            # Create interactive 3D plot
            import plotly.graph_objects as go
            
            fig = go.Figure(data=[go.Scatter3d(
                x=projected[:, 0],
                y=projected[:, 1],
                z=projected[:, 2],
                mode='markers',
                marker=dict(
                    size=6,
                    color=list(range(len(projected))),
                    colorscale='Viridis',
                    showscale=True
                ),
                text=labels if labels else [f"Doc {i}" for i in range(len(projected))],
                hovertemplate='<b>%{text}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>'
            )])
            
            fig.update_layout(
                title=f"3D Embedding Projection ({method.upper()})",
                scene=dict(
                    xaxis_title="Dimension 1",
                    yaxis_title="Dimension 2",
                    zaxis_title="Dimension 3"
                ),
                hovermode='closest',
                template="plotly_white",
                height=700,
                width=900
            )
            
            html_plot = fig.to_html(include_plotlyjs='cdn', full_html=False, div_id="plotly-viz-3d")
            # Wrap in div for proper rendering in Gradio
            wrapped_html = f'<div style="width:100%; height:700px; overflow:auto;">{html_plot}</div>'
            logger.info(f"Projected {len(embeddings)} embeddings to 3D using {method}")
            return projected, wrapped_html
            
        except Exception as e:
            logger.error(f"Error projecting embeddings to 3D: {e}")
            return np.array([]), "<p>Error generating 3D visualization</p>"


def estimate_tokens(text: str, model: str = "gpt-4") -> Tuple[int, float]:
    """
    Estimate token count and cost for a given text.
    
    Args:
        text: Input text
        model: Model name for pricing
        
    Returns:
        Tuple of (token_count, estimated_cost_usd)
    """
    try:
        import tiktoken
        
        # Get encoding for model
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")  # Default for GPT-4
        
        tokens = encoding.encode(text)
        token_count = len(tokens)
        
        # Pricing (as of 2024, approximate)
        pricing = {
            "gpt-4": 0.03 / 1000,  # $0.03 per 1K tokens
            "gpt-4-turbo": 0.01 / 1000,
            "gpt-3.5-turbo": 0.001 / 1000,
            "text-embedding-3-small": 0.00002 / 1000,
            "text-embedding-3-large": 0.00013 / 1000,
        }
        
        cost_per_token = pricing.get(model, 0.01 / 1000)
        estimated_cost = token_count * cost_per_token
        
        return token_count, estimated_cost
        
    except Exception as e:
        logger.error(f"Error estimating tokens: {e}")
        return 0, 0.0


def analyze_cluster_quality(embeddings: np.ndarray) -> dict:
    """
    Analyze the quality of embedding clusters.
    
    Args:
        embeddings: Array of embeddings
        
    Returns:
        Dict with quality metrics
    """
    try:
        from sklearn.metrics import silhouette_score, davies_bouldin_score
        from sklearn.cluster import KMeans
        
        if len(embeddings) < 10:
            return {"error": "Need at least 10 embeddings for analysis"}
        
        # Determine optimal number of clusters (2 to min(10, n_samples//2))
        max_clusters = min(10, len(embeddings) // 2)
        n_clusters = min(5, max_clusters)  # Default to 5 or less
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        
        # Calculate metrics
        silhouette = silhouette_score(embeddings, labels)
        davies_bouldin = davies_bouldin_score(embeddings, labels)
        
        # Interpret scores
        if silhouette > 0.5:
            quality = "🟢 Good"
        elif silhouette > 0.25:
            quality = "🟡 Fair"
        else:
            quality = "🔴 Poor"
        
        return {
            "n_clusters": n_clusters,
            "silhouette_score": round(silhouette, 3),
            "davies_bouldin_score": round(davies_bouldin, 3),
            "quality": quality,
            "interpretation": f"Silhouette: {silhouette:.3f} (higher is better, range: -1 to 1)"
        }
        
    except Exception as e:
        logger.error(f"Error analyzing cluster quality: {e}")
        return {"error": str(e)}

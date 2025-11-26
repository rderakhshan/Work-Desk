"""
Visualization tab component for AI Workdesk UI.
"""
import gradio as gr

def create_visualization_tab(ui):
    """
    Create the Visualization tab content.
    
    Args:
        ui: The AIWorkdeskUI instance
        
    Returns:
        None (adds to current context)
    """
    with gr.TabItem("📊 Visualization"):
        gr.Markdown("### 🌌 Embedding Space Visualization")
        
        with gr.Row():
            viz_method = gr.Dropdown(
                choices=["PCA", "t-SNE", "UMAP"],
                value="PCA",
                label="Dimensionality Reduction"
            )
            viz_dim = gr.Dropdown(
                choices=["2D", "3D"],
                value="3D",
                label="Dimensions"
            )
            viz_btn = gr.Button("🎨 Visualize Embeddings", variant="primary")
        
        viz_plot = gr.Plot(label="Embedding Visualization", elem_id="viz-plot-container")
        
        viz_btn.click(
            ui.handle_visualization,
            inputs=[viz_method, viz_dim],
            outputs=[viz_plot]
        )

    with gr.TabItem("🕸️ Knowledge Graph"):
        gr.Markdown("### 🔗 Knowledge Graph Explorer")
        gr.Markdown("*Visualize relationships between documents and concepts*")
        
        with gr.Row():
            graph_btn = gr.Button("🕸️ Generate Graph", variant="primary")
            
        graph_plot = gr.HTML(label="Knowledge Graph", elem_id="graph-plot-container")
        
        graph_btn.click(
            ui.handle_graph_generation,
            inputs=[],
            outputs=[graph_plot]
        )

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
            with gr.Column(scale=1):
                graph_btn = gr.Button("🕸️ Generate Graph", variant="primary")
                viz_mode = gr.Dropdown(
                    choices=["2D", "3D"],
                    value="2D",
                    label="Visualization Mode"
                )
            with gr.Column(scale=2):
                max_nodes = gr.Slider(minimum=10, maximum=500, value=100, step=10, label="Max Nodes (Top Connected)")
                min_weight = gr.Slider(minimum=1, maximum=10, value=1, step=1, label="Min Connection Strength")
            
        graph_plot = gr.HTML(label="Knowledge Graph", elem_id="graph-plot-container")
        
        graph_btn.click(
            ui.handle_graph_generation,
            inputs=[max_nodes, min_weight, viz_mode],
            outputs=[graph_plot]
        )
        
        # Dynamic updates on slider release
        max_nodes.release(
            ui.handle_graph_generation,
            inputs=[max_nodes, min_weight, viz_mode],
            outputs=[graph_plot]
        )
        min_weight.release(
            ui.handle_graph_generation,
            inputs=[max_nodes, min_weight, viz_mode],
            outputs=[graph_plot]
        )
        viz_mode.change(
            ui.handle_graph_generation,
            inputs=[max_nodes, min_weight, viz_mode],
            outputs=[graph_plot]
        )

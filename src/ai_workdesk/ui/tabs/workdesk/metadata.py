"""
Metadata tab component for AI Workdesk UI.
"""
import gradio as gr

def create_metadata_tab(ui):
    """
    Create the Metadata tab content.
    
    Args:
        ui: The AIWorkdeskUI instance
        
    Returns:
        None (adds to current context)
    """
    with gr.TabItem("📋 Metadata Manager"):
        gr.Markdown("### 🗂️ Ingested Files Metadata")
        
        with gr.Row():
            refresh_meta_btn = gr.Button("🔄 Refresh List", variant="secondary")
            delete_meta_btn = gr.Button("🗑️ Delete Selected", variant="stop")
        
        metadata_list = gr.Dataframe(
            headers=["ID", "Filename", "Type", "Size", "Date", "Chunks"],
            datatype=["number", "str", "str", "str", "str", "number"],
            interactive=False,
            label="Stored Documents"
        )
        
        with gr.Row():
            prev_page_btn = gr.Button("⬅️ Previous")
            page_info = gr.Textbox(value="Page 1", label="Page", interactive=False, scale=1)
            next_page_btn = gr.Button("➡️ Next")
        
        # Metadata Events
        refresh_meta_btn.click(
            ui.load_metadata,
            inputs=[],
            outputs=[metadata_list, page_info]
        )
        
        # Initial load
        # Note: We can't use demo.load here easily without passing demo, 
        # but we can rely on the user clicking refresh or handle it in the parent.

    with gr.TabItem("📚 Collections"):
        gr.Markdown("### 🗃️ Manage Collections")
        
        with gr.Row():
            new_col_name = gr.Textbox(label="New Collection Name", placeholder="my-collection")
            create_col_btn = gr.Button("➕ Create", variant="primary")
        
        with gr.Row():
            col_list = gr.Dropdown(label="Select Collection", choices=["default"], value="default")
            switch_col_btn = gr.Button("📂 Switch", variant="secondary")
            delete_col_btn = gr.Button("🗑️ Delete", variant="stop")
        
        col_status = gr.Textbox(label="Status", interactive=False)
        
        # Collection Events
        create_col_btn.click(
            ui.create_collection,
            inputs=[new_col_name],
            outputs=[col_status, col_list]
        )
        
        switch_col_btn.click(
            ui.switch_collection,
            inputs=[col_list],
            outputs=[col_status]
        )
        
        delete_col_btn.click(
            ui.delete_collection,
            inputs=[col_list],
            outputs=[col_status, col_list]
        )

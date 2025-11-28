"""
Ingestion tab component for AI Workdesk UI.
"""
import gradio as gr

def create_ingestion_tab(ui):
    """
    Create the Ingestion tab content.
    
    Args:
        ui: The AIWorkdeskUI instance
        
    Returns:
        None (adds to current context)
    """
    with gr.TabItem("📤 Ingestion"):
        gr.Markdown("### 📄 Document Ingestion")
        
        with gr.Tabs():
            # Files Tab
            with gr.TabItem("📄 Files"):
                file_input = gr.File(
                    file_count="multiple", 
                    label="Upload Documents (TXT, PDF, MD, DOCX, CSV, JSON, HTML, PPTX, XLSX) + Images (PNG, JPG, TIFF) with OCR",
                    file_types=[
                        ".txt", ".pdf", ".md", ".docx", ".csv", ".json",
                        ".html", ".htm", ".pptx", ".xlsx", ".xls",
                        ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".gif"
                    ]
                )
                with gr.Row():
                    ingest_chunk_size = gr.Dropdown([256, 512, 1024], value=512, label="Chunk Size")
                    ingest_chunk_overlap = gr.Slider(0, 200, 50, step=10, label="Overlap")
                
                chunking_strategy = gr.Dropdown(
                    choices=["Fixed Size", "Semantic"],
                    value="Fixed Size",
                    label="Chunking Strategy"
                )
                
                with gr.Row():
                    estimate_btn = gr.Button("💰 Estimate Cost", variant="secondary")
                    ingest_btn = gr.Button("🚀 Ingest Documents", variant="primary", elem_classes=["primary-btn"])
                
                ingest_status = gr.Textbox(label="Status", interactive=False, lines=3, max_lines=10, show_copy_button=True)
                
                estimate_btn.click(
                    ui.handle_token_estimation,
                    inputs=[file_input],
                    outputs=[ingest_status]
                )
                
                ingest_btn.click(
                    ui.handle_ingestion,
                    inputs=[file_input, ingest_chunk_size, ingest_chunk_overlap, chunking_strategy],
                    outputs=[ingest_status]
                )

            # Web Tab
            with gr.TabItem("🌐 Web"):
                gr.Markdown("### 🕸️ Web Crawler")
                web_url = gr.Textbox(label="URL", placeholder="https://example.com")
                web_depth = gr.Slider(0, 5, value=1, step=1, label="Crawl Depth")
                
                with gr.Row():
                    web_chunk_size = gr.Dropdown([256, 512, 1024], value=512, label="Chunk Size")
                    web_chunk_overlap = gr.Slider(0, 200, 50, step=10, label="Overlap")
                
                web_ingest_btn = gr.Button("🚀 Crawl & Ingest", variant="primary")
                web_status = gr.Textbox(label="Status", interactive=False, lines=3, max_lines=10, show_copy_button=True)
                
                web_ingest_btn.click(
                    ui.handle_web_ingestion,
                    inputs=[web_url, web_depth, web_chunk_size, web_chunk_overlap],
                    outputs=[web_status]
                )

            # YouTube Tab
            with gr.TabItem("🎥 YouTube"):
                gr.Markdown("### 🎬 YouTube Video Ingestion")
                gr.Markdown("*Chat with YouTube videos by ingesting their transcripts with timestamp citations*")
                
                youtube_url = gr.Textbox(
                    label="YouTube Video URL(s)",
                    placeholder="https://www.youtube.com/watch?v=dQw4w9WgXcQ\n(one URL per line for batch processing)",
                    lines=3
                )
                
                with gr.Row():
                    yt_chunk_size = gr.Dropdown([256, 512, 1024], value=512, label="Chunk Size")
                    yt_chunk_overlap = gr.Slider(0, 200, 50, step=10, label="Overlap")
                
                with gr.Row():
                    yt_summary = gr.Checkbox(label="Generate AI Summary", value=True)
                    yt_playlist = gr.Checkbox(label="Support Playlists", value=True)
                
                yt_ingest_btn = gr.Button("🚀 Process Video(s)", variant="primary")
                yt_status = gr.Textbox(label="Status", interactive=False, lines=5, max_lines=15, show_copy_button=True)
                
                yt_ingest_btn.click(
                    ui.handle_youtube_ingestion,
                    inputs=[youtube_url, yt_chunk_size, yt_chunk_overlap, yt_summary, yt_playlist],
                    outputs=[yt_status]
                )
            
            # Audio Tab
            with gr.TabItem("🎙️ Audio"):
                gr.Markdown("### 🗣️ Audio Transcription & Ingestion")
                gr.Markdown("*Transcribe audio files or recordings and ingest them into the knowledge base*")
                
                with gr.Row():
                    audio_input = gr.Audio(
                        label="Record or Upload Audio",
                        sources=["microphone", "upload"],
                        type="filepath"
                    )
                    
                with gr.Row():
                    audio_lang = gr.Dropdown(
                        choices=["auto", "en", "es", "fr", "de", "it", "ja", "zh"],
                        value="auto",
                        label="Language"
                    )
                    transcribe_btn = gr.Button("📝 Transcribe", variant="primary")
                
                transcription_output = gr.Textbox(
                    label="Transcription", 
                    lines=10, 
                    max_lines=20,
                    show_copy_button=True,
                    interactive=True
                )
                transcription_status = gr.Textbox(label="Status", interactive=False)
                
                transcribe_btn.click(
                    ui.handle_audio_transcription,
                    inputs=[audio_input, audio_lang],
                    outputs=[transcription_output, transcription_status]
                )
                
                gr.Markdown("#### Ingest Transcription")
                with gr.Row():
                    audio_chunk_size = gr.Dropdown([256, 512, 1024], value=512, label="Chunk Size")
                    audio_chunk_overlap = gr.Slider(0, 200, 50, step=10, label="Overlap")
                    audio_ingest_btn = gr.Button("🚀 Ingest Transcription", variant="primary")
                
                audio_ingest_status = gr.Textbox(label="Ingestion Status", interactive=False)
                
                audio_ingest_btn.click(
                    ui.handle_audio_ingestion,
                    inputs=[transcription_output, audio_chunk_size, audio_chunk_overlap],
                    outputs=[audio_ingest_status]
                )

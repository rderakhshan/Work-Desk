"""
RAG LAB tab component for AI Workdesk UI.
"""
import gradio as gr
from ai_workdesk.ui.constants import MODELS

def create_rag_lab_tab(ui):
    """
    Create the RAG LAB tab content.
    
    Args:
        ui: The AIWorkdeskUI instance
        
    Returns:
        None (adds to current context)
    """
    with gr.TabItem("🧠 RAG LAB"):
        with gr.Row(elem_classes=["chat-row"]):
            with gr.Column(scale=7):
                chatbot = gr.Chatbot(
                    label="AI Assistant",
                    show_copy_button=True,
                    elem_classes=["chat-container"],
                    type="messages",
                    height=60,  # Reduced to 1/4 per user request
                    render_markdown=True,
                )

                # Multimodal Input
                chat_input = gr.MultimodalTextbox(
                    interactive=True,
                    file_count="multiple",
                    placeholder="Ask me anything or upload files...",
                    show_label=False,
                    scale=7,
                    file_types=[
                        ".txt", ".pdf", ".md", ".docx", ".csv", ".json",
                        ".html", ".htm", ".pptx", ".xlsx", ".xls",
                        ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".gif"
                    ]
                )

                # Voice Input Section
                with gr.Accordion("🎤 Voice Input (Optional)", open=False):
                    with gr.Row():
                        voice_input = gr.Audio(
                            label="Record or Upload Voice Query",
                            sources=["microphone", "upload"],
                            type="filepath"
                        )
                        voice_transcribe_btn = gr.Button(
                            "🔄 Transcribe to Text",
                            variant="secondary"
                        )
                    
                    # Microphone permission status for voice input
                    gr.HTML("""
                    <div id="voice-mic-status" style="
                        padding: 12px;
                        margin: 8px 0;
                        border-radius: 6px;
                        background: #fff7ed;
                        border: 1px solid #fed7aa;
                        font-size: 0.9em;
                    ">
                        <p id="voice-mic-status-text" style="margin: 0; color: #9a3412;">
                            🎤 <strong>Microphone Setup:</strong><br>
                            1. Click the <strong>lock icon 🔒</strong> in your browser's address bar<br>
                            2. Set <strong>Microphone</strong> to "Allow"<br>
                            3. Reload the page (F5)<br>
                            4. Click <strong>Record</strong> button above
                        </p>
                    </div>
                    <script>
                    // Check microphone for voice input
                    (function() {
                        const statusDiv = document.getElementById('voice-mic-status');
                        const statusText = document.getElementById('voice-mic-status-text');
                        
                        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
                            navigator.mediaDevices.getUserMedia({audio: true})
                                .then(stream => {
                                    console.log('✅ Voice input microphone ready');
                                    statusDiv.style.background = '#f0fdf4';
                                    statusDiv.style.borderColor = '#bbf7d0';
                                    statusText.style.color = '#15803d';
                                    statusText.innerHTML = '✅ <strong>Microphone Ready!</strong> Click the Record button above to start recording.';
                                    stream.getTracks().forEach(track => track.stop());
                                })
                                .catch(err => {
                                    console.error('❌ Voice input microphone error:', err.name);
                                    statusDiv.style.background = '#fef2f2';
                                    statusDiv.style.borderColor = '#fecaca';
                                    statusText.style.color = '#991b1b';
                                    if (err.name === 'NotAllowedError') {
                                        statusText.innerHTML = '🔒 <strong>Permission Denied!</strong><br>Click the <strong>lock icon 🔒</strong> in address bar → Set Microphone to "Allow" → Reload (F5)';
                                    } else {
                                        statusText.innerHTML = '⚠️ <strong>Microphone Error:</strong> ' + err.name + '<br>Check Windows Settings → Privacy → Microphone';
                                    }
                                });
                        } else {
                            statusDiv.style.background = '#fef2f2';
                            statusDiv.style.borderColor = '#fecaca';
                            statusText.style.color = '#991b1b';
                            statusText.innerHTML = '❌ <strong>Microphone not supported</strong> in this browser. Try Chrome or Edge.';
                        }
                    })();
                    </script>
                    """)


                with gr.Row():
                    with gr.Column(scale=1):
                        clear_btn = gr.Button(
                            "🗑️ Clear Chat", variant="secondary", elem_classes=["secondary-btn"]
                        )
                    with gr.Column(scale=1):
                        download_btn = gr.Button("📥 Download Chat", variant="secondary", elem_classes=["secondary-btn"])



            with gr.Column(scale=2, elem_classes=["glass-panel"]):
                provider_dropdown = gr.Dropdown(
                    choices=["Ollama", "OpenAI"],
                    value="Ollama",
                    label="Provider",
                )
                
                model_dropdown = gr.Dropdown(
                    choices=MODELS["Ollama"],
                    value="deepseek-r1:7b",
                    label="Model",
                    allow_custom_value=True,
                )

                rag_dropdown = gr.Dropdown(
                    choices=[
                        "Naive RAG",
                        "Hybrid Search",
                        "Graph + Vector Hybrid",
                        "HyDE (Hypothetical Document Embeddings)",
                        "RAG Fusion",
                        "None",
                    ],
                    value="Naive RAG",
                    label="RAG Technique",
                )

                gr.Markdown(
                    """
                    ### ⚙️ Settings
                    """
                )
                
                temp_slider = gr.Slider(
                    minimum=0.0,
                    maximum=2.0,
                    value=0.7,
                    step=0.1,
                    label="Temperature",
                )
                
                max_tokens_slider = gr.Slider(
                    minimum=100,
                    maximum=8000,
                    value=2000,
                    step=100,
                    label="Max Tokens",
                )
                
                top_k_slider = gr.Slider(
                    minimum=1,
                    maximum=20,
                    value=4,
                    step=1,
                    label="Top K Documents",
                )
                
                similarity_threshold = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.5,
                    step=0.05,
                    label="Similarity Threshold",
                )

                # Advanced RAG Settings
                with gr.Accordion("🛠️ Advanced Settings", open=False):
                    chunk_size_slider = gr.Slider(
                        minimum=128,
                        maximum=2048,
                        value=512,
                        step=64,
                        label="Chunk Size",
                    )
                    
                    chunk_overlap_slider = gr.Slider(
                        minimum=0,
                        maximum=512,
                        value=50,
                        step=10,
                        label="Chunk Overlap",
                    )
                    
                    use_reranker = gr.Checkbox(
                        label="Use Reranker",
                        value=True,
                        info="Re-rank retrieved documents for better relevance"
                    )
                    
                    enable_voice_response = gr.Checkbox(
                        label="Enable Voice Response",
                        value=False,
                        info="Read out AI responses"
                    )

                system_prompt = gr.Textbox(
                    label="System Prompt",
                    value="You are a helpful AI assistant. Use the provided context to answer questions accurately.",
                    lines=3,
                )

        # Event Handlers
        
        # 1. Chat with RAG
        chat_input.submit(
            ui.chat_router,
            inputs=[
                chat_input, 
                chatbot, 
                model_dropdown, 
                rag_dropdown, 
                provider_dropdown,
                temp_slider, 
                max_tokens_slider, 
                top_k_slider, 
                similarity_threshold,
                chunk_size_slider, 
                chunk_overlap_slider, 
                use_reranker, 
                system_prompt,
                enable_voice_response
            ],
            outputs=[chat_input, chatbot]
        )
        
        # 2. Voice Query
        # Helper to convert voice text to multimodal input format
        def voice_to_multimodal(text):
            return {"text": text, "files": []}

        voice_transcribe_btn.click(
            ui.handle_voice_query,
            inputs=[voice_input],
            outputs=[chat_input] # handle_voice_query needs to return compatible format or we wrap it
        )
        
        # 3. Clear Chat
        clear_btn.click(lambda: None, None, chatbot, queue=False)
        
        # 4. Download Chat
        download_btn.click(
            ui.export_chat,
            inputs=[chatbot],
            outputs=None,  # File download handled by function return if configured
            js="() => { alert('Chat history exported to Downloads folder!'); }"
        )
        
        # 5. Update Models based on Provider
        provider_dropdown.change(
            ui.update_models,
            inputs=[provider_dropdown],
            outputs=[model_dropdown]
        )

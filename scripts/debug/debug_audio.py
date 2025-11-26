"""
Debug script to test Gradio audio recording functionality.

This minimal script helps isolate whether the audio recording issue is:
1. Browser permissions (microphone access)
2. HTTPS requirement (some browsers require HTTPS for microphone)
3. Gradio configuration issue
4. CSS/layout issue in the main app

Run with: uv run debug_audio.py
"""

import gradio as gr


def process_audio(audio_file):
    """Process uploaded or recorded audio."""
    if audio_file is None:
        return "No audio received"
    return f"✅ Audio received: {audio_file}"


# Create minimal interface
with gr.Blocks(title="Audio Recording Test") as demo:
    gr.Markdown("# 🎤 Audio Recording Debug Test")
    gr.Markdown("""
    **Instructions:**
    1. Check if you see both "Record" and "Upload" options below
    2. Try recording audio using the microphone
    3. If you only see "Upload", check:
       - Browser permissions (allow microphone access)
       - HTTPS requirement (some browsers need HTTPS for microphone)
       - Browser console for errors (F12)
    """)
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Test 1: Both Sources")
            audio_both = gr.Audio(
                label="Record or Upload Audio",
                sources=["microphone", "upload"],
                type="filepath"
            )
            output_both = gr.Textbox(label="Result")
            audio_both.change(process_audio, inputs=[audio_both], outputs=[output_both])
        
        with gr.Column():
            gr.Markdown("### Test 2: Microphone Only")
            audio_mic = gr.Audio(
                label="Record Audio (Microphone Only)",
                sources=["microphone"],
                type="filepath"
            )
            output_mic = gr.Textbox(label="Result")
            audio_mic.change(process_audio, inputs=[audio_mic], outputs=[output_mic])
    
    gr.Markdown("""
    ---
    **Expected Behavior:**
    - Test 1 should show both record and upload buttons
    - Test 2 should show only record button
    
    **Common Issues:**
    - **No microphone option**: Browser doesn't have mic permission or requires HTTPS
    - **Microphone grayed out**: No microphone detected on system
    - **Error in console**: Check browser console (F12) for specific errors
    """)


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7861,  # Different port from main app
        share=False
    )

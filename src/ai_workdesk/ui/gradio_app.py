"""
AI Workdesk - Gradio Web Interface with Authentication

A beautiful, secure web interface for your AI Workdesk with:
- User authentication
- Multiple AI service support
- Chat history
- Modern glassmorphism design
- Multi-page navigation with sidebar

Run with: uv run ai-workdesk-ui
"""

import os
from datetime import datetime
from typing import Optional

import gradio as gr
from openai import OpenAI

from ai_workdesk import get_auth_manager, get_logger, get_settings

logger = get_logger(__name__)

# User credentials (in production, use a database)
USERS = {
    "admin": "admin123",  # Default user
    "demo": "demo123",
}


class AIWorkdeskUI:
    """AI Workdesk Gradio UI with authentication and multi-page navigation."""

    def __init__(self):
        """Initialize the UI."""
        self.settings = get_settings()
        self.auth_manager = get_auth_manager()
        self.openai_client: Optional[OpenAI] = None

        # Initialize OpenAI if configured
        if self.auth_manager.validate_service("openai"):
            self.openai_client = OpenAI(api_key=self.settings.openai_api_key)
            logger.info("OpenAI client initialized")

    def authenticate(self, username: str, password: str) -> bool:
        """
        Authenticate user.

        Args:
            username: Username
            password: Password

        Returns:
            True if authenticated, False otherwise
        """
        logger.info(f"Authentication attempt for user: {username}")

        if username in USERS and USERS[username] == password:
            logger.info(f"User {username} authenticated successfully")
            return True

        logger.warning(f"Failed authentication attempt for user: {username}")
        return False

    def get_auth_status(self) -> str:
        """Get authentication status as formatted string."""
        status = self.auth_manager.check_authentication_status()

        lines = ["### 🔐 API Service Status\n"]
        for service, authenticated in status.items():
            emoji = "✅" if authenticated else "❌"
            state = "Configured" if authenticated else "Not Configured"
            lines.append(f"- **{service}**: {emoji} {state}")

        return "\n".join(lines)

    def chat_with_ai(
        self,
        message: str,
        history: list,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> tuple:
        """
        Chat with AI service.

        Args:
            message: User message
            history: Chat history  
            model: Model name
            temperature: Temperature setting
            max_tokens: Maximum tokens

        Returns:
            Tuple of (updated history, empty string for input)
        """
        if not message.strip():
            return history, ""

        if not self.openai_client:
            error_msg = "⚠️ OpenAI is not configured. Please set OPENAI_API_KEY in .env"
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": error_msg})
            logger.warning("Chat attempted without OpenAI configuration")
            return history, ""

        logger.info(f"Processing chat with model: {model}, temp: {temperature}")

        try:
            # Prepare messages from history
            messages = []
            for msg in history:
                if isinstance(msg, dict) and "role" in msg and "content" in msg:
                    messages.append({"role": msg["role"], "content": msg["content"]})

            # Add current message
            messages.append({"role": "user", "content": message})

            # Get completion
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            ai_response = response.choices[0].message.content

            # Log token usage
            usage = response.usage
            logger.info(
                f"Tokens - Prompt: {usage.prompt_tokens}, "
                f"Completion: {usage.completion_tokens}, "
                f"Total: {usage.total_tokens}"
            )

            # Update history with proper format
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": ai_response})

            return history, ""

        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": error_msg})
            logger.error(f"Chat error: {e}")
            return history, ""

    def create_interface(self) -> gr.Blocks:
        """Create the Gradio interface with sidebar navigation."""

        # Enhanced custom CSS for sidebar and pages
        custom_css = """
        .gradio-container {
            font-family: 'Inter', sans-serif;
            max-width: 95% !important;
        }
        
        /* Sidebar Styling */
        .sidebar-nav {
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%);
            border-radius: 15px;
            padding: 10px;
            margin-bottom: 10px;
        }
        
        .nav-button {
            width: 100%;
            margin: 5px 0;
            transition: all 0.3s ease;
        }
        
        .nav-button:hover {
            transform: translateX(5px);
        }
        
        /* Page Content Styling */
        .header-title {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2.5em;
            font-weight: 800;
            text-align: center;
            margin: 20px 0;
        }
        
        .status-box {
            background: rgba(102, 126, 234, 0.1);
            border-radius: 10px;
            padding: 15px;
            border-left: 4px solid #667eea;
        }
        
        .chat-container {
            border-radius: 15px;
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
            height: 70vh !important;
            min-height: 500px;
        }
        
        /* Hide Gradio footer */
        footer {
            display: none !important;
        }
        
        /* Smooth transitions for page changes */
        .transition-fade {
            transition: opacity 0.3s ease-in-out;
        }
        """

        with gr.Blocks(
            theme=gr.themes.Soft(
                primary_hue="indigo",
                secondary_hue="purple",
                neutral_hue="slate",
            ),
            css=custom_css,
            title="AI Workdesk",
        ) as demo:
            # Header
            gr.HTML(
                """
                <div style="text-align: center; padding: 20px;">
                    <h1 class="header-title">🚀 AI Workdesk</h1>
                    <p style="color: #666; font-size: 1.1em;">
                        Your Professional AI Development Environment
                    </p>
                </div>
                """
            )

            with gr.Row():
                # Sidebar
                with gr.Sidebar(open=True, width=250):
                    gr.Markdown("### 🧭 Navigation")

                    home_btn = gr.Button(
                        "🏠 Home",
                        variant="primary",
                        elem_classes=["nav-button"],
                        size="lg",
                    )

                    workdesk_btn = gr.Button(
                        "💼 Work Desk",
                        variant="secondary",
                        elem_classes=["nav-button"],
                        size="lg",
                    )

                    about_btn = gr.Button(
                        "ℹ️ About",
                        variant="secondary",
                        elem_classes=["nav-button"],
                        size="lg",
                    )

                    # Push subsequent content to bottom
                    with gr.Column(scale=1, min_width=0):
                        pass

                    # Logout Button
                    logout_btn = gr.Button(
                        "🚪 Logout",
                        variant="secondary",
                        elem_classes=["nav-button"],
                        size="lg",
                    )

                    logout_btn.click(
                        fn=None,
                        inputs=None,
                        outputs=None,
                        js="function() { window.location.href = '/logout'; }",
                    )

                    gr.Markdown("---")
                    gr.Markdown(
                        """
                        <div style="text-align: center; padding: 10px; color: #666; font-size: 0.9em;">
                            <p>v0.1.0</p>
                            <p>Powered by UV & Gradio</p>
                        </div>
                        """
                    )

                # Main Content Area
                with gr.Column(scale=4):
                    # Home Page
                    with gr.Column(visible=True) as home_page:
                        # Welcome Section
                        gr.HTML(
                            """
                            <div style="text-align: center; padding: 40px 20px;">
                                <h1 style="
                                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                                    -webkit-background-clip: text;
                                    -webkit-text-fill-color: transparent;
                                    font-size: 3.5em;
                                    font-weight: 800;
                                    margin-bottom: 10px;
                                ">🚀 Welcome to AI Workdesk</h1>
                                <p style="color: #666; font-size: 1.3em; margin-top: 10px;">
                                    Your Professional AI Development Environment
                                </p>
                            </div>
                            """
                        )

                        # Status Cards
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.HTML(
                                    """
                                    <div style="
                                        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
                                        border-radius: 15px;
                                        padding: 25px;
                                        border: 1px solid rgba(102, 126, 234, 0.3);
                                        backdrop-filter: blur(10px);
                                        text-align: center;
                                    ">
                                        <div style="font-size: 3em; margin-bottom: 10px;">💬</div>
                                        <h3 style="color: #667eea; margin: 10px 0;">AI Chat</h3>
                                        <p style="color: #666;">Interactive conversations with GPT models</p>
                                    </div>
                                    """
                                )

                            with gr.Column(scale=1):
                                gr.HTML(
                                    """
                                    <div style="
                                        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
                                        border-radius: 15px;
                                        padding: 25px;
                                        border: 1px solid rgba(102, 126, 234, 0.3);
                                        backdrop-filter: blur(10px);
                                        text-align: center;
                                    ">
                                        <div style="font-size: 3em; margin-bottom: 10px;">🔐</div>
                                        <h3 style="color: #667eea; margin: 10px 0;">Secure</h3>
                                        <p style="color: #666;">Authentication and API key management</p>
                                    </div>
                                    """
                                )

                            with gr.Column(scale=1):
                                gr.HTML(
                                    """
                                    <div style="
                                        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
                                        border-radius: 15px;
                                        padding: 25px;
                                        border: 1px solid rgba(102, 126, 234, 0.3);
                                        backdrop-filter: blur(10px);
                                        text-align: center;
                                    ">
                                        <div style="font-size: 3em; margin-bottom: 10px;">⚙️</div>
                                        <h3 style="color: #667eea; margin: 10px 0;">Configurable</h3>
                                        <p style="color: #666;">Customize models and parameters</p>
                                    </div>
                                    """
                                )

                        gr.Markdown("---")

                        # API Status Section
                        gr.Markdown("## 📊 System Status")
                        auth_status = gr.Markdown(
                            self.get_auth_status(),
                            elem_classes=["status-box"],
                        )

                        with gr.Row():
                            refresh_status_btn = gr.Button("🔄 Refresh Status", variant="secondary", scale=1)

                        refresh_status_btn.click(
                            lambda: self.get_auth_status(),
                            None,
                            auth_status,
                        )

                        gr.Markdown("---")

                        # Quick Actions
                        gr.Markdown("## 🚀 Quick Actions")
                        gr.Markdown("Ready to start working with AI? Head to the Work Desk to begin chatting!")

                    # Work Desk Page
                    with gr.Column(visible=False) as workdesk_page:
                        gr.Markdown("# 💬 AI Work Desk")

                        with gr.Row():
                            with gr.Column(scale=4):
                                chatbot = gr.Chatbot(
                                    label="AI Assistant",
                                    show_copy_button=True,
                                    elem_classes=["chat-container"],
                                    type="messages",
                                )

                                with gr.Row():
                                    msg = gr.Textbox(
                                        label="Message",
                                        placeholder="Ask me anything...",
                                        scale=4,
                                        show_label=False,
                                    )
                                    send_btn = gr.Button("Send 📤", scale=1, variant="primary")

                                with gr.Row():
                                    clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary", scale=1)

                            with gr.Column(scale=1):
                                model_dropdown = gr.Dropdown(
                                    choices=[
                                        "gpt-4o", 
                                        "gpt-4o-mini", 
                                        "gpt-4-turbo", 
                                        "gpt-4", 
                                        "gpt-3.5-turbo"
                                    ],
                                    value=self.settings.default_llm_model,
                                    label="Model",
                                    allow_custom_value=True  # Allow custom model names
                                )

                                temperature_slider = gr.Slider(
                                    minimum=0.0,
                                    maximum=2.0,
                                    value=self.settings.default_temperature,
                                    step=0.1,
                                    label="Temperature",
                                )

                                max_tokens_slider = gr.Slider(
                                    minimum=100,
                                    maximum=8192,  # Increased to support larger token limits
                                    value=min(self.settings.max_tokens, 8192),  # Ensure value is within range
                                    step=100,
                                    label="Max Tokens",
                                )

                        # Chat interaction handlers
                        msg.submit(
                            self.chat_with_ai,
                            [msg, chatbot, model_dropdown, temperature_slider, max_tokens_slider],
                            [chatbot, msg],
                        )

                        send_btn.click(
                            self.chat_with_ai,
                            [msg, chatbot, model_dropdown, temperature_slider, max_tokens_slider],
                            [chatbot, msg],
                        )

                        clear_btn.click(lambda: ([], ""), None, [chatbot, msg])

                    # About Page
                    with gr.Column(visible=False) as about_page:
                        gr.Markdown("# ℹ️ About AI Workdesk")

                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.HTML(
                                    """
                                    <div style="
                                        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
                                        border-radius: 15px;
                                        padding: 20px;
                                        border: 1px solid rgba(102, 126, 234, 0.3);
                                        margin-bottom: 20px;
                                    ">
                                        <h3 style="color: #667eea;">💬 AI Chat</h3>
                                        <p style="color: #666;">Interactive conversations with GPT models</p>
                                    </div>
                                    """
                                )

                            with gr.Column(scale=1):
                                gr.HTML(
                                    """
                                    <div style="
                                        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
                                        border-radius: 15px;
                                        padding: 20px;
                                        border: 1px solid rgba(102, 126, 234, 0.3);
                                        margin-bottom: 20px;
                                    ">
                                        <h3 style="color: #667eea;">🔐 Secure Auth</h3>
                                        <p style="color: #666;">User login system for security</p>
                                    </div>
                                    """
                                )

                            with gr.Column(scale=1):
                                gr.HTML(
                                    """
                                    <div style="
                                        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
                                        border-radius: 15px;
                                        padding: 20px;
                                        border: 1px solid rgba(102, 126, 234, 0.3);
                                        margin-bottom: 20px;
                                    ">
                                        <h3 style="color: #667eea;">⚙️ Configurable</h3>
                                        <p style="color: #666;">Adjust models and parameters</p>
                                    </div>
                                    """
                                )

                        gr.Markdown(
                            """
                            ## Version 0.1.0
                            
                            ### Supported Services
                            - OpenAI (GPT-4, GPT-3.5)
                            - Anthropic (Claude) - Coming Soon
                            - Google AI - Coming Soon
                            
                            ### Tech Stack
                            - **Framework**: Gradio 5.0
                            - **Backend**: Python with UV package manager
                            - **AI Services**: OpenAI, LangChain
                            - **Authentication**: Secure user sessions
                            
                            ### Quick Tips
                            1. Adjust temperature for creativity (0 = focused, 2 = creative)
                            2. Use higher max tokens for longer responses
                            3. Check Home page for API configuration status
                            4. Clear chat history for a fresh start
                            
                            ---
                            
                            **Made with ❤️ using AI Workdesk**
                            """
                        )

                        # Configuration Table
                        gr.Markdown("## 🔧 Current Configuration")

                        config_info = gr.DataFrame(
                            value=[
                                ["Default Model", self.settings.default_llm_model],
                                ["Temperature", str(self.settings.default_temperature)],
                                ["Max Tokens", str(self.settings.max_tokens)],
                                ["Environment", self.settings.environment],
                                ["Log Level", self.settings.log_level],
                                ["Log File", str(self.settings.log_file)],
                            ],
                            headers=["Setting", "Value"],
                            label="System Configuration",
                        )

            # Navigation button handlers
            def show_home():
                return {
                    home_page: gr.update(visible=True),
                    workdesk_page: gr.update(visible=False),
                    about_page: gr.update(visible=False),
                    home_btn: gr.update(variant="primary"),
                    workdesk_btn: gr.update(variant="secondary"),
                    about_btn: gr.update(variant="secondary"),
                }

            def show_workdesk():
                return {
                    home_page: gr.update(visible=False),
                    workdesk_page: gr.update(visible=True),
                    about_page: gr.update(visible=False),
                    home_btn: gr.update(variant="secondary"),
                    workdesk_btn: gr.update(variant="primary"),
                    about_btn: gr.update(variant="secondary"),
                }

            def show_about():
                return {
                    home_page: gr.update(visible=False),
                    workdesk_page: gr.update(visible=False),
                    about_page: gr.update(visible=True),
                    home_btn: gr.update(variant="secondary"),
                    workdesk_btn: gr.update(variant="secondary"),
                    about_btn: gr.update(variant="primary"),
                }

            home_btn.click(
                show_home,
                None,
                [home_page, workdesk_page, about_page, home_btn, workdesk_btn, about_btn],
            )

            workdesk_btn.click(
                show_workdesk,
                None,
                [home_page, workdesk_page, about_page, home_btn, workdesk_btn, about_btn],
            )

            about_btn.click(
                show_about,
                None,
                [home_page, workdesk_page, about_page, home_btn, workdesk_btn, about_btn],
            )

        return demo

    def launch(
        self,
        share: bool = False,
        server_port: int = 7860,
        auth: bool = True,
    ) -> None:
        """
        Launch the Gradio interface.

        Args:
            share: Create public link
            server_port: Port to run on
            auth: Enable authentication
        """
        logger.info(f"Launching AI Workdesk UI on port {server_port}")

        demo = self.create_interface()

        if auth:
            demo.launch(
                auth=self.authenticate,
                auth_message="🔐 Please login to AI Workdesk",
                share=share,
                server_port=server_port,
                favicon_path=None,
            )
        else:
            demo.launch(
                share=share,
                server_port=server_port,
                favicon_path=None,
            )


def main() -> None:
    """Main entry point."""
    print("\n" + "=" * 60)
    print("🚀 AI Workdesk - Web Interface")
    print("=" * 60)
    print("\n📝 Default Login Credentials:")
    print("   Username: admin")
    print("   Password: admin123")
    print("\n   Username: demo")
    print("   Password: demo123")
    print("\n💡 Tip: Edit credentials in src/ai_workdesk/ui/gradio_app.py")
    print("=" * 60 + "\n")

    ui = AIWorkdeskUI()
    ui.launch(
        share=False,  # Set to True for public link
        server_port=7860,
        auth=True,  # Enable authentication
    )


if __name__ == "__main__":
    main()
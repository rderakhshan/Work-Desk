"""
Layout components for AI Workdesk UI.
"""
import gradio as gr
from ai_workdesk.smart_dashboard.ui import DASHBOARD_CSS

CUSTOM_CSS = """
/* Project Ambitions: Mesh Gradient & Global Reset */
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');

:root, .dark, body, gradio-app {
    --font-sans: 'Outfit', sans-serif !important;
    --background-fill-primary: transparent !important;
    --background-fill-secondary: transparent !important;
    --block-background-fill: transparent !important;
    --border-color-primary: transparent !important;
    --body-background-fill: transparent !important;
    font-size: 130% !important; /* Increased font size by 1.3x */
}

/* Animated Mesh Gradient Background - REMOVED per user request */
body {
    background-color: #ffffff !important;
    margin: 0;
    padding: 0;
    overflow: hidden;
    font-family: 'Outfit', sans-serif !important;
}

.gradio-container {
    background: #ffffff !important;
    height: 100vh !important;
    max-height: 100vh !important;
    overflow: hidden !important;
    display: flex;
    flex-direction: column;
}

/* Sidebar - Glass & Borderless */
.sidebar-container {
    height: 100vh !important;
    overflow-y: auto !important;
    border-right: 1px solid rgba(255,255,255,0.1) !important;
    background: rgba(255, 255, 255, 0.05) !important;
    backdrop-filter: blur(10px) !important;
}

/* Main Content - Full Width & Transparent */
.main-content {
    height: 100vh !important;
    overflow-y: auto !important;
    background: transparent !important;
    padding: 0 !important;
}

/* Force Transparency on Panels */
.glass-panel, .gray-panel, .panel {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

/* Tabs - Clean */
.tabs, .tabitem {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

/* Remove any vertical lines from tab containers */
.tab-nav, .tab-content, .tabitem > div {
    border-left: none !important;
    border-right: none !important;
}

/* Chatbot - Floating Glass */
.chat-container {
    background: rgba(255, 255, 255, 0.1) !important;
    backdrop-filter: blur(20px) !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    border-radius: 24px !important;
    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37) !important;
    height: 17.5vh !important; /* Reduced to 25% of previous 70vh per user request */
    min-height: 150px !important;
    overflow-y: auto !important;
    flex-grow: 1 !important;
}

/* Primary Buttons - Indigo (Active State) */
.primary-btn {
    background: #6366f1 !important;
    border: none !important;
    color: white !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.2) !important;
    transition: all 0.2s ease !important;
}
.primary-btn:hover {
    background: #4f46e5 !important;
    transform: translateY(-1px);
    box-shadow: 0 6px 16px rgba(99, 102, 241, 0.3) !important;
}

/* Secondary Buttons - White (Inactive State) */
.secondary-btn {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    color: #64748b !important;
    border-radius: 10px !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
}
.secondary-btn:hover {
    background: #f8fafc !important;
    border-color: #cbd5e1 !important;
    color: #334155 !important;
}

/* Inputs - White */
.gradio-dropdown, .gradio-slider, .gradio-textbox, .gradio-number {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 10px !important;
}

/* Typography */
h1, h2, h3, h4 {
    color: #1e293b !important;
    padding: 10px 0 !important;
}
p, span, label {
    color: #475569 !important;
}

/* Status Box */
.status-box {
    background: #eff6ff !important;
    border: 1px solid #dbeafe !important;
    border-radius: 10px !important;
    padding: 12px !important;
    color: #1e40af !important;
}

/* Headers - Remove Grey */
.gradio-container h1, .gradio-container h2, .gradio-container h3 {
    background: transparent !important;
}

/* Tabs - White */
.tab-nav {
    background: transparent !important;
    border-bottom: 1px solid #e2e8f0 !important;
}
.tab-nav.selected {
    border-bottom: 2px solid #6366f1 !important;
    color: #6366f1 !important;
}

/* Sidebar Layout */
.sidebar-container {
    display: flex;
    flex-direction: column;
    height: 100%;
    min-height: 80vh; /* Match chat height roughly */
}
.sidebar-spacer {
    flex-grow: 1;
}

/* Button Active States */
.primary-btn:active, .secondary-btn:active {
    background: #6366f1 !important;
    color: white !important;
    transform: scale(0.98);
}

/* Title Fix - Bring to Front with Maximum Specificity */
.gradio-container h1,
.gradio-container h2,
h1, h2 {
    position: relative !important;
    z-index: 1000 !important;
    background: white !important;
    padding: 10px 0 !important;
    opacity: 1 !important;
    visibility: visible !important;
}

/* Logo Styling - 80% Sidebar Width */
.sidebar-logo {
    text-align: center;
    margin-top: 20px;
    padding: 15px;
    width: 100%;
}
.sidebar-logo img {
    width: 80% !important;
    height: auto !important;
    max-width: 180px;
    border-radius: 20px;
    box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4);
    transition: transform 0.2s ease;
}
.sidebar-logo img:hover {
    transform: scale(1.05);
}

/* Visualization and Graph Plot Containers */
#viz-plot-container, #graph-plot-container {
    min-height: 600px !important;
    height: auto !important;
}

#viz-plot-container iframe, #graph-plot-container iframe {
    min-height: 600px !important;
}

/* Sidebar Toggle Button */
.sidebar-toggle {
    position: fixed;
    top: 20px;
    left: 200px;
    width: 30px;
    height: 30px;
    background: #6366f1 !important;
    border: none !important;
    border-radius: 50% !important;
    cursor: pointer;
    z-index: 1001;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white !important;
    font-size: 18px;
    box-shadow: 0 2px 8px rgba(99, 102, 241, 0.3);
    transition: all 0.3s ease;
}

.sidebar-toggle:hover {
    background: #4f46e5 !important;
    transform: scale(1.1);
}

/* Collapsed Sidebar State */
.sidebar-container.collapsed {
    width: 60px !important;
    min-width: 60px !important;
}

.sidebar-container.collapsed .sidebar-logo,
.sidebar-container.collapsed hr,
.sidebar-container.collapsed .sidebar-spacer {
    display: none !important;
}

.sidebar-container.collapsed button {
    width: 40px !important;
    padding: 8px !important;
    font-size: 20px !important;
    justify-content: center !important;
}

.sidebar-container.collapsed button span {
    display: none !important;
}

.sidebar-container.collapsed + * .sidebar-toggle {
    left: 60px !important;
}

/* Smooth Transitions */
.sidebar-container {
    transition: all 0.3s ease !important;
    position: relative;
}

/* External Link Button Indicator */
.external-link-btn::after {
    content: " ↗";
    font-size: 0.8em;
    opacity: 0.7;
}
""" + DASHBOARD_CSS

def create_sidebar_content():
    """Create the sidebar content."""
    # Toggle button for sidebar collapse
    gr.HTML(
        """
        <button class="sidebar-toggle" onclick="toggleSidebar()" title="Toggle Sidebar">
            ☰
        </button>
        <script>
        function toggleSidebar() {
            const sidebar = document.querySelector('.sidebar-container');
            const toggle = document.querySelector('.sidebar-toggle');
            if (sidebar) {
                sidebar.classList.toggle('collapsed');
                const isCollapsed = sidebar.classList.contains('collapsed');
                localStorage.setItem('sidebarCollapsed', isCollapsed);
                toggle.style.left = isCollapsed ? '60px' : '200px';
            }
        }
        
        // Restore sidebar state on load
        window.addEventListener('load', function() {
            const isCollapsed = localStorage.getItem('sidebarCollapsed') === 'true';
            const sidebar = document.querySelector('.sidebar-container');
            const toggle = document.querySelector('.sidebar-toggle');
            if (isCollapsed && sidebar) {
                sidebar.classList.add('collapsed');
                if (toggle) toggle.style.left = '60px';
            }
        });
        </script>
        """
    )
    
    gr.HTML(
        """
        <div class="sidebar-logo">
            <img src="https://i.imgur.com/7bM7V7M.png" alt="RAG Evolution Logo">
            <h2 style="
                margin-top: 15px; 
                font-weight: 800; 
                font-size: 1.4em;
                background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                text-shadow: 0 2px 10px rgba(99, 102, 241, 0.2);
            ">RAG Evolution</h2>
        </div>
        """
    )
    
    gr.Markdown("---")
    
    home_btn = gr.Button("🏠 Home", variant="primary", elem_classes=["primary-btn"])
    workdesk_btn = gr.Button("🛠️ Work Desk", variant="secondary", elem_classes=["secondary-btn"])
    autogen_btn = gr.Button("🤖 AutoGen Studio", variant="secondary", elem_classes=["secondary-btn", "external-link-btn"])
    about_btn = gr.Button("ℹ️ About", variant="secondary", elem_classes=["secondary-btn"])
    
    gr.Markdown("---")
    
    # Spacer
    gr.HTML('<div class="sidebar-spacer"></div>')
    
    logout_btn = gr.Button("🚪 Logout", variant="secondary", elem_classes=["secondary-btn"])
    
    return home_btn, workdesk_btn, autogen_btn, about_btn, logout_btn

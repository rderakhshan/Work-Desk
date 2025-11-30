"""
Documentation renderer utility for AI Workdesk.
Automatically generates table of contents from Markdown files.
"""
from pathlib import Path
from typing import Tuple
import markdown
from markdown.extensions.toc import TocExtension
from markdown.extensions.fenced_code import FencedCodeExtension
from markdown.extensions.tables import TableExtension
from markdown.extensions.codehilite import CodeHiliteExtension
from loguru import logger


class DocumentationRenderer:
    """Renders Markdown documentation with auto-generated TOC."""
    
    def __init__(self, docs_dir: str = "docs/user-guide"):
        """
        Initialize the documentation renderer.
        
        Args:
            docs_dir: Path to documentation directory
        """
        self.docs_dir = Path(docs_dir)
        
        # Initialize Markdown with extensions
        self.md = markdown.Markdown(
            extensions=[
                TocExtension(
                    permalink=True,
                    permalink_title="Link to this section",
                    toc_depth=3,
                    baselevel=1
                ),
                FencedCodeExtension(),
                TableExtension(),
                CodeHiliteExtension(
                    linenums=False,
                    css_class='highlight'
                ),
                'nl2br',  # Convert newlines to <br>
                'sane_lists'  # Better list handling
            ]
        )
    
    def render(self, filename: str) -> Tuple[str, str]:
        """
        Load and render a Markdown file with auto-generated TOC.
        
        Args:
            filename: Name of the Markdown file (e.g., "index.md")
            
        Returns:
            Tuple of (toc_html, content_html)
        """
        try:
            file_path = self.docs_dir / filename
            
            if not file_path.exists():
                logger.error(f"Documentation file not found: {file_path}")
                return self._error_toc(), self._error_content(filename)
            
            # Read Markdown content
            content = file_path.read_text(encoding='utf-8')
            
            # Reset markdown instance for fresh conversion
            self.md.reset()
            
            # Convert to HTML
            html_content = self.md.convert(content)
            
            # Extract auto-generated TOC
            toc_html = self.md.toc
            
            # Post-process HTML for better styling
            html_content = self._post_process_html(html_content)
            toc_html = self._post_process_toc(toc_html)
            
            logger.info(f"Rendered documentation: {filename}")
            return toc_html, html_content
            
        except Exception as e:
            logger.error(f"Error rendering documentation {filename}: {e}")
            return self._error_toc(), self._error_content(filename, str(e))
    
    def _post_process_html(self, html: str) -> str:
        """Add custom classes and styling to HTML content."""
        # Add copy buttons to code blocks
        html = html.replace(
            '<pre>',
            '<div class="code-block-wrapper"><pre>'
        )
        html = html.replace(
            '</pre>',
            '</pre><button class="copy-code-btn" onclick="copyCode(this)">📋 Copy</button></div>'
        )
        
        # Add table wrapper for responsive tables
        html = html.replace('<table>', '<div class="table-wrapper"><table>')
        html = html.replace('</table>', '</table></div>')
        
        return html
    
    def _post_process_toc(self, toc: str) -> str:
        """Add custom classes to TOC."""
        if not toc or toc == '':
            return '<div class="toc-empty">No table of contents available</div>'
        
        # Wrap TOC in custom div
        return f'<div class="doc-toc">{toc}</div>'
    
    def _error_toc(self) -> str:
        """Generate error TOC."""
        return '<div class="toc-error">⚠️ Error loading TOC</div>'
    
    def _error_content(self, filename: str, error: str = "") -> str:
        """Generate error content."""
        error_msg = f"<p>Error: {error}</p>" if error else ""
        return f"""
        <div class="doc-error">
            <h2>⚠️ Documentation Not Found</h2>
            <p>Could not load: <code>{filename}</code></p>
            {error_msg}
            <p>Please check that the file exists in <code>docs/user-guide/</code></p>
        </div>
        """
    
    def list_available_docs(self) -> list[str]:
        """List all available documentation files."""
        if not self.docs_dir.exists():
            return []
        
        return [
            f.name for f in self.docs_dir.glob("*.md")
            if not f.name.startswith('.')
        ]


# JavaScript for copy code functionality
COPY_CODE_JS = """
<script>
function copyCode(button) {
    const codeBlock = button.previousElementSibling;
    const code = codeBlock.textContent;
    
    navigator.clipboard.writeText(code).then(() => {
        const originalText = button.textContent;
        button.textContent = '✅ Copied!';
        button.style.background = '#10b981';
        
        setTimeout(() => {
            button.textContent = originalText;
            button.style.background = '#6366f1';
        }, 2000);
    }).catch(err => {
        console.error('Failed to copy:', err);
        button.textContent = '❌ Failed';
    });
}
</script>
"""

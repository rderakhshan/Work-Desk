@echo off
echo Starting AI Workdesk in Development Mode (Hot Reload Enabled)...
echo.
echo ------------------------------------------------------------------
echo  NOTE: Changes to files will automatically reload the server.
echo  Press Ctrl+C to stop.
echo ------------------------------------------------------------------
echo.

uv run python -m gradio src/ai_workdesk/ui/gradio_app.py
pause

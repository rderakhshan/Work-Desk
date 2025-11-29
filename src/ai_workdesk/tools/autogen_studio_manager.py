import os
import subprocess
import time
import requests
import threading
from loguru import logger
from typing import Optional

class AutoGenStudioManager:
    """Manages the AutoGen Studio subprocess."""

    def __init__(self, port: int = 8081):
        self.port = port
        self.host = "127.0.0.1"
        self.process: Optional[subprocess.Popen] = None
        self.ui_url = f"http://{self.host}:{self.port}"

    def start_server(self):
        """Starts the AutoGen Studio server in a subprocess."""
        if self.is_running():
            logger.info(f"AutoGen Studio is already running at {self.ui_url}")
            return

        logger.info(f"Starting AutoGen Studio on port {self.port}...")
        try:
            # Run 'autogenstudio ui --port <port>'
            # We use shell=True for Windows compatibility with some commands, but for 'autogenstudio' 
            # it might be better to run it as a module if possible, or just the command.
            # Let's try running the command directly.
            cmd = ["autogenstudio", "ui", "--port", str(self.port), "--host", self.host]
            
            # Use CREATE_NO_WINDOW to prevent a console window from popping up on Windows
            creationflags = 0
            if os.name == 'nt':
                creationflags = subprocess.CREATE_NO_WINDOW

            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                creationflags=creationflags
            )

            # Start a thread to monitor output (optional, for debugging)
            threading.Thread(target=self._monitor_output, args=(self.process.stdout, "STDOUT"), daemon=True).start()
            threading.Thread(target=self._monitor_output, args=(self.process.stderr, "STDERR"), daemon=True).start()

            # Wait a bit for it to start
            self._wait_for_server()

        except Exception as e:
            logger.error(f"Failed to start AutoGen Studio: {e}")

    def stop_server(self):
        """Stops the AutoGen Studio server."""
        if self.process:
            logger.info("Stopping AutoGen Studio...")
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None
            logger.info("AutoGen Studio stopped.")

    def is_running(self) -> bool:
        """Checks if the server is responsive."""
        try:
            response = requests.get(self.ui_url, timeout=1)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def _wait_for_server(self, timeout: int = 60):
        """Waits for the server to become responsive."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.is_running():
                logger.info(f"AutoGen Studio is ready at {self.ui_url}")
                return
            time.sleep(1)
        logger.warning(f"AutoGen Studio start timed out after {timeout}s. Process PID: {self.process.pid if self.process else 'None'}")

    def _monitor_output(self, pipe, name):
        """Reads output from the subprocess."""
        try:
            for line in iter(pipe.readline, ''):
                if line:
                    logger.info(f"[AutoGenStudio {name}] {line.strip()}") # Changed to INFO to see output in main logs
        except Exception:
            pass

    def get_ui_url(self) -> str:
        return self.ui_url

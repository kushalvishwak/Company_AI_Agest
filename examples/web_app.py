"""Run the offline agent in a local browser-based chat interface."""

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agent_stack.orchestrator import bootstrap

HOST = "127.0.0.1"
PORT = 8000
MAX_REQUEST_BYTES = 16_384
STACK = bootstrap()
STACK_LOCK = threading.Lock()
PAGE = Path(__file__).resolve().parent / "web" / "index.html"


class AgentRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            page = PAGE.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(page)))
            self.end_headers()
            self.wfile.write(page)
        elif path == "/health":
            self._send_json(200, {"status": "ok"})
        else:
            self._send_json(404, {"error": "Not found"})

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/chat":
            self._send_json(404, {"error": "Not found"})
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json(400, {"error": "Invalid Content-Length"})
            return

        if content_length <= 0 or content_length > MAX_REQUEST_BYTES:
            self._send_json(413, {"error": "Message payload is empty or too large"})
            return

        try:
            payload = json.loads(self.rfile.read(content_length))
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._send_json(400, {"error": "Request must contain valid JSON"})
            return

        message = payload.get("message") if isinstance(payload, dict) else None
        if not isinstance(message, str) or not message.strip():
            self._send_json(400, {"error": "Enter a message before sending"})
            return

        try:
            with STACK_LOCK:
                response = STACK.handle_message(
                    user_id="browser-user",
                    role="employee",
                    message=message.strip(),
                )
        except Exception as exc:
            self._send_json(500, {"error": f"Agent request failed: {exc}"})
            return

        self._send_json(
            200,
            {
                "content": response.content,
                "cache_hit": response.cache_hit,
                "guardrail_flagged": response.guardrail_flagged,
                "retrieval_sources": response.retrieval_sources,
            },
        )

    def _send_json(self, status: int, payload: dict[str, object]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        print(f"{self.address_string()} - {format % args}")


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), AgentRequestHandler)
    print(f"Company AI Agent is running at http://{HOST}:{PORT}")
    print("Press Ctrl+C to stop the server.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping the agent server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

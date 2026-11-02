"""Run the Groq-powered kushal.ai desktop chat app."""

from __future__ import annotations

import json
import os
import re
import sys
import threading
from email import policy
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import requests

PROJECT_ROOT = Path(
    os.environ.get(
        "KUSHAL_AI_PROJECT_ROOT",
        Path(sys.executable).resolve().parent
        if getattr(sys, "frozen", False)
        else Path(__file__).resolve().parents[1],
    )
).resolve()
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agent_stack.layer1_identity import GroqLLM
from agent_stack.layer2_rag import HashEmbedder, extract_document
from agent_stack.orchestrator import AgentStack, bootstrap

RESOURCE_ROOT = (
    Path(sys._MEIPASS) / "examples"
    if getattr(sys, "frozen", False)
    else Path(__file__).resolve().parent
)
HOST = "127.0.0.1"
MAX_REQUEST_BYTES = 16_384
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_UPLOAD_BATCH_BYTES = 40 * 1024 * 1024
STACK: AgentStack | None = None
STACK_LOCK = threading.Lock()
PAGE = RESOURCE_ROOT / "web" / "index.html"
ICON = RESOURCE_ROOT / "web" / "kushal_ai.ico"
UPLOADS_DIR = PROJECT_ROOT / "data" / "uploads"
DOCUMENT_REFERENCE_STOPWORDS = {
    "a", "about", "an", "and", "can", "could", "document", "file", "for",
    "from", "give", "help", "i", "in", "info", "it", "me", "my", "of",
    "on", "please", "share", "show", "summarise", "summarize", "summary",
    "tell", "the", "to", "what", "with", "you", "your", "overview",
}
FILENAME_GENERIC_WORDS = {"draft", "final", "copy", "kushal", "document", "file"}


def _find_referenced_documents(
    message: str, filenames: list[str]
) -> tuple[list[str], list[str]]:
    sources: list[str] = []
    full_document_sources: list[str] = []
    clauses = [
        clause.strip()
        for clause in re.split(r"\b(?:and\s+also|also|and)\b", message, flags=re.I)
        if clause.strip()
    ]

    for clause in clauses:
        query_words = set(re.findall(r"[a-z0-9]+", clause.lower()))
        query_words -= DOCUMENT_REFERENCE_STOPWORDS
        if not query_words:
            continue

        matches: list[tuple[int, str]] = []
        for filename in filenames:
            name_words = set(re.findall(r"[a-z0-9]+", Path(filename).stem.lower()))
            name_words -= FILENAME_GENERIC_WORDS
            overlap = len(query_words & name_words)
            if overlap:
                matches.append((overlap, filename))

        if not matches:
            continue
        highest_score = max(score for score, _filename in matches)
        best_matches = [
            filename for score, filename in matches if score == highest_score
        ]
        if len(best_matches) != 1:
            continue

        source = f"uploads/{best_matches[0]}"
        if source not in sources:
            sources.append(source)
        if (
            re.search(r"\b(?:summarize|summarise|summary|overview)\b", clause, re.I)
            and source not in full_document_sources
        ):
            full_document_sources.append(source)

    return sources, full_document_sources


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
        elif path == "/favicon.ico":
            icon = ICON.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "image/x-icon")
            self.send_header("Content-Length", str(len(icon)))
            self.end_headers()
            self.wfile.write(icon)
        elif path == "/health":
            self._send_json(200, {"status": "ok"})
        elif path == "/api/status":
            try:
                with STACK_LOCK:
                    _get_stack()
                self._send_json(
                    200,
                    {
                        "status": "ready",
                        "chat_model": GroqLLM().model,
                        "embedding_model": "local hash embeddings",
                    },
                )
            except RuntimeError as exc:
                self._send_json(503, {"status": "setup_required", "error": str(exc)})
        elif path == "/api/documents":
            documents = (
                sorted(file.name for file in UPLOADS_DIR.iterdir() if file.is_file())
                if UPLOADS_DIR.exists()
                else []
            )
            with STACK_LOCK:
                indexed_documents = (
                    {
                        source.removeprefix("uploads/")
                        for source in STACK.retriever._full_texts
                        if source.startswith("uploads/")
                    }
                    if STACK is not None
                    else set()
                )
            self._send_json(
                200,
                {
                    "documents": documents,
                    "indexed_documents": sorted(indexed_documents),
                },
            )
        else:
            self._send_json(404, {"error": "Not found"})

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/documents":
            self._upload_documents()
            return
        if path != "/api/chat":
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
            available_documents = (
                [path.name for path in UPLOADS_DIR.iterdir() if path.is_file()]
                if UPLOADS_DIR.exists()
                else []
            )
            document_sources, full_document_sources = _find_referenced_documents(
                message.strip(), available_documents
            )
            with STACK_LOCK:
                stack = _get_stack()
                response = stack.handle_message(
                    user_id="browser-user",
                    role="employee",
                    message=message.strip(),
                    document_sources=document_sources,
                    full_document_sources=full_document_sources,
                )
        except RuntimeError as exc:
            self._send_json(503, {"error": str(exc)})
            return
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

    def do_DELETE(self) -> None:
        parsed_path = urlparse(self.path)
        if parsed_path.path != "/api/documents":
            self._send_json(404, {"error": "Not found"})
            return

        filename_values = parse_qs(parsed_path.query).get("name", [])
        if len(filename_values) != 1:
            self._send_json(400, {"error": "Provide one document name to delete"})
            return
        filename = filename_values[0]
        if (
            not filename
            or filename in {".", ".."}
            or Path(filename).name != filename
            or "/" in filename
            or "\\" in filename
        ):
            self._send_json(400, {"error": "Invalid document name"})
            return

        document_path = UPLOADS_DIR / filename
        try:
            with STACK_LOCK:
                if not document_path.is_file():
                    self._send_json(404, {"error": "Document not found"})
                    return
                document_path.unlink()
                if STACK is not None:
                    STACK.retriever.remove_document(f"uploads/{filename}")
                    STACK.cache.clear()
        except OSError as exc:
            self._send_json(500, {"error": f"Could not delete the document: {exc}"})
            return

        self._send_json(200, {"deleted": filename})

    def _upload_documents(self) -> None:
        content_type = self.headers.get("Content-Type", "")
        if not content_type.lower().startswith("multipart/form-data;"):
            self._send_json(400, {"error": "Upload files using multipart/form-data"})
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json(400, {"error": "Invalid Content-Length"})
            return
        if content_length <= 0 or content_length > MAX_UPLOAD_BATCH_BYTES:
            self._send_json(413, {"error": "Upload is empty or exceeds the 40 MB limit"})
            return

        message = (
            f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n"
        ).encode("ascii") + self.rfile.read(content_length)
        multipart = BytesParser(policy=policy.default).parsebytes(message)
        if not multipart.is_multipart():
            self._send_json(400, {"error": "Invalid multipart upload"})
            return

        files: list[tuple[str, bytes]] = []
        total_bytes = 0
        try:
            for part in multipart.iter_parts():
                filename = part.get_filename()
                if not filename:
                    continue
                content = part.get_payload(decode=True)
                if not isinstance(content, bytes):
                    raise ValueError(f"{filename} has an invalid file payload")
                if len(content) > MAX_UPLOAD_BYTES:
                    raise ValueError(f"{filename} exceeds the 20 MB per-file limit")
                total_bytes += len(content)
                safe_name = Path(filename.replace("\\", "/")).name
                if safe_name in {"", ".", ".."}:
                    raise ValueError("A file has an invalid filename")
                extract_document(safe_name, content)
                files.append((safe_name, content))
            if not files:
                raise ValueError("Select at least one supported document")
        except (ValueError, UnicodeDecodeError) as exc:
            self._send_json(400, {"error": str(exc)})
            return
        if total_bytes > MAX_UPLOAD_BATCH_BYTES:
            self._send_json(413, {"error": "Upload exceeds the 40 MB total limit"})
            return

        UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        try:
            with STACK_LOCK:
                for filename, content in files:
                    (UPLOADS_DIR / filename).write_bytes(content)
                stack_was_initialized = STACK is not None
                stack = _get_stack()
                if stack_was_initialized:
                    for filename, content in files:
                        text = extract_document(filename, content)
                        stack.index_document(
                            f"uploads/{filename}", text, allowed_roles={"employee"}
                        )
                stack.cache.clear()
        except RuntimeError as exc:
            self._send_json(
                503,
                {
                    "error": (
                        f"{exc} The files were saved locally but are not indexed yet. "
                        "Set up Groq, then retry the upload."
                    )
                },
            )
            return
        except requests.RequestException as exc:
            self._send_json(
                503,
                {
                    "error": (
                        "Groq could not process these documents. Check your API "
                        f"key, model access, and internet connection. ({exc})"
                    )
                },
            )
            return
        except (OSError, ValueError, UnicodeDecodeError) as exc:
            self._send_json(400, {"error": f"Could not read the uploaded document: {exc}"})
            return
        except Exception as exc:
            self._send_json(500, {"error": f"Document indexing failed: {exc}"})
            return

        self._send_json(200, {"documents": [name for name, _ in files]})

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


def _get_stack() -> AgentStack:
    global STACK
    if STACK is not None:
        return STACK
    try:
        stack = bootstrap(llm=GroqLLM(), embedder=HashEmbedder())
        UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        for file_path in sorted(UPLOADS_DIR.iterdir()):
            if not file_path.is_file():
                continue
            text = extract_document(file_path.name, file_path.read_bytes())
            stack.index_document(
                f"uploads/{file_path.name}", text, allowed_roles={"employee"}
            )
    except requests.ConnectionError as exc:
        raise RuntimeError(
            "Could not connect to Groq. Check your internet connection and "
            "confirm GROQ_BASE_URL is correct."
        ) from exc
    except requests.HTTPError as exc:
        raise RuntimeError(
            "Groq rejected a model request. Check that your API key has access "
            "to the configured model and billing is enabled. "
            f"Details: {exc}"
        ) from exc
    except requests.RequestException as exc:
        raise RuntimeError(f"Could not use the Groq API: {exc}") from exc
    STACK = stack
    return stack


def main() -> None:
    server = ThreadingHTTPServer((HOST, 0), AgentRequestHandler)
    port = server.server_address[1]
    try:
        import webview
    except ImportError as exc:
        server.server_close()
        raise SystemExit(
            "The desktop window dependency is missing. Install the project "
            "dependencies with: pip install -e ."
        ) from exc

    server_thread = threading.Thread(
        target=server.serve_forever,
        name="kushal-ai-local-server",
        daemon=True,
    )
    server_thread.start()
    print("Starting kushal.ai desktop app…")
    print(f"Local service: http://{HOST}:{port}")
    try:
        webview.create_window(
            "kushal.ai",
            f"http://{HOST}:{port}/",
            width=1280,
            height=820,
            min_size=(900, 600),
            resizable=True,
        )
        webview.start(icon=str(ICON))
    except KeyboardInterrupt:
        print("\nClosing kushal.ai.")
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=2)


if __name__ == "__main__":
    main()

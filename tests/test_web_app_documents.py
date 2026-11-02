import importlib.util
import sys
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import requests

from agent_stack.layer1_identity.llm_client import LLMClient, LLMResponse
from agent_stack.orchestrator import AgentStack
from agent_stack.layer2_rag import HashEmbedder


def _load_web_app(module_name: str):
    app_path = Path(__file__).resolve().parents[1] / "examples" / "web_app.py"
    spec = importlib.util.spec_from_file_location(module_name, app_path)
    assert spec is not None and spec.loader is not None
    app = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = app
    spec.loader.exec_module(app)
    return app


class RecordingLLM(LLMClient):
    def __init__(self) -> None:
        self.context: str | None = None
        self.user_message: str | None = None
        self.system_prompt: str | None = None

    def complete(
        self,
        system_prompt: str,
        user_message: str,
        *,
        context: str | None = None,
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        self.context = context
        self.user_message = user_message
        self.system_prompt = system_prompt
        return LLMResponse(content="Summary from selected document.")


def test_delete_document_removes_file_and_retrieval_index(tmp_path):
    app = _load_web_app("desktop_web_app_delete_test")

    app.UPLOADS_DIR = tmp_path
    app.STACK = AgentStack(embedder=HashEmbedder())
    filename = "company guide.txt"
    (tmp_path / filename).write_text("Company holiday schedule.", encoding="utf-8")
    app.STACK.index_document(
        f"uploads/{filename}",
        "Company holiday schedule.",
        allowed_roles={"employee"},
    )

    server = ThreadingHTTPServer(("127.0.0.1", 0), app.AgentRequestHandler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        invalid_response = requests.delete(
            f"http://127.0.0.1:{server.server_port}/api/documents",
            params={"name": "../outside.txt"},
            timeout=5,
        )
        assert invalid_response.status_code == 400
        assert (tmp_path / filename).exists()

        response = requests.delete(
            f"http://127.0.0.1:{server.server_port}/api/documents",
            params={"name": filename},
            timeout=5,
        )
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()

    assert response.status_code == 200
    assert response.json() == {"deleted": filename}
    assert not (tmp_path / filename).exists()
    assert f"uploads/{filename}" not in app.STACK.retriever._full_texts
    assert not any(
        chunk.source == f"uploads/{filename}"
        for chunk in app.STACK.retriever.store._chunks.values()
    )


def test_chat_summarize_by_filename_uses_only_the_complete_referenced_document(
    tmp_path,
):
    app = _load_web_app("desktop_web_app_summary_test")
    original_uploads_dir = app.UPLOADS_DIR
    original_stack = app.STACK
    app.UPLOADS_DIR = tmp_path
    llm = RecordingLLM()
    app.STACK = AgentStack(llm=llm, embedder=HashEmbedder())

    target_name = "Kushal_InfoQ_Article_Draft.docx"
    target_text = " ".join(
        [f"InfoQ article section{i} describes a distinct research finding." for i in range(250)]
        + ["distinctivefinalfact"]
    )
    (tmp_path / target_name).write_text(target_text, encoding="utf-8")
    (tmp_path / "IBM_Guide.txt").write_text(
        "Unrelated IBM consulting guide.", encoding="utf-8"
    )
    app.STACK.index_document(
        f"uploads/{target_name}", target_text, allowed_roles={"employee"}
    )
    app.STACK.index_document(
        "uploads/IBM_Guide.txt",
        "Unrelated IBM consulting guide.",
        allowed_roles={"employee"},
    )

    server = ThreadingHTTPServer(("127.0.0.1", 0), app.AgentRequestHandler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        response = requests.post(
            f"http://127.0.0.1:{server.server_port}/api/chat",
            json={"message": "summarize the infoQ_Article"},
            timeout=5,
        )
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
        app.UPLOADS_DIR = original_uploads_dir
        app.STACK = original_stack

    assert response.status_code == 200
    assert response.json()["retrieval_sources"] == [f"uploads/{target_name}"]
    assert llm.context is not None
    assert "distinctivefinalfact" in llm.context
    assert "Unrelated IBM consulting guide." not in llm.context


def test_chat_answers_every_part_using_each_referenced_document(tmp_path):
    app = _load_web_app("desktop_web_app_multi_document_test")
    original_uploads_dir = app.UPLOADS_DIR
    original_stack = app.STACK
    app.UPLOADS_DIR = tmp_path
    llm = RecordingLLM()
    app.STACK = AgentStack(llm=llm, embedder=HashEmbedder())

    article_name = "Research_Article.txt"
    article_text = " ".join(
        [f"Article section{i} contains a separate research result." for i in range(90)]
        + ["distinctive_article_conclusion"]
    )
    guide_name = "IBM_Consulting_Core_Badges_Guide.txt"
    guide_text = (
        "IBM Consulting Core Badges recognize consulting skills. "
        "The Foundation badge requires completing the core learning path."
    )
    (tmp_path / article_name).write_text(article_text, encoding="utf-8")
    (tmp_path / guide_name).write_text(guide_text, encoding="utf-8")
    app.STACK.index_document(
        f"uploads/{article_name}", article_text, allowed_roles={"employee"}
    )
    app.STACK.index_document(
        f"uploads/{guide_name}", guide_text, allowed_roles={"employee"}
    )

    server = ThreadingHTTPServer(("127.0.0.1", 0), app.AgentRequestHandler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        message = (
            "summarize the Article and also let me know about the IBM consulting "
            "code badges"
        )
        response = requests.post(
            f"http://127.0.0.1:{server.server_port}/api/chat",
            json={"message": message},
            timeout=5,
        )
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
        app.UPLOADS_DIR = original_uploads_dir
        app.STACK = original_stack

    assert response.status_code == 200
    assert response.json()["retrieval_sources"] == [
        f"uploads/{article_name}",
        f"uploads/{guide_name}",
    ]
    assert llm.user_message == message
    assert llm.system_prompt is not None
    assert "every distinct request" in llm.system_prompt
    assert llm.context is not None
    assert "distinctive_article_conclusion" in llm.context
    assert "Foundation badge" in llm.context

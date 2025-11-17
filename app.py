# app.py (chat-only, Gemini)
import os
import re
import json
import asyncio
from dataclasses import asdict, is_dataclass
from collections.abc import Mapping, Sequence

from markupsafe import Markup, escape

from dotenv import load_dotenv
from flask import Flask, render_template, request

from browser_use import Agent, ChatGoogle

# Load .env (put GEMINI_API_KEY=... or GOOGLE_API_KEY=... there)
load_dotenv()

if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
    raise SystemExit("Set GEMINI_API_KEY or GOOGLE_API_KEY in your environment or .env file")

# Model configuration
MODEL_NAME = "gemini-flash-latest"
llm = ChatGoogle(model=MODEL_NAME)

app = Flask(__name__)


def run_agent_sync(task: str):
    """Helper to run the async agent inside a sync Flask view."""
    agent = Agent(task=task, llm=llm)
    return asyncio.run(agent.run())


def _to_builtin(value):
    """Convert Agent results (dataclasses, pydantic models, etc.) into builtin types."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if is_dataclass(value):
        return {key: _to_builtin(val) for key, val in asdict(value).items()}
    if hasattr(value, "model_dump"):
        return _to_builtin(value.model_dump())
    if hasattr(value, "dict"):
        return _to_builtin(value.dict())
    if isinstance(value, Mapping):
        return {key: _to_builtin(val) for key, val in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_to_builtin(item) for item in value]
    if hasattr(value, "__dict__"):
        return _to_builtin(vars(value))
    return str(value)


def format_result_for_display(result):
    """Return a pretty JSON string for the UI."""
    try:
        serializable = _to_builtin(result)
        return json.dumps(serializable, indent=2, ensure_ascii=False)
    except Exception:
        return str(result)


def _describe_action(action):
    if not isinstance(action, Mapping):
        return str(action)

    pieces = []
    for kind, payload in action.items():
        if not isinstance(payload, Mapping):
            pieces.append(f"{kind}: {payload}")
            continue

        if "query" in payload:
            pieces.append(f"{kind}: {payload['query']}")
        elif "index" in payload:
            pieces.append(f"{kind}: index {payload['index']}")
        elif "text" in payload and kind == "done":
            preview = payload["text"].strip().splitlines()[0]
            pieces.append(f"{kind}: {preview[:120]}{'…' if len(preview) > 120 else ''}")
        else:
            pieces.append(f"{kind}: {json.dumps(payload, ensure_ascii=False)[:120]}")
    return "; ".join(pieces)


def simple_markdown_to_html(text: str | None):
    if not text:
        return None
    html = escape(text)
    html = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html)
    html = re.sub(r"`([^`]+)`", r"<code>\1</code>", html)
    paragraphs = []
    for block in html.split("\n\n"):
        cleaned = block.strip()
        if not cleaned:
            continue
        paragraphs.append(f"<p>{cleaned.replace(chr(10), '<br>')}</p>")
    if not paragraphs:
        paragraphs.append(f"<p>{html}</p>")
    return Markup("".join(paragraphs))


def summarize_agent_history(result):
    if not isinstance(result, Mapping):
        return None

    history = result.get("history")
    if not isinstance(history, Sequence):
        return None

    steps = []
    final_text = None

    for entry in history:
        if not isinstance(entry, Mapping):
            continue
        meta = entry.get("metadata") or {}
        model_output = entry.get("model_output") or {}
        step_results = entry.get("result") or []

        actions = model_output.get("action") or []
        described_actions = [_describe_action(action) for action in actions] if actions else []

        steps.append(
            {
                "number": meta.get("step_number"),
                "status": model_output.get("evaluation_previous_goal"),
                "next_goal": model_output.get("next_goal"),
                "actions": [text for text in described_actions if text],
                "outcome": [res.get("extracted_content") for res in step_results if isinstance(res, Mapping)],
            }
        )

        for action in actions:
            if isinstance(action, Mapping) and "done" in action:
                payload = action["done"] or {}
                final_text = payload.get("text") or final_text

        for res in step_results:
            if not isinstance(res, Mapping):
                continue
            if res.get("is_done") and res.get("success"):
                final_text = res.get("extracted_content") or res.get("long_term_memory") or final_text

    summary = {"steps": steps}
    if final_text:
        summary["final_text"] = final_text
        summary["final_html"] = simple_markdown_to_html(final_text)
    return summary


@app.route("/", methods=["GET", "POST"])
def index():
    task = ""
    result = None
    result_pretty = None
    result_summary = None
    error = None

    if request.method == "POST":
        task = (request.form.get("task") or "").strip()
        if task:
            try:
                result = run_agent_sync(task)
                result_pretty = format_result_for_display(result)
                result_summary = summarize_agent_history(_to_builtin(result))
            except Exception as exc:
                error = str(exc)
        else:
            error = "Please enter a task for the agent to run."

    return render_template(
        "index.html",
        model=MODEL_NAME,
        task=task,
        result=result,
        result_pretty=result_pretty,
        result_summary=result_summary,
        error=error,
    )


# Optional: keep a simple CLI entrypoint for quick testing
async def run_cli_example():
    example_task = "search for 3rd year cs major internships in india (list at least 4)"
    agent = Agent(task=example_task, llm=llm)
    result = await agent.run()
    print("=== RESULT ===")
    print(result)


if __name__ == "__main__":
    # Run the Flask app when you do: python app.py
    # Make sure to activate your venv first.
    app.run(host="0.0.0.0", port=5000, debug=True)
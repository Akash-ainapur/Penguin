# app.py (chat-only, Gemini)
import os
import asyncio

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


@app.route("/", methods=["GET", "POST"])
def index():
    task = ""
    result = None
    error = None

    if request.method == "POST":
        task = (request.form.get("task") or "").strip()
        if task:
            try:
                result = run_agent_sync(task)
            except Exception as exc:
                error = str(exc)
        else:
            error = "Please enter a task for the agent to run."

    return render_template(
        "index.html",
        model=MODEL_NAME,
        task=task,
        result=result,
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
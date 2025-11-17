# app.py (chat-only, Gemini)
import os
import asyncio
from dotenv import load_dotenv
from browser_use import Agent, ChatGoogle

# Load .env (put GEMINI_API_KEY=... or GOOGLE_API_KEY=... there)
load_dotenv()

if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
    raise SystemExit("Set GEMINI_API_KEY or GOOGLE_API_KEY in your environment or .env file")

# Choose a model name supported by the Gemini API. You used 'gemini-flash-latest' previously.
llm = ChatGoogle(model="gemini-flash-latest")

agent = Agent(
    task="search for 3rd year cs major internships in india(list me at least 4)",
    llm=llm,
)

async def main():
    # Properly await the coroutine returned by Agent.run()
    result = await agent.run()
    print("=== RESULT ===")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())

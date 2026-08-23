import os
from datetime import date
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain.agents import create_agent

# Load environment variables FIRST, before anything reads them
load_dotenv()
print("DEBUG - GROQ key present:", os.getenv("GROQ_API_KEY") is not None)
# Strong model for reasoning-heavy tasks; swap to "openai/gpt-oss-20b" for faster/cheaper calls
llm = ChatGroq(model="openai/gpt-oss-20b")

search_tool = TavilySearch(max_results=3)

system_prompt = (
    f"Today's date is {date.today().isoformat()}. When searching for 'latest' "
    f"or 'recent' developments, use this date to determine what's actually "
    f"current, not your training data's sense of recency."
)

agent = create_agent(llm, [search_tool], system_prompt=system_prompt)

result = agent.invoke({
    "messages": [("user", "What are the latest developments in fusion energy?")]
})

for msg in result["messages"]:
    if type(msg).__name__ == "ToolMessage":
        print("=== TOOL RESULT ===")
        print(msg.content)
        print()

print("=== FINAL ANSWER ===")
print(result["messages"][-1].content)
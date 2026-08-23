from dotenv import load_dotenv
import os

load_dotenv()
print("GROQ key loaded:", os.getenv("GROQ_API_KEY") is not None)

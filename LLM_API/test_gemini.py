from google import genai
import os
from dotenv import load_dotenv

# Load .env
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# call Gemini model
client = genai.Client(api_key=api_key)

response = client.models.generate_content(
    model="gemini-2.5-flash-lite", 
    contents="Giải thích cho tôi về deep learning một cách ngắn gọn và dễ hiểu nhất."
)
print(response.text)

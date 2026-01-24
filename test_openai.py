import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

api_key = os.getenv("OPENAI_API_KEY")
print(f"API Key: {api_key[:7]}...{api_key[-4:]}")

try:
    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model="gpt-4o-mini", messages=[{"role": "user", "content": "Hello!"}]
    )

    print("✅ API 키 정상 작동")
    print(f"응답: {response.choices[0].message.content}")

except Exception as e:
    print(f"❌ API 키 오류: {e}")

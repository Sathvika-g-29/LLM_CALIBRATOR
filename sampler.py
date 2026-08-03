import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # reads from .env file automatically

# in sampler.py, change client to groq_client
groq_client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ["GROQ_API_KEY"]
)

def ask_llm(question, n=5):
    answers = []
    for i in range(n):
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": question}],
            temperature=0.8,
            max_tokens=100
        )
        answer = response.choices[0].message.content.strip()
        answers.append(answer)
        print(f"Run {i+1}: {answer}")
    return answers

if __name__ == "__main__":
    question = "Who will win the next FIFA World Cup?"
    print(f"Question: {question}\n")
    answers = ask_llm(question)
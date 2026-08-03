import os
from openai import OpenAI

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ["GROQ_API_KEY"]
)

def ask_llm(question, n=5):
    """Ask the same question n times and collect answers."""
    answers = []
    for i in range(n):
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": question}],
            temperature=0.8,  # adds randomness so answers vary
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
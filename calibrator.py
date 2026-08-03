import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Groq for chat
groq_client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ["GROQ_API_KEY"]
)

# NVIDIA for embeddings
nvidia_client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ["NVIDIA_API_KEY"]
)

def get_embedding(text):
    """Convert text to a vector of numbers."""
    response = nvidia_client.embeddings.create(
        model="nvidia/nv-embedqa-e5-v5",
        input=text,
        extra_body={"input_type": "query", "truncate": "END"}
    )
    return response.data[0].embedding
def cosine_similarity(vec1, vec2):
    """Measure how similar two vectors are. Returns 0.0 to 1.0"""
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = sum(a ** 2 for a in vec1) ** 0.5
    magnitude2 = sum(b ** 2 for b in vec2) ** 0.5
    return dot_product / (magnitude1 * magnitude2)

def compute_confidence(answers):
    """Compare all answers against each other and return a confidence score."""
    embeddings = [get_embedding(ans) for ans in answers]
    
    scores = []
    n = len(embeddings)
    for i in range(n):
        for j in range(i + 1, n):  # compare every pair
            score = cosine_similarity(embeddings[i], embeddings[j])
            scores.append(score)
    
    avg_score = sum(scores) / len(scores)
    return round(avg_score, 4)

if __name__ == "__main__":
    from sampler import ask_llm

    questions = [
        "What is the capital of France?",       # should be HIGH confidence
        "Who will win the next FIFA World Cup?"  # should be LOW confidence
    ]

    for question in questions:
        print(f"\nQuestion: {question}")
        answers = ask_llm(question)
        confidence = compute_confidence(answers)
        print(f"Confidence Score: {confidence}")
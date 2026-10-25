import os
from dataclasses import dataclass
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

@dataclass
class CalibrationResult:
    confidence: float
    label: str
    answers: list

def check(question: str, n: int = 5) -> CalibrationResult:
    """
    Check how confident an LLM is about a question.
    
    Args:
        question: The question to evaluate
        n: Number of times to sample the LLM (default 5)
    
    Returns:
        CalibrationResult with confidence score, label and answers
    
    Example:
        from hallucinationguard import check
        result = check("What is the capital of France?")
        print(result.confidence)  # 0.99
        print(result.label)       # HIGH confidence
    """
    groq_client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.environ["GROQ_API_KEY"]
    )
    nvidia_client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=os.environ["NVIDIA_API_KEY"]
    )

    # Sample LLM n times
    answers = []
    for _ in range(n):
        response = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": question}],
            temperature=0.8,
            max_tokens=100
        )
        answers.append(response.choices[0].message.content.strip())

    # Embed answers
    embeddings = []
    for ans in answers:
        response = nvidia_client.embeddings.create(
            model="nvidia/nemotron-3.5-embed-1b",
            input=ans,
            extra_body={"input_type": "query", "truncate": "END"}
        )
        embeddings.append(response.data[0].embedding)

    # Compute cosine similarity
    scores = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            dot = sum(a * b for a, b in zip(embeddings[i], embeddings[j]))
            mag1 = sum(a**2 for a in embeddings[i]) ** 0.5
            mag2 = sum(b**2 for b in embeddings[j]) ** 0.5
            scores.append(dot / (mag1 * mag2))

    confidence = round(sum(scores) / len(scores), 4)

    if confidence >= 0.97:
        label = "HIGH confidence"
    elif confidence >= 0.75:
        label = "MEDIUM confidence"
    else:
        label = "LOW confidence"

    return CalibrationResult(
        confidence=confidence,
        label=label,
        answers=answers
    )
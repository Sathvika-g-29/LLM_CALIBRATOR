import json
from sampler import ask_llm
from calibrator import compute_confidence
from reporter import classify

# Known questions with expected confidence levels
BENCHMARK = [
    # HIGH confidence expected
    {"question": "What is the capital of Japan?", "expected": "HIGH"},
    {"question": "What is 10 multiplied by 10?", "expected": "HIGH"},
    {"question": "How many continents are there?", "expected": "HIGH"},
    {"question": "What language is spoken in Brazil?", "expected": "HIGH"},

    # MEDIUM confidence expected
    {"question": "Who is the greatest footballer of all time?", "expected": "MEDIUM"},
    {"question": "Is social media good or bad for society?", "expected": "MEDIUM"},

    # LOW confidence expected
    {"question": "What will Bitcoin price be next year?", "expected": "LOW"},
    {"question": "Who will win the next US election?", "expected": "LOW"},
]

def run_benchmark():
    print("Running benchmark...\n")
    results = []
    correct = 0

    for item in BENCHMARK:
        question = item["question"]
        expected = item["expected"]

        print(f"Q: {question}")
        answers = ask_llm(question, n=5)
        confidence = compute_confidence(answers)
        label = classify(confidence)

        match = "✓" if label.startswith(expected) else "✗"
        if match == "✓":
            correct += 1

        print(f"   Score: {confidence} | Got: {label} | Expected: {expected} | {match}\n")

        results.append({
            "question": question,
            "expected": expected,
            "confidence_score": confidence,
            "label": label,
            "match": match
        })

    accuracy = round(correct / len(BENCHMARK) * 100, 1)
    print("="*50)
    print(f"Benchmark Accuracy: {correct}/{len(BENCHMARK)} ({accuracy}%)")
    print("="*50)

    with open("benchmark_results.json", "w") as f:
        json.dump({"accuracy": accuracy, "results": results}, f, indent=2)
    print("\nSaved to benchmark_results.json")

if __name__ == "__main__":
    run_benchmark()
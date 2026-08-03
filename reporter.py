import json
from datetime import datetime
from sampler import ask_llm
from calibrator import compute_confidence

def run_report(questions):
    results = []

    for question in questions:
        print(f"\nProcessing: {question}")
        answers = ask_llm(question)
        confidence = compute_confidence(answers)

        results.append({
            "question": question,
            "confidence_score": confidence,
            "label": classify(confidence),
            "answers": answers,
            "timestamp": datetime.now().isoformat()
        })

    return results

def classify(score):
    """Turn a number into a human readable label."""
    if score >= 0.95:
        return "HIGH confidence"
    elif score >= 0.80:
        return "MEDIUM confidence"
    else:
        return "LOW confidence"

def save_report(results, filename="report.json"):
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nReport saved to {filename}")

def print_summary(results):
    print("\n" + "="*50)
    print("CALIBRATION SUMMARY")
    print("="*50)
    for r in results:
        print(f"\nQ: {r['question']}")
        print(f"   Score : {r['confidence_score']}")
        print(f"   Label : {r['label']}")

if __name__ == "__main__":
    questions = [
        "What is the capital of France?",
        "Who will win the next FIFA World Cup?",
        "What is 2 + 2?",
        "What will the stock market do tomorrow?"
    ]

    results = run_report(questions)
    print_summary(results)
    save_report(results)
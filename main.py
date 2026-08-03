import sys
from sampler import ask_llm
from calibrator import compute_confidence
from reporter import classify, save_report

def run(question, n=5, save=False):
    print(f"\nQuestion: {question}")
    print(f"Sampling {n} times...\n")
    
    answers = ask_llm(question, n=n)
    confidence = compute_confidence(answers)
    label = classify(confidence)

    print("\n" + "="*50)
    print(f"Confidence Score : {confidence}")
    print(f"Label            : {label}")
    print("="*50)

    if save:
        result = [{
            "question": question,
            "confidence_score": confidence,
            "label": label,
            "answers": answers
        }]
        save_report(result, filename="report.json")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py \"your question here\" [--save]")
        sys.exit(1)

    question = sys.argv[1]
    save = "--save" in sys.argv

    run(question, save=save)
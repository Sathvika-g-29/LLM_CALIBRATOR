# HallucinationGuard

A Python package that measures how confident an LLM is about any question — by sampling it multiple times and measuring semantic consistency across responses.

## Install

```bash
pip install hallucinationguard-sathvika
```

## Usage

```python
from hallucinationguard import check

result = check("What is the capital of France?")
print(result.confidence)  # 0.99
print(result.label)       # HIGH confidence
```

## What it does

Most LLMs give one answer. This tool asks the same question 5 times, converts each answer to an embedding vector, and computes cosine similarity across all pairs.

- High similarity = consistent answers = high confidence
- Low similarity = varying answers = uncertain

## How it works

```
Question -> Sample LLM 5x -> Embed each answer ->
Cosine Similarity across all pairs -> Confidence Score
```

## Key Finding

The calibrator measures response consistency, not factual certainty. Questions the model consistently refuses to answer score unexpectedly high because refusals are semantically similar across runs. This reveals an important limitation: a confidently wrong model looks the same as a confidently right one.

## Benchmark Results

Tested against 8 questions with expected confidence levels:

- Accuracy: 75%
- HIGH confidence questions (factual): 3/4 correct
- MEDIUM confidence questions (opinion): 2/2 correct
- LOW confidence questions (future prediction): 1/2 correct

## Tech Stack

- LLM: gpt-oss-20b via Groq API
- Embeddings: NVIDIA NIM (nemotron-3.5-embed-1b)
- Similarity: Cosine similarity (pure Python, no ML libraries)
- UI: Streamlit

## Project Structure

```
llm-calibrator/
├── hallucinationguard/   # Installable package
│   ├── __init__.py
│   └── core.py
├── sampler.py            # Calls LLM n times
├── calibrator.py         # Embeds answers, computes similarity
├── reporter.py           # Labels and saves results as JSON
├── benchmark.py          # Tests against known questions
├── main.py               # CLI interface
├── app.py                # Streamlit web UI
└── setup.py              # Package configuration
```

##
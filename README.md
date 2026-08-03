# LLM Confidence Calibrator

A tool that measures how confident a Large Language Model is about any question — by sampling it multiple times and measuring semantic consistency across responses.

## What it does

Most LLMs give one answer. This tool asks the same question 5 times, converts each answer to an embedding vector, and computes cosine similarity across all pairs. High similarity = the model is consistent = high confidence. Low similarity = the model is uncertain.

## Key Finding

The calibrator measures **response consistency**, not factual certainty. Questions the model consistently refuses to answer (e.g. "Will it rain tomorrow?") score unexpectedly high because refusals are semantically similar across runs. This reveals an important limitation: a confidently wrong model looks the same as a confidently right one.

## How it works

Question → Sample LLM 5x → Embed each answer →
Cosine Similarity across all pairs → Confidence Score


## Benchmark Results

Tested against 8 questions with expected confidence levels:
- **Accuracy: 75%**
- HIGH confidence questions (factual): 3/4 correct
- MEDIUM confidence questions (opinion): 2/2 correct  
- LOW confidence questions (future prediction): 1/2 correct

Misclassified: Future predictions that trigger consistent refusals score MEDIUM instead of LOW.

## Tech Stack

- **LLM**: Llama 3.3 70B via Groq API
- **Embeddings**: NVIDIA NIM (nv-embedqa-e5-v5)
- **Similarity**: Cosine similarity (pure Python, no ML libraries)
- **UI**: Streamlit
- **Languages**: Python

## Project Structure

llm-calibrator/
├── sampler.py # Calls LLM n times, collects answers
├── calibrator.py # Embeds answers, computes cosine similarity
├── reporter.py # Labels and saves results as JSON
├── benchmark.py # Tests against known questions, reports accuracy
├── main.py # CLI interface
└── app.py # Streamlit web UI


## Run it yourself

```bash
git clone https://github.com/YOUR_USERNAME/llm-calibrator
cd llm-calibrator
pip install -r requirements.txt

# Set your keys
export GROQ_API_KEY=your_key
export NVIDIA_API_KEY=your_key

# CLI
python main.py "Is water wet?"

# Web UI
streamlit run app.py

# Benchmark
python benchmark.py
```
## Limitations

- Scores the full response, not just the core claim — verbose answers with 
  varying details score lower than they should
- Consistent refusals score HIGH even when the question is genuinely uncertain
- Does not detect confidently wrong answers
## What I learned

- Embeddings capture semantic meaning — two differently worded answers can be 95% similar
- Consistency ≠ correctness — a calibrator measures one, not the other
- This is the same embedding concept used in RAG systems, applied to uncertainty measurement

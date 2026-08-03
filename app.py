import streamlit as st
from sampler import ask_llm
from calibrator import compute_confidence
from reporter import classify

st.set_page_config(page_title="LLM Confidence Calibrator", page_icon="🎯")

st.title("🎯 LLM Confidence Calibrator")
st.write("Type any question and see how confident the LLM is about its answer.")

question = st.text_input("Enter your question:", placeholder="e.g. What is the capital of France?")

n = st.slider("Number of samples", min_value=3, max_value=10, value=5)

if st.button("Analyze"):
    if not question:
        st.warning("Please enter a question.")
    else:
        with st.spinner("Sampling LLM responses..."):
            answers = ask_llm(question, n=n)
            confidence = compute_confidence(answers)
            label = classify(confidence)

        # Score display
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Confidence Score", confidence)
        with col2:
            st.metric("Label", label)

        # Visual bar
        st.progress(confidence)

        # Color coded result
        if label.startswith("HIGH"):
            st.success("The LLM is highly consistent — likely a factual question.")
        elif label.startswith("MEDIUM"):
            st.warning("The LLM shows some variation — moderately uncertain.")
        else:
            st.error("The LLM is inconsistent — highly uncertain topic.")

        # Show individual answers
        with st.expander("See all sampled answers"):
            for i, ans in enumerate(answers, 1):
                st.markdown(f"**Run {i}:** {ans}")
                st.markdown("---")
import streamlit as st
from main import run_research_pipeline

st.set_page_config(page_title="AI Research Assistant", layout="centered")

st.title("🔬 AI Research Assistant")
st.write("Ask a complex research question and get an AI-generated answer, backed by real sources.")

query = st.text_area("💬 Enter your research question:", height=100)

if st.button("Run Research"):
    if not query.strip():
        st.warning("Please enter a research question.")
    else:
        with st.spinner("Researching and drafting answer..."):
            try:
                final_answer = run_research_pipeline(query)
                st.success("✅ Answer generated!")
                st.subheader("📌 Final Answer")
                st.markdown(final_answer)
            except Exception as e:
                st.error(f"⚠️ Error: {str(e)}")

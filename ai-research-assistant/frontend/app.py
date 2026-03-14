import streamlit as st
import requests

st.title("📚 AI Research Assistant")

backend = "http://127.0.0.1:8000"

if "history" not in st.session_state:
    st.session_state.history = []

question = st.text_input("Ask a question")

if st.button("Ask"):

    response = requests.get(
        f"{backend}/ask",
        params={"question": question}
    )

    answer = response.json()["answer"]

    st.session_state.history.append((question, answer))

for q, a in st.session_state.history:
    st.markdown(f"**You:** {q}")
    st.markdown(f"**AI:** {a}")
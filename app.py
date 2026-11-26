"""Streamlit UI: upload a PDF, get a summary, then chat about it.

Run: streamlit run app.py
"""

import streamlit as st

from pdfbot.assistant import Chat, summarise
from pdfbot.document import chunk_pages, extract_pages
from pdfbot.llm import BedrockLlm

st.set_page_config(page_title="PDF summariser", page_icon=None)
st.title("PDF summariser and chat")
st.caption("Text is sent to Amazon Bedrock in your AWS account. Do not upload documents you are not allowed to share.")

upload = st.file_uploader("PDF file", type=["pdf"])
if upload is None:
    st.stop()

if st.session_state.get("file_id") != upload.file_id:
    try:
        pages = extract_pages(upload)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    chunks = chunk_pages(pages)
    llm = BedrockLlm()
    with st.spinner(f"Summarising {len(pages)} pages ..."):
        st.session_state.summary = summarise(chunks, llm)
    st.session_state.chat = Chat(chunks, llm)
    st.session_state.messages = []
    st.session_state.file_id = upload.file_id

st.subheader("Summary")
st.markdown(st.session_state.summary)

st.subheader("Ask about the document")
for role, text in st.session_state.messages:
    st.chat_message(role).markdown(text)

if question := st.chat_input("For example: what are the main risks?"):
    st.chat_message("user").markdown(question)
    with st.spinner("Searching the document ..."):
        answer, pages = st.session_state.chat.ask(question)
    shown = answer + (f"\n\n_Sources: pages {', '.join(map(str, pages))}_" if pages else "")
    st.chat_message("assistant").markdown(shown)
    st.session_state.messages += [("user", question), ("assistant", shown)]

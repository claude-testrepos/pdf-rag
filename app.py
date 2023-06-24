import tempfile
from pathlib import Path

import streamlit as st

from chain import build_chain
from ingest import ingest

st.set_page_config(page_title="Chat with your PDFs", page_icon="📄")
st.title("📄 Chat with your PDFs")

with st.sidebar:
    files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    if files and st.button("Index"):
        with tempfile.TemporaryDirectory() as tmp:
            for f in files:
                (Path(tmp) / f.name).write_bytes(f.getbuffer())
            with st.spinner("Embedding..."):
                ingest(Path(tmp))
        st.session_state.pop("chain", None)
        st.success("Indexed!")
    model = st.selectbox("Model", ["gpt-3.5-turbo", "gpt-4"])

if "chain" not in st.session_state or st.session_state.get("model") != model:
    st.session_state.model = model
    st.session_state.chain = build_chain(model=model)
    st.session_state.messages = []

for m in st.session_state.messages:
    st.chat_message(m["role"]).write(m["content"])

if q := st.chat_input("Ask something about your documents"):
    st.chat_message("user").write(q)
    st.session_state.messages.append({"role": "user", "content": q})
    with st.spinner("Thinking..."):
        result = st.session_state.chain({"question": q})
    answer = result["answer"]
    with st.chat_message("assistant"):
        st.write(answer)
        with st.expander("Sources"):
            for doc in result["source_documents"]:
                src = Path(doc.metadata.get("source", "?")).name
                st.markdown(f"**{src}** p.{doc.metadata.get('page', 0) + 1}")
                st.caption(doc.page_content[:300] + "...")
    st.session_state.messages.append({"role": "assistant", "content": answer})

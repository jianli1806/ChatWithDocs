import streamlit as st
import os
import tempfile
from rag_engine import RAGEngine

# ==================== Page Config ====================
st.set_page_config(page_title="ChatWithDocs", page_icon="📚", layout="wide")

st.title("📚 ChatWithDocs")
st.caption("RAG-powered Document Assistant | Powered by Groq & ChromaDB")

# ==================== Session State Init ====================
if "engine" not in st.session_state:
    with st.spinner("🔧 Initializing RAG Engine (Downloading models)..."):
        st.session_state.engine = RAGEngine()
        
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! Upload a PDF document and ask me anything about it."}
    ]

# ==================== Sidebar: File Upload ====================
with st.sidebar:
    st.header("📂 Document Upload")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file and st.button("Analyze Document"):
        with st.spinner("Processing PDF... (This may take a moment)"):
            # 1. Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            # 2. Process PDF
            try:
                st.session_state.vectorstore = st.session_state.engine.process_pdf(tmp_path)
                st.success("✅ Document indexed successfully!")
            except Exception as e:
                st.error(f"Error processing file: {e}")
            finally:
                os.remove(tmp_path) # Clean up temp file

    st.markdown("---")
    st.markdown("### How it works")
    st.info(
        "1. **Upload** your PDF.\n"
        "2. The system **embeds** the text into a Vector DB.\n"
        "3. **Ask** questions and get sourced answers."
    )

# ==================== Main Chat Interface ====================

# 1. Display Chat History
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 2. Handle User Input
if prompt := st.chat_input("Ask a question about your document..."):
    
    # Check if document is loaded
    if not st.session_state.vectorstore:
        st.error("⚠️ Please upload and analyze a document first!")
    else:
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching knowledge base..."):
                try:
                    response = st.session_state.engine.chat(prompt, st.session_state.vectorstore)
                    answer = response["answer"]
                    
                    # (Optional) Display Sources in an expander
                    with st.expander("View Source Context"):
                        for i, doc in enumerate(response["context"]):
                            st.markdown(f"**Chunk {i+1}:**")
                            st.caption(doc.page_content[:300] + "...")
                    
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                except Exception as e:
                    st.error(f"An error occurred: {e}")
import streamlit as st
import chromadb
import fitz
import os
import tempfile
import warnings
warnings.filterwarnings("ignore")

from openai import OpenAI
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END
from agent_state import AgentState
from agent_nodes import router_node, retriever_node, generator_node, meta_node, clarifier_node

load_dotenv()

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="NVIDIA Document Agent",
    page_icon="docs/nvidia-logo.png" if os.path.exists("docs/nvidia-logo.png") else "🤖",
    layout="wide"
)

# ── NVIDIA client ────────────────────────────────────────────
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY")
)

# ── Build LangGraph agent ────────────────────────────────────
def route_decision(state: AgentState) -> str:
    decision = state["decision"]
    if decision == "meta":
        return "meta"
    elif decision == "clarify":
        return "clarifier"
    return "retriever"

@st.cache_resource
def build_agent():
    graph = StateGraph(AgentState)
    graph.add_node("router", router_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("generator", generator_node)
    graph.add_node("meta", meta_node)
    graph.add_node("clarifier", clarifier_node)
    graph.set_entry_point("router")
    graph.add_conditional_edges("router", route_decision, {
        "retriever": "retriever",
        "meta": "meta",
        "clarifier": "clarifier"
    })
    graph.add_edge("retriever", "generator")
    graph.add_edge("generator", END)
    graph.add_edge("meta", END)
    graph.add_edge("clarifier", END)
    return graph.compile()

# ── PDF ingestion ────────────────────────────────────────────
def ingest_pdf(file_bytes, filename):
    """Load, chunk, embed and store a PDF into ChromaDB."""
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    
    # Extract text
    doc = fitz.open(tmp_path)
    full_text = ""
    for page_num in range(doc.page_count):
        page = doc[page_num]
        full_text += f"\n--- Page {page_num + 1} ---\n{page.get_text()}"
    doc.close()
    os.unlink(tmp_path)
    
    # Chunk text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )
    chunks = splitter.split_text(full_text)
    
    # Store in ChromaDB
    db_client = chromadb.PersistentClient(path="./chroma_db")
    
    # Clear old collection and create fresh
    try:
        db_client.delete_collection("nvidia_docs")
    except:
        pass
    collection = db_client.get_or_create_collection(name="nvidia_docs")
    
    # Embed and store each chunk
    progress = st.progress(0, text="Embedding chunks...")
    
    for i, chunk in enumerate(chunks):
        embedding_response = client.embeddings.create(
            model="nvidia/nv-embedqa-e5-v5",
            input=chunk,
            encoding_format="float",
            extra_body={"input_type": "passage", "truncate": "NONE"}
        )
        embedding = embedding_response.data[0].embedding
        
        collection.add(
            ids=[f"chunk_{i}"],
            embeddings=[embedding],
            documents=[chunk],
            metadatas=[{"chunk_index": i, "source": filename}]
        )
        
        progress.progress((i + 1) / len(chunks), 
                         text=f"Embedding chunks... {i+1}/{len(chunks)}")
    
    progress.empty()
    return len(chunks)

# ── Session state init ───────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pdf_loaded" not in st.session_state:
    st.session_state.pdf_loaded = False

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = ""

if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.title("NVIDIA Document Agent")
    st.caption("Powered by Nemotron + ChromaDB + LangGraph")
    
    st.divider()
    
    st.subheader("Upload a document")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type="pdf",
        help="Upload any PDF — research papers, reports, manuals"
    )
    
    if uploaded_file is not None:
        if uploaded_file.name != st.session_state.pdf_name:
            with st.spinner(f"Processing {uploaded_file.name}..."):
                chunk_count = ingest_pdf(
                    uploaded_file.read(),
                    uploaded_file.name
                )
            st.session_state.pdf_loaded = True
            st.session_state.pdf_name = uploaded_file.name
            st.session_state.chunk_count = chunk_count
            st.session_state.messages = []  # Clear chat for new doc
            st.success(f"Ready! {chunk_count} chunks indexed.")
    
    if st.session_state.pdf_loaded:
        st.divider()
        st.subheader("Document info")
        st.write(f"File: `{st.session_state.pdf_name}`")
        st.write(f"Chunks: `{st.session_state.chunk_count}`")
        
        if st.button("Clear conversation"):
            st.session_state.messages = []
            st.rerun()
    
    st.divider()
    st.caption("Built with NVIDIA NIM + LangGraph")
    st.caption("github.com/Jenny3306/NVIDIA-Doc-Agent")

# ── Main chat area ───────────────────────────────────────────
st.title("Document Q&A Agent")

if not st.session_state.pdf_loaded:
    # Welcome screen
    st.info("Upload a PDF in the sidebar to get started.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model", "Nemotron 3 Nano")
    with col2:
        st.metric("Vector DB", "ChromaDB")
    with col3:
        st.metric("Agent", "LangGraph")
    
    st.markdown("""
    ### What can this agent do?
    - **Answer questions** from any PDF you upload
    - **Cite sources** — tells you which chunk the answer came from
    - **Remember context** — ask follow-up questions naturally
    - **Handle summaries** — ask it to summarize what it told you
    - **Stay honest** — refuses to make up answers not in the document
    """)

else:
    # Chat interface
    st.caption(f"Chatting about: **{st.session_state.pdf_name}**")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "decision" in message:
                st.caption(f"Agent mode: {message['decision'].upper()}")
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your document..."):
        
        # Show user message
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Build chat history for agent
        chat_history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages[:-1]  # Exclude current message
            if m["role"] in ["user", "assistant"]
        ]
        
        # Run agent
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                agent = build_agent()
                
                initial_state: AgentState = {
                    "question": prompt,
                    "retrieved_chunks": [],
                    "answer": "",
                    "decision": "",
                    "iterations": 0,
                    "chat_history": chat_history[-12:],
                    "retrieval_confidence": 0.0
                }
                
                final_state = agent.invoke(initial_state)
                answer = final_state["answer"]
                decision = final_state["decision"]
            
            st.markdown(answer)
            st.caption(f"Agent mode: {decision.upper()}")
        
        # Save to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "decision": decision
        })
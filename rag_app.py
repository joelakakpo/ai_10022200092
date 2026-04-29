#Name: Joel Selorm Akakpo
#Roll no: 10022200092
#Project: Ghana Election Data RAG System

import streamlit as st
import os
import json
import re
import requests
from dotenv import load_dotenv
from openai import OpenAI
import numpy as np

# Try to import PDF reader
try:
    from pypdf import PdfReader
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# Page Configuration
st.set_page_config(
    page_title="ACITY RAG - Ghana Data",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Clean White & Grey Theme
st.markdown("""
    <style>
    /* Main Background */
    body, .stApp {
        background: #f8f9fa;
        color: #2c3e50;
    }
    
    .stMainBlockContainer {
        background: transparent;
        padding: 2rem 1.5rem;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #1a5490;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    
    h1 { font-size: 2.5rem; }
    h2 { font-size: 1.6rem; margin-top: 25px; }
    h3 { font-size: 1.2rem; }
    
    p, span { color: #555; }
    
    /* Input Fields */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div > select,
    .stSlider > div > div {
        background-color: #fff !important;
        border: 2px solid #ddd !important;
        color: #333 !important;
        border-radius: 8px !important;
        padding: 12px 14px !important;
        font-size: 14px !important;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus,
    .stSelectbox > div > div > select:focus {
        border-color: #1a5490 !important;
        box-shadow: 0 0 10px rgba(26, 84, 144, 0.1) !important;
        background-color: #fff !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: #1a5490;
        color: #fff !important;
        border: 2px solid #1a5490 !important;
        border-radius: 8px !important;
        padding: 12px 28px !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        transition: all 0.3s ease !important;
        cursor: pointer;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    
    .stButton > button:hover {
        background: #0d3a6e !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        transform: translateY(-2px);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Cards and Boxes */
    .chunk-display {
        background: #fff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
        font-family: 'Courier New', monospace;
        font-size: 12.5px;
        color: #333;
        max-height: 150px;
        overflow-y: auto;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        line-height: 1.5;
    }
    
    .chunk-display::-webkit-scrollbar {
        width: 6px;
    }
    
    .chunk-display::-webkit-scrollbar-track {
        background: #f0f0f0;
    }
    
    .chunk-display::-webkit-scrollbar-thumb {
        background: #bbb;
        border-radius: 3px;
    }
    
    .retrieval-box {
        background: #fff;
        border: 1px solid #e0e0e0;
        border-left: 4px solid #1a5490;
        border-radius: 8px;
        padding: 14px;
        margin: 10px 0;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
    }
    
    .final-answer-box {
        background: #fff;
        border: 2px solid #1a5490;
        border-radius: 8px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    }
    
    .stat-box {
        background: #f0f4f8;
        border: 1px solid #ddd;
        border-left: 4px solid #1a5490;
        padding: 12px;
        border-radius: 6px;
        margin: 6px 0;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #f5f5f5;
        border-right: 1px solid #ddd;
    }
    
    [data-testid="stSidebar"] label {
        color: #1a5490 !important;
        font-weight: 600 !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #f0f4f8 !important;
        border: 1px solid #ddd !important;
        border-radius: 6px !important;
        padding: 10px 12px !important;
        color: #333 !important;
        font-weight: 600 !important;
        transition: all 0.3s ease;
    }
    
    .streamlit-expanderHeader:hover {
        background-color: #e8f0f8 !important;
        border-color: #1a5490 !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.08);
    }
    
    .streamlit-expanderContent {
        border: 1px solid #ddd !important;
        border-top: 0 !important;
        border-radius: 0 0 6px 6px !important;
        background-color: #fff !important;
    }
    
    /* Metrics */
    .metric-value {
        color: #1a5490;
        font-weight: 700;
        font-size: 1.1rem;
    }
    
    .metric-label {
        color: #666;
        font-weight: 600;
        font-size: 0.85rem;
    }
    
    /* Horizontal Line */
    hr {
        border-color: #ddd !important;
        opacity: 1 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Load environment variables
load_dotenv()

# Initialize Groq client
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

@st.cache_data
def load_chunks():
    """Load chunks from JSON file"""
    try:
        with open("data/processed/chunks.json", "r") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Could not load chunks: {e}")
        return []

@st.cache_data
def load_full_text():
    """Load full text from file"""
    try:
        with open("data/processed/full_text.txt", "r") as f:
            return f.read()
    except Exception as e:
        st.error(f"Could not load text: {e}")
        return ""

@st.cache_data
def load_pdf_text():
    """Load and extract text from 2025 Budget PDF"""
    try:
        if not PDF_AVAILABLE:
            return ""
        
        pdf_path = "budget.pdf"
        if not os.path.exists(pdf_path):
            import requests
            st.info("⏳ Downloading 2025 Budget PDF...")
            pdf_url = "https://mofep.gov.gh/sites/default/files/budget-statements/2025-Budget-Statement-and-Economic-Policy_v4.pdf"
            response = requests.get(pdf_url, timeout=30)
            if response.status_code == 200:
                with open(pdf_path, "wb") as f:
                    f.write(response.content)
        
        if os.path.exists(pdf_path):
            reader = PdfReader(pdf_path)
            pdf_text = ""
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    pdf_text += text + "\n"
            return re.sub(r'\s+', ' ', pdf_text)  # Clean whitespace
    except Exception as e:
        st.warning(f"Could not load PDF: {str(e)[:50]}")
    return ""

def chunk_pdf_text(text, chunk_size=400, overlap=50):
    """Chunk PDF text into retrieval units"""
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start = end - overlap
    return chunks

def retrieve_chunks(query, election_chunks, pdf_chunks, top_k=4):
    """Retrieve chunks from both election data and PDF - more lenient matching"""
    # Combine both datasets
    all_chunks = election_chunks + pdf_chunks
    if not all_chunks:
        return []
    
    query_terms = [t.lower() for t in query.split() if len(t) > 2]
    if not query_terms:
        # If query is too short or only short words, return first chunks
        return [(chunk, 0.5) for chunk in all_chunks[:top_k]]
    
    scores = []
    
    for chunk in all_chunks:
        chunk_lower = chunk.lower()
        score = 0
        matches = 0
        
        # Count how many query terms appear in the chunk
        for term in query_terms:
            if term in chunk_lower:
                matches += 1
                # Score based on occurrences
                count = chunk_lower.count(term)
                score += min(count * 5, 30)  # Cap individual term score
        
        # Bonus if this is from PDF (more relevant for budget questions)
        is_pdf_chunk = chunk in pdf_chunks
        if is_pdf_chunk and any(term in ['budget', 'expenditure', 'revenue', 'spending', '2025', '2024'] for term in query_terms):
            score += 20
        
        # Always include chunks if they have at least 1 match
        if matches > 0 or score > 0:
            # Normalize score based on match ratio
            normalized_score = min(1.0, (matches / len(query_terms)) * 0.8 + (score / 100) * 0.2)
            scores.append((chunk, max(0.1, normalized_score)))
        elif not scores:
            # If no matches at all, include first chunk with low score
            scores.append((chunk, 0.1))
    
    # Sort by score and return top_k
    sorted_results = sorted(scores, key=lambda x: x[1], reverse=True)[:max(top_k, 1)]
    return sorted_results

def ask_llm_with_context(prompt: str, context: str, chunks_used: int, top_k: int) -> str:
    """Send a prompt with dataset context to Groq LLM"""
    try:
        if not context.strip():
            return "ℹ️ No relevant information found in the dataset for your query. Please try different keywords related to Ghana's election data, budget, or economic information."
        
        system_msg = f"""You are an assistant specialized in Ghana's election data and 2025 budget information.

**CRITICAL INSTRUCTIONS:**
1. ONLY use the provided dataset chunks to answer - do NOT use external knowledge
2. Extract EXACT data from chunks: candidate names, vote counts, percentages, regions, years
3. If specific information is not in the chunks provided, clearly say: "This information is not available in the retrieved chunks"
4. Be precise with numbers, percentages, and dates - quote them exactly from the data
5. Always cite the source region/year/party when relevant
6. Structure your answer to show which chunk data came from

**DATASET RETRIEVED ({chunks_used} chunks, Top-K={top_k}):**
{context}

Answer ONLY based on the above information. If the chunks contain the answer, provide it with exact numbers and details."""
        
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,  # Even lower temp for maximum factual accuracy
            max_tokens=1200
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ Error generating response: {str(e)[:100]}"

# Load data
chunks = load_chunks()
full_text = load_full_text()
pdf_text = load_pdf_text()
pdf_chunks = chunk_pdf_text(pdf_text) if pdf_text else []

total_chunks = len(chunks) + len(pdf_chunks)

# ===== TWO COLUMN LAYOUT: LEFT (Configuration) | RIGHT (Features) =====
left_col, right_col = st.columns([1.2, 2], gap="large")

# ===== LEFT COLUMN: CONFIGURATION PANEL =====
with left_col:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1a5490 0%, #0d3a6e 100%); 
                border-radius: 12px; padding: 25px; color: white; text-align: center; 
                box-shadow: 0 4px 15px rgba(0,0,0,0.1); margin-bottom: 20px;">
        <h2 style="color: white; margin-top: 0; font-size: 1.8rem;">⚙️ Configuration</h2>
        <p style="color: #e0e0e0; font-size: 13px; margin-bottom: 0;">Customize retrieval settings</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Retrieval Top-K
    st.markdown("### 🎯 Retrieval Top-K")
    top_k = st.slider(
        "Chunks to retrieve:",
        min_value=1,
        max_value=min(10, total_chunks),
        value=4,
        help="More chunks = more context",
        label_visibility="collapsed",
        key="topk_slider"
    )
    
    st.markdown(f"""
    <div class="stat-box">
    <b>Setting: {top_k} chunks</b><br>
    <span style="font-size: 12px; color: #666;">This many chunks will be used</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Data Sources Info
    st.markdown("### 📊 Data Sources")
    
    st.markdown(f"""
    <div class="stat-box">
    <b style="color: #1a5490;">🗳️ Election Data</b><br>
    <span style="font-size: 12px;">Chunks: <b>{len(chunks)}</b></span><br>
    <span style="font-size: 12px;">Years: <b>2000-2020</b></span><br>
    <span style="font-size: 12px;">Source: <code>Ghana_Election_Result.csv</code></span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("")
    
    st.markdown(f"""
    <div class="stat-box">
    <b style="color: #1a5490;">💰 2025 Budget</b><br>
    <span style="font-size: 12px;">Chunks: <b>{len(pdf_chunks)}</b></span><br>
    <span style="font-size: 12px;">Status: <b>{"✅ Loaded" if pdf_chunks else "⏳ Loading..."}</b></span><br>
    <span style="font-size: 12px;">Source: <code>MOFEP PDF</code></span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Overall Stats
    st.markdown("### 📈 Overview")
    
    st.markdown(f"""
    <div class="stat-box" style="background: linear-gradient(135deg, #f0f4f8 0%, #e8f0f8 100%); border-left-color: #1a5490; text-align: center;">
    <span class="metric-label">TOTAL CHUNKS</span><br>
    <span class="metric-value">{total_chunks}</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("")
    
    st.markdown("""
    <div class="stat-box" style="background: linear-gradient(135deg, #f0f4f8 0%, #e8f0f8 100%); border-left-color: #1a5490;">
    <b style="color: #1a5490;">How It Works:</b><br>
    <span style="font-size: 12px; color: #666;">
    1. Enter your question<br>
    2. Search both datasets<br>
    3. Retrieve top chunks<br>
    4. Generate AI answer<br>
    5. View results
    </span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### ✨ Status")
    st.markdown("""
    <div class="stat-box" style="background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%); border-left-color: #28a745;">
    <b style="color: #155724;">✅ SYSTEM READY</b><br>
    <span style="font-size: 12px; color: #666;">All components initialized</span>
    </div>
    """, unsafe_allow_html=True)

# ===== RIGHT COLUMN: FEATURES & RESULTS =====
with right_col:
    # Header
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1a5490 0%, #0d3a6e 100%); 
                border-radius: 12px; padding: 25px; color: white; margin-bottom: 20px;">
        <div style="text-align: center;">
            <h1 style="color: white; margin-bottom: 5px; font-size: 2rem;">📊 ACITY RAG</h1>
            <p style="color: #e0e0e0; font-size: 13px; margin: 0;">GHANA DATA ANALYSIS</p>
            <p style="color: #b3d9ff; font-size: 12px; margin-top: 8px;">Election Data + 2025 Budget | Data-Driven Responses</p>
        </div>
    </div>
    
    # Top Stats Bar
    """, unsafe_allow_html=True)
    
    col_chunk1, col_chunk2, col_chunk3 = st.columns(3)
    
    with col_chunk1:
        st.markdown(f"""
        <div class="stat-box" style="text-align: center; background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); border-left-color: #f59e0b;">
        <span class="metric-label">📦 TOTAL CHUNKS</span><br>
        <span class="metric-value">{total_chunks}</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col_chunk2:
        st.markdown(f"""
        <div class="stat-box" style="text-align: center; background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); border-left-color: #0ea5e9;">
        <span class="metric-label">🗳️ ELECTION DATA</span><br>
        <span class="metric-value">{len(chunks)}</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col_chunk3:
        st.markdown(f"""
        <div class="stat-box" style="text-align: center; background: linear-gradient(135deg, #e9d5ff 0%, #d8b4fe 100%); border-left-color: #a855f7;">
        <span class="metric-label">💰 BUDGET DATA</span><br>
        <span class="metric-value">{len(pdf_chunks)}</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Query Section
    st.markdown("## ❓ Ask Your Question")
    
    user_query = st.text_area(
        "Your Query:",
        placeholder="E.g., 'Which party won in 2020?' | 'Show me Greater Accra results' | 'How much was budgeted for education?'",
        height=100,
        label_visibility="collapsed",
        key="query_input"
    )
    
    st.markdown("")
    
    # Send Button
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    with col_btn2:
        send_button = st.button("🚀 SEND QUERY", use_container_width=True, key="send_btn", help="Send your query to retrieve relevant chunks")
    
    st.markdown("")
    
    # Quick Examples
    st.markdown("### 💡 Quick Examples:")
    col_ex1, col_ex2, col_ex3 = st.columns(3)
    
    quick_query_1 = None
    quick_query_2 = None
    quick_query_3 = None
    
    with col_ex1:
        if st.button("🗳️ 2020 Winner", use_container_width=True, key="ex1"):
            quick_query_1 = "Which party won the most votes in 2020?"
    
    with col_ex2:
        if st.button("📊 Top Results", use_container_width=True, key="ex2"):
            quick_query_2 = "Show top voting results from latest election"
    
    with col_ex3:
        if st.button("🌍 By Region", use_container_width=True, key="ex3"):
            quick_query_3 = "What are the results by region?"
    
    # Handle quick queries
    if quick_query_1:
        user_query = quick_query_1
        send_button = True
    elif quick_query_2:
        user_query = quick_query_2
        send_button = True
    elif quick_query_3:
        user_query = quick_query_3
        send_button = True
    
    # ===== PROCESS QUERY =====
    if (user_query and user_query.strip() and send_button):
        st.markdown("---")
        
        # ===== RETRIEVAL PIPELINE =====
        st.markdown("## 🔍 RETRIEVAL RESULTS")
        
        with st.spinner("🔎 Searching both datasets..."):
            retrieved_data = retrieve_chunks(user_query, chunks, pdf_chunks, top_k=top_k)
            retrieved_chunks = [item[0] for item in retrieved_data]
            relevance_scores = [item[1] for item in retrieved_data]
            context = "\n\n---CHUNK SEPARATOR---\n\n".join(retrieved_chunks)
        
        # Display retrieval summary
        col_ret1, col_ret2, col_ret3 = st.columns(3)
        
        with col_ret1:
            st.markdown(f"""
            <div class="stat-box" style="text-align: center;">
            <span class="metric-label">✓ CHUNKS FOUND</span><br>
            <span class="metric-value">{len(retrieved_chunks)}</span>
            </div>
            """, unsafe_allow_html=True)
        
        with col_ret2:
            st.markdown(f"""
            <div class="stat-box" style="text-align: center;">
            <span class="metric-label">🎯 TOP-K</span><br>
            <span class="metric-value">{top_k}</span>
            </div>
            """, unsafe_allow_html=True)
        
        with col_ret3:
            avg_relevance = np.mean(relevance_scores) if relevance_scores else 0
            st.markdown(f"""
            <div class="stat-box" style="text-align: center;">
            <span class="metric-label">📈 AVG RELEVANCE</span><br>
            <span class="metric-value">{avg_relevance:.1%}</span>
            </div>
            """, unsafe_allow_html=True)
        
        # ===== EXPANDABLE CHUNKS SECTION =====
        st.markdown("### 📂 Retrieved Chunks")
        
        if retrieved_data:
            st.markdown(f"**{len(retrieved_data)} chunks found from the dataset:**")
            
            # Create expandable sections for each chunk
            for idx, (chunk, score) in enumerate(retrieved_data, 1):
                relevance_bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
                relevance_percentage = f"{score:.1%}"
                is_budget = chunk in pdf_chunks
                source = "💰 Budget" if is_budget else "🗳️ Election"
                
                with st.expander(
                    f"{source} | Chunk {idx} | Match: {relevance_percentage}",
                    expanded=(idx == 1)  # First chunk expanded by default
                ):
                    st.markdown(f"""
                    <div style="background-color: #f9f9f9; border-left: 4px solid #1a5490; padding: 10px; border-radius: 6px;">
                    <span style="color: #666; font-size: 11px;">Relevance: {relevance_bar}</span>
                    <p style="margin-top: 8px; color: #333; line-height: 1.6; font-size: 13px;">{chunk}</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # ===== GENERATION SECTION =====
        st.markdown("## ✨ RESPONSE")
        
        with st.spinner("⏳ Generating answer..."):
            final_answer = ask_llm_with_context(user_query, context, len(retrieved_chunks), top_k)
        
        st.markdown(f"""
        <div class="final-answer-box">
        <h3 style="color: #1a5490; margin-top: 0;">📌 Answer</h3>
        <p style="color: #333; line-height: 1.8; font-size: 15px; margin: 12px 0;">{final_answer}</p>
        <hr style="border-color: #ddd; opacity: 0.5; margin: 15px 0;">
        <p style="font-size: 12px; color: #666; margin: 0;">
        ✓ <b>Sources:</b> Ghana Elections + 2025 Budget | <b>Chunks:</b> {len(retrieved_chunks)} | <b>Top-K:</b> {top_k} | <b>Relevance:</b> {avg_relevance:.0%}
        </p>
        </div>
        """, unsafe_allow_html=True)
    
    # ===== FOOTER =====
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #999; font-size: 11px; padding: 15px 0;">
            <p style="margin: 0;">📊 ACITY RAG | Ghana Data Analysis | Data-Driven Intelligence</p>
            <p style="font-size: 10px; margin-top: 5px;">Election Data (2000-2020) + 2025 Budget Statement</p>
        </div>
    """, unsafe_allow_html=True)

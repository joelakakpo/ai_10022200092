#Name: Joel Selorm Akakpo
#Roll no: 10022200092


import streamlit as st
import os
from dotenv import load_dotenv
from openai import OpenAI

# Page Configuration
st.set_page_config(
    page_title="ACITY Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Bright Theme CSS
st.markdown("""
    <style>
    :root {
        --primary-color: #2563eb;   /* Bright blue */
        --secondary-color: #7c3aed; /* Purple accent */
        --accent-color: #06b6d4;    /* Teal accent */
        --bg-color: #f9fafb;        /* Light background */
        --surface-color: #ffffff;   /* White cards */
        --surface-alt: #f1f5f9;     /* Light gray alt */
        --text-primary: #1e293b;    /* Dark text */
        --text-secondary: #475569;  /* Muted text */
        --border-color: #e2e8f0;    /* Light border */
    }
    
    /* Main background */
    body, .stApp {
        background: linear-gradient(135deg, #f9fafb 0%, #f1f5f9 100%);
        color: var(--text-primary);
    }
    
    /* Header styling */
    h1, h2, h3 {
        color: var(--primary-color);
        font-weight: 700;
        text-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    
    /* Text styling */
    p, span {
        color: var(--text-secondary);
    }
    
    /* Input fields */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div > select {
        background-color: var(--surface-color) !important;
        border: 2px solid var(--border-color) !important;
        color: var(--text-primary) !important;
        border-radius: 8px !important;
        padding: 10px 12px !important;
        font-size: 15px !important;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:hover,
    .stTextArea > div > div > textarea:hover,
    .stSelectbox > div > div > select:hover {
        border-color: var(--primary-color) !important;
        box-shadow: 0 0 10px rgba(37, 99, 235, 0.2) !important;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus,
    .stSelectbox > div > div > select:focus {
        border-color: var(--secondary-color) !important;
        box-shadow: 0 0 15px rgba(124, 58, 237, 0.3) !important;
        background-color: var(--surface-alt) !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 12px 28px !important;
        font-weight: 600 !important;
        font-size: 15px !important;
        transition: all 0.3s ease !important;
        cursor: pointer;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.25);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 18px rgba(124, 58, 237, 0.3);
        background: linear-gradient(135deg, #3b82f6 0%, #7c3aed 100%) !important;
    }
    
    /* Response box styling */
    .response-box {
        background: var(--surface-color);
        border-left: 4px solid var(--primary-color);
        border-radius: 8px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* Messages */
    .stSuccess {
        background-color: rgba(34, 197, 94, 0.1) !important;
        border-left: 4px solid #22c55e !important;
        border-radius: 8px !important;
    }
    .stWarning {
        background-color: rgba(234, 179, 8, 0.1) !important;
        border-left: 4px solid #eab308 !important;
        border-radius: 8px !important;
    }
    .stError {
        background-color: rgba(239, 68, 68, 0.1) !important;
        border-left: 4px solid #ef4444 !important;
        border-radius: 8px !important;
    }
    .stInfo {
        background-color: rgba(59, 130, 246, 0.1) !important;
        border-left: 4px solid #3b82f6 !important;
        border-radius: 8px !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f9fafb 100%);
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: var(--text-secondary);
    }
    
    /* Spinner */
    .stSpinner > div > div {
        border-top-color: var(--primary-color) !important;
    }
    
    /* Labels */
    label {
        color: var(--text-primary) !important;
        font-weight: 600 !important;
        margin-bottom: 8px !important;
    }
    
    /* Divider */
    hr {
        border-color: var(--border-color) !important;
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

def ask_llm(prompt: str, model: str = "llama-3.3-70b-versatile") -> str:
    """Send a prompt to Groq LLM and return the response text."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful, knowledgeable assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

# Header
st.markdown("""
    <div style="text-align: center; margin-bottom: 30px; padding: 20px 0;">
        <h1 style="font-size: 2.5rem; margin-bottom: 10px;">🤖 ACITY ASSISTANT</h1>
        <p style="font-size: 16px; color: #cbd5e1; margin: 0;">RAG-powered Intelligent Chatbot</p>
        <p style="font-size: 13px; color: #94a3b8; margin-top: 5px;">Powered by Groq LLaMA 3.3 70B</p>
    </div>
    <hr style="margin-bottom: 30px; border-color: #475569;">
""", unsafe_allow_html=True)

# Main content in columns
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("**⚙️ Configuration**")
    model_choice = st.selectbox(
        "Select Model:",
        ["llama-3.3-70b-versatile", "gemma-7b-it"],
        help="Choose the LLM model for your queries"
    )

# Input section
st.markdown("---")
st.markdown("**📝 Your Question**")
user_input = st.text_area(
    "Enter your question:",
    placeholder="Ask anything about your topic...",
    height=120,
    label_visibility="collapsed"
)

# Button and action
col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    send_button = st.button(
        "🚀 Send",
        use_container_width=True,
        key="send_btn"
    )

with col2:
    clear_button = st.button(
        "🔄 Clear",
        use_container_width=True,
        key="clear_btn"
    )

# Handle button clicks
if clear_button:
    st.session_state.clear()
    st.rerun()

if send_button:
    if user_input.strip():
        st.markdown("---")
        col_spinner, col_empty = st.columns([1, 3])
        with col_spinner:
            with st.spinner("⏳ Thinking..."):
                answer = ask_llm(user_input, model=model_choice)
        
        # Display response
        st.markdown("**✨ Response**")
        st.markdown(f"""
            <div class="response-box">
                <p style="color: #f1f5f9; line-height: 1.8; font-size: 15px;">{answer}</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please enter a question before sending.")

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #94a3b8; font-size: 12px; margin-top: 20px; padding: 15px 0;">
        <p>✨ ACITY Assistant | Powered by Groq API | Advanced RAG System</p>
    </div>
""", unsafe_allow_html=True)

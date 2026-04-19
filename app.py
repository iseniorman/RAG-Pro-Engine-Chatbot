import os
import shutil
import time
import streamlit as st
from src.engine import get_chat_engine
from src.model_loader import initialise_llm, get_embedding_model
from src.config import DATA_PATH, VECTOR_STORE_PATH, SIMILARITY_TOP_K, LLM_SYSTEM_PROMPT

# ==========================================
# 1. PAGE CONFIG & ULTRA-MODERN STYLING
# ==========================================
st.set_page_config(
    page_title="RAG Pro Engine",
    layout="wide",
    page_icon="🧠",
    initial_sidebar_state="expanded"
)

# Ultra-modern gradient UI with glassmorphism effects
st.markdown("""
<style>
    /* Base dark theme with gradient background */
    body {
        background: linear-gradient(135deg, #0f1117 0%, #1a1b23 100%);
        color: #ffffff;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        min-height: 100vh;
    }
    
    /* Main container with glass effect */
    .main .block-container {
        max-width: 850px;
        padding-top: 2rem;
        padding-bottom: 4rem;
        background: rgba(26, 27, 35, 0.7);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    /* Chat message styling with glass effect */
    .stChatMessage {
        background: rgba(42, 43, 53, 0.6);
        border-radius: 16px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(5px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    
    /* User message specific styling */
    .stChatMessage[data-testid="chat-message-USER"] {
        background: rgba(42, 43, 53, 0.8);
        border-left: 4px solid #4a90e2;
    }
    
    /* Assistant message specific styling */
    .stChatMessage[data-testid="chat-message-AI"] {
        background: rgba(26, 27, 35, 0.7);
        border-left: 4px solid #8b5cf6;
        box-shadow: 0 4px 20px rgba(139, 92, 246, 0.2);
    }
    
    /* Status indicator styling */
    .stStatusWidget {
        background: rgba(26, 27, 35, 0.7);
        border-radius: 12px;
        padding: 0.75rem 1.25rem;
        backdrop-filter: blur(5px);
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Source card styling with glass effect */
    .source-card {
        background: rgba(37, 38, 48, 0.8);
        border-radius: 12px;
        padding: 1rem;
        margin-top: 0.75rem;
        border-left: 4px solid #8b5cf6;
        backdrop-filter: blur(5px);
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Sidebar styling with glass effect */
    .css-1d391kg {
        background: rgba(26, 27, 35, 0.8);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Input field styling with glass effect */
    .stTextInput > div > div > input {
        background: rgba(37, 38, 48, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        color: white;
        backdrop-filter: blur(5px);
    }
    
    /* Button styling with gradient */
    .stButton > button {
        background: linear-gradient(135deg, #4a90e2 0%, #357abd 100%);
        color: white;
        border-radius: 12px;
        border: none;
        padding: 0.6rem 1.25rem;
        font-weight: 500;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(74, 144, 226, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(74, 144, 226, 0.4);
    }
    
    /* Expander styling with glass effect */
    .stExpander {
        background: rgba(26, 27, 35, 0.7);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Success/error styling */
    .stSuccess {
        background: rgba(26, 27, 35, 0.7);
        border-left: 4px solid #4caf50;
    }
    
    .stError {
        background: rgba(26, 27, 35, 0.7);
        border-left: 4px solid #f44336;
    }
    
    /* Animated gradient for status */
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    
    .status-pulse {
        animation: pulse 2s infinite;
    }
    
    /* Warning styling for rate limit */
    .rate-limit-warning {
        background: rgba(244, 67, 54, 0.1);
        border: 1px solid rgba(244, 67, 54, 0.3);
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# 2. SESSION STATE INITIALIZATION
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_engine" not in st.session_state:
    st.session_state.chat_engine = None

if "selected_model" not in st.session_state:
    st.session_state.selected_model = "llama-3.1-8b-instant"  # Default model


# ==========================================
# 3. RAG OPTIMIZATION: Pre-loaded Engine with Enhanced Caching and Rate Limit Handling
# ==========================================
@st.cache_resource(show_spinner=":rocket: Initializing ultra-fast RAG engine...")
def load_optimized_engine(model_name: str):
    """Optimized engine with pre-loaded models and advanced caching"""
    # Use faster model for initial loading to save tokens
    llm = initialise_llm(model_name=model_name)
    embed_model = get_embedding_model()
    
    # Create and cache the engine
    chat_engine = get_chat_engine(llm=llm, embed_model=embed_model)
    
    # Pre-warm the engine with a dummy query to initialize everything
    try:
        chat_engine.chat("Initialize RAG system")
    except:
        pass  # Ignore initialization errors
    
    return chat_engine


# ==========================================
# 4. SOPHISTICATED SIDEBAR WITH GLASS EFFECT
# ==========================================
with st.sidebar:
    # Branding with gradient
    st.markdown("""
    <div style='background: linear-gradient(135deg, #4a90e2 0%, #8b5cf6 100%); padding: 1.5rem; border-radius: 16px; margin-bottom: 1.5rem; text-align: center; box-shadow: 0 8px 24px rgba(139, 92, 246, 0.3);'>
        <h2 style='color: white; margin: 0; font-weight: 700;'>🧠 RAG Pro Engine</h2>
        <p style='color: rgba(255,255,255,0.8); margin: 0.5rem 0 0 0; font-size: 0.9rem;'>Powered by LlamaIndex & Groq</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()

    # --- Core Actions ---
    if st.button(":rotating_light: Clear Chat History", use_container_width=True, type="secondary"):
        st.session_state.messages = []
        if st.session_state.chat_engine:
            st.session_state.chat_engine.reset()
        st.rerun()

    st.divider()

    # --- Model Selection ---
    st.subheader(":gear: Model Configuration")
    model_options = [
        "llama-3.1-8b-instant",  # Fast and efficient (recommended)
        "llama-3.3-70b-versatile",  # Powerful but expensive
        "mixtral-8x7b-32768",  # Good balance of speed and quality
        "gemma2-9b-it"  # Alternative option
    ]
    
    selected_model = st.selectbox(
        "Select AI Model:",
        options=model_options,
        index=model_options.index(st.session_state.selected_model),
        help="Choose a model based on your needs. Smaller models are faster and use fewer tokens."
    )
    
    # Update session state when model changes
    if selected_model != st.session_state.selected_model:
        st.session_state.selected_model = selected_model
        st.session_state.chat_engine = None  # Force re-initialization
        st.rerun()

    st.divider()

    # --- Document Manager ---
    with st.expander("📁 Document Manager", expanded=True):
        all_files = [f for f in os.listdir(DATA_PATH) if f.endswith('.pdf')]
        
        if all_files:
            st.success(f"Indexed: **{len(all_files)}** documents")
            selected_files = st.multiselect(
                "Filter by Source:",
                options=all_files,
                default=all_files,
                help="Highlight which files to display sources for."
            )
        else:
            st.error("No PDFs found in `/data` directory!")
            selected_files = []

    # --- Advanced Configuration ---
    with st.expander("⚙️ Advanced Configuration", expanded=False):
        top_k_val = st.slider("Retrieval Depth (Top K)", 1, 10, SIMILARITY_TOP_K, help="How many document chunks to retrieve.")
        
        sys_prompt_choice = st.selectbox(
            "AI Persona",
            ["Default Mentor", "Strict Researcher", "Concise Bot", "Custom"],
            index=0
        )

        custom_prompt = ""
        if sys_prompt_choice == "Custom":
            custom_prompt = st.text_area("Enter Custom System Prompt:", height=100)

        # Apply prompt changes dynamically
        if st.session_state.chat_engine:
            if sys_prompt_choice == "Strict Researcher":
                st.session_state.chat_engine.system_prompt = "Answer using ONLY the provided sources. If unknown, say 'I don't know'. Cite sources explicitly."
            elif sys_prompt_choice == "Concise Bot":
                st.session_state.chat_engine.system_prompt = "Give very brief, one-sentence answers. No fluff."
            elif sys_prompt_choice == "Custom" and custom_prompt:
                st.session_state.chat_engine.system_prompt = custom_prompt
            else:
                st.session_state.chat_engine.system_prompt = LLM_SYSTEM_PROMPT

    # --- Danger Zone ---
    with st.expander("🛠️ Maintenance", expanded=False):
        st.warning("Clearing the vector store requires re-indexing all documents.")
        if st.button(":arrows_counterclockwise: Force Re-Index", use_container_width=True, type="primary"):
            with st.spinner("Wiping old index..."):
                if os.path.exists(VECTOR_STORE_PATH):
                    shutil.rmtree(VECTOR_STORE_PATH)
            st.cache_resource.clear()
            st.session_state.chat_engine = None
            st.success("Cache cleared! Please wait for re-indexing...")
            time.sleep(1)
            st.rerun()


# ==========================================
# 5. MAIN UI HEADER WITH ANIMATED GRADIENT
# ==========================================
if not st.session_state.messages:
    st.markdown("""
    <div style='text-align: center; padding: 2rem 0;'>
        <h1 style='font-size: 3rem; font-weight: 700; margin: 0; background: linear-gradient(135deg, #4a90e2 0%, #8b5cf6 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;'>What would you like to know?</h1>
        <p style='color: #888; margin: 1rem 0 0 0; font-size: 1.1rem;'>Your private RAG assistant is ready to query your documents.</p>
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# 6. CONVERSATION HISTORY RENDERING
# ==========================================
for message in st.session_state.messages:
    role = message["role"]
    with st.chat_message(role, avatar="🧠" if role == "assistant" else "👤"):
        st.markdown(message["content"])
        
        # Render sources if they exist
        if role == "assistant" and message.get("sources"):
            with st.expander(":chains: View Evidence", expanded=False):
                for s in message["sources"]:
                    st.markdown(s, unsafe_allow_html=True)


# ==========================================
# 7. EMPTY STATE: CLICKABLE STARTERS (PYTHON-SPECIFIC)
# ==========================================
if not st.session_state.messages:
    st.markdown("<br>", unsafe_allow_html=True) # Spacer
    col1, col2 = st.columns(2)
    
    # PYTHON-SPECIFIC STARTERS
    starters = [
        "Explain Python's object-oriented programming concepts",
        "What are Python's best practices for error handling?",
        "Show me how to implement a Python decorator",
        "Compare Python lists vs dictionaries"
    ]
    
    icons = ["📚", "⚠️", "🎨", "⚖️"]
    
    for i, starter in enumerate(starters):
        with col1 if i % 2 == 0 else col2:
            if st.button(f"{icons[i]}{starter}", key=f"starter_{i}", use_container_width=True):
                # Add the starter to messages and process it immediately
                st.session_state.messages.append({"role": "user", "content": starter.strip()})
                
                # Process the message immediately
                with st.chat_message("user", avatar="👤"):
                    st.markdown(starter.strip())
                
                # Generate response with optimized engine
                with st.chat_message("assistant", avatar="🧠"):
                    # Status indicator with animation
                    with st.status("Analyzing request...", expanded=True) as status:
                        status.markdown("<span class='status-pulse'>:mag_right: Searching knowledge base...</span>", unsafe_allow_html=True)
                        time.sleep(0.2)
                        
                        try:
                            response = st.session_state.chat_engine.chat(starter.strip())
                            answer = response.response
                            
                            status.markdown(":file_cabinet: Evaluating evidence...")
                            time.sleep(0.15)
                            status.markdown(":light_bulb: Synthesizing answer...")
                            time.sleep(0.15)
                            
                            status.update(label="✅ Response generated", state="complete", expanded=False)
                        
                        except Exception as e:
                            # Enhanced error handling for rate limits
                            if "rate_limit_exceeded" in str(e).lower():
                                error_msg = "⚠️ Rate limit reached! Please wait a few minutes or try a smaller model."
                                status.update(label=error_msg, state="error", expanded=True)
                                answer = error_msg
                                response = None
                            else:
                                status.update(label=f"❌ Error: {str(e)}", state="error", expanded=True)
                                answer = f"An error occurred: `{str(e)}`"
                                response = None

                    # Render the final text
                    st.markdown(answer)

                    # Process and Render Sources as "Cards"
                    source_info = []
                    if response and hasattr(response, 'source_nodes') and response.source_nodes:
                        for node in response.source_nodes:
                            name = node.metadata.get('file_name', 'Unknown Source')
                            if name in selected_files:
                                score = f"{node.score:.2f}" if hasattr(node, 'score') else "N/A"
                                snippet = node.get_content().replace("\n", " ")[:150] + "..."
                                
                                card_html = f"""
                                <div class="source-card">
                                    <strong>📄 {name}</strong> &nbsp; <span style="color: #888; font-size: 0.8em;">Relevance: {score}</span>
                                    <br><span style="font-size: 0.9em; color: #ccc;">{snippet}</span>
                                </div>
                                """
                                source_info.append(card_html)

                        if source_info:
                            with st.expander(":books: View Extracted Evidence", expanded=False):
                                for src in source_info:
                                    st.markdown(src, unsafe_allow_html=True)

                    # Feedback UI
                    c1, c2, c3 = st.columns([0.05, 0.05, 0.9])
                    with c1:
                        st.button("👍", key=f"up_{len(st.session_state.messages)}", help="Good response")
                    with c2:
                        st.button("👎", key="dn_{len(st.session_state.messages)}", help="Bad response")
                    with c3:
                        st.caption("Helps improve future responses.")

                # Save to Session State
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": source_info
                })
                st.rerun()


# ==========================================
# 8. GUARD CLAUSES
# ==========================================
# Lazy load the optimized engine on first prompt
if st.session_state.chat_engine is None:
    st.session_state.chat_engine = load_optimized_engine(st.session_state.selected_model)

if not selected_files:
    st.info(":pushpin: Please add PDF documents to the `/data` folder and reload.")
    st.stop()


# ==========================================
# 9. CHAT INPUT & GENERATION LOGIC
# ==========================================
if prompt := st.chat_input("Ask a question about your documents...", key="chat_input"):
    
    # 1. Display User Message
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)
    
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Generate Assistant Response
    with st.chat_message("assistant", avatar="🧠"):
        
        # Multi-step status indicator for premium feel
        with st.status("Analyzing request...", expanded=True) as status:
            status.markdown("<span class='status-pulse'>:mag_right: Searching knowledge base...</span>", unsafe_allow_html=True)
            time.sleep(0.2)
            
            try:
                response = st.session_state.chat_engine.chat(prompt)
                answer = response.response
                
                status.markdown(":file_cabinet: Evaluating evidence...")
                time.sleep(0.15)
                status.markdown(":light_bulb: Synthesizing answer...")
                time.sleep(0.15)
                
                status.update(label="✅ Response generated", state="complete", expanded=False)
            
            except Exception as e:
                # Enhanced error handling for rate limits
                if "rate_limit_exceeded" in str(e).lower():
                    error_msg = "⚠️ Rate limit reached! Please wait a few minutes or try a smaller model."
                    status.update(label=error_msg, state="error", expanded=True)
                    answer = error_msg
                    response = None
                else:
                    status.update(label=f"❌ Error: {str(e)}", state="error", expanded=True)
                    answer = f"An error occurred: `{str(e)}`"
                    response = None

        # Render the final text - clean and direct
        st.markdown(answer)

        # 3. Process and Render Sources as "Cards"
        source_info = []
        if response and hasattr(response, 'source_nodes') and response.source_nodes:
            for node in response.source_nodes:
                name = node.metadata.get('file_name', 'Unknown Source')
                # Only show sources for files the user selected in the sidebar
                if name in selected_files:
                    # Create a concise source card
                    score = f"{node.score:.2f}" if hasattr(node, 'score') else "N/A"
                    snippet = node.get_content().replace("\n", " ")[:150] + "..."
                    
                    card_html = f"""
                    <div class="source-card">
                        <strong>📄 {name}</strong> &nbsp; <span style="color: #888; font-size: 0.8em;">Relevance: {score}</span>
                        <br><span style="font-size: 0.9em; color: #ccc;">{snippet}</span>
                    </div>
                    """
                    source_info.append(card_html)

            if source_info:
                with st.expander(":books: View Extracted Evidence", expanded=False):
                    for src in source_info:
                        st.markdown(src, unsafe_allow_html=True)

        # 4. Feedback UI (Thumbs up/down layout)
        c1, c2, c3 = st.columns([0.05, 0.05, 0.9])
        with c1:
            st.button("👍", key=f"up_{len(st.session_state.messages)}", help="Good response")
        with c2:
            st.button("👎", key="dn_{len(st.session_state.messages)}", help="Bad response")
        with c3:
            st.caption("Helps improve future responses.")

    # 5. Save to Session State
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": source_info
    })

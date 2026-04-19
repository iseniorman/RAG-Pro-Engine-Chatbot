import streamlit as st
# Importing from your src folder
from src.engine import query_engine  # Replace 'query_engine' with your actual function/object name
from src.config import SYSTEM_PROMPTS # If you have prompts defined in config.py

# 1. Page Configuration
st.set_page_config(page_title="AI RAG Explorer", layout="wide")
st.title(":open_file_folder: Document Intelligence Assistant")

# 2. Sidebar: Configuration Panel
with st.sidebar:
    st.header(":gear: RAG Configuration")

    # Slider for similarity_top_k
    top_k = st.slider("Similarity Top K", min_value=1, max_value=10, value=3,
                      help="Number of document chunks to retrieve")

    # Dropdown for System Prompt (fetching from your config if possible)
    prompt_option = st.selectbox("System Role",
                                 ["Default Assistant", "Technical Researcher", "Simple Summarizer"])

    st.divider()

    # Add Chat Controls: New Chat
    if st.button(":arrows_counterclockwise: Start New Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# 3. Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# 4. Display Conversation History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Show Sources if available
        if "sources" in message and message["sources"]:
            with st.expander(":mag: Verified Sources"):
                for source in message["sources"]:
                    st.info(source)

# 5. User Input Logic
if user_query := st.chat_input("Ask a question about your data..."):
    # Display user query
    st.chat_message("user").markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.chat_message("assistant"):
        # Placeholder for spinner while AI "thinks"
        with st.spinner("Searching documents..."):
            try:
                # --- INTEGRATING YOUR ENGINE ---
                # Pass the top_k from the slider into your engine here
                # Example: response = query_engine(user_query, top_k=top_k)

                # For now, these are placeholders based on your requirements:
                response_text = "This is a placeholder response. Connect your src/engine.py logic here."
                source_chunks = ["Source 1: [Text from your data folder]", "Source 2: [Text from your data folder]"]
                # -------------------------------

                st.markdown(response_text)

                # 6. User Feedback Buttons
                col1, col2 = st.columns([0.05, 0.95])
                with col1:
                    st.button(":+1:", key=f"up_{len(st.session_state.messages)}")
                with col2:
                    st.button(":-1:", key=f"down_{len(st.session_state.messages)}")

                # 7. Show Sources
                with st.expander(":mag: Verified Sources"):
                    for chunk in source_chunks:
                        st.info(chunk)

                # Save assistant response to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_text,
                    "sources": source_chunks
                })

            except Exception as e:
                st.error(f"Error connecting to RAG engine: {e}")
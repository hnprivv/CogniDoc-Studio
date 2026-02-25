import streamlit as st
import os
import time
import random
from converter import PDFGenerator
from utils import AppUtils
from streamlit_mic_recorder import speech_to_text

# Initializing all session state keys at the start
if "content_area" not in st.session_state:
    st.session_state.content_area = ""

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if 'pdf_generated' not in st.session_state:
    st.session_state.pdf_generated = False

# 1. Setup & Styling
st.set_page_config(page_title="CogniDoc Studio", page_icon="📄", layout="centered")
AppUtils.load_css(os.path.join("assets", "styles.css"))

st.title("CogniDoc: AI-Powered PDF Studio")

# 2. Sidebar
with st.sidebar:
    st.header("📂 Import / Export")
    
    # Link to the utility function via on_change
    uploaded_file = st.file_uploader(
        "Upload File", 
        type=["txt", "docx", "pdf"],
        key="file_import_widget",
        on_change=lambda: AppUtils.handle_file_upload(progress_placeholder)
    )
    
    filename = st.text_input("Export Name", "document.pdf")
    
    st.divider()

    st.header("🎤 Speech -> Text")
    text_from_voice = speech_to_text(
        language='en', 
        start_prompt="Start Recording", 
        stop_prompt="Stop & Transcribe", 
        just_once=True, 
        key='STT'
    )

    AppUtils.handle_voice_input(text_from_voice)

    st.divider()

    st.header("🪄 AI Powered Tools")

    #  Auto-Format
    if st.session_state.content_area.strip():
        if st.button("✨ Auto-Format", use_container_width=True):
            with st.spinner("Processing..."):
                formatted = AppUtils.ai_auto_format(st.session_state.content_area)
                st.session_state.content_area = formatted
                # Reset Vector DB since the content changed
                st.session_state.vector_db = None 
                st.rerun()

        # Feature 2: CogniDoc (Inquire about your document)
        if st.button("🗨️ Chat", use_container_width=True):
            if st.session_state.content_area.strip():
                # Initialize Vector DB if missing
                if not st.session_state.get("vector_db"):
                    with st.spinner("Indexing..."):
                        st.session_state.vector_db = AppUtils.create_vector_store(st.session_state.content_area)
                
                AppUtils.show_chat_modal()
    else:
        st.info("Upload or enter text to enable AI tools.")

    st.divider()

    st.header("🛠️ Processing")
    trim = st.checkbox("Trim Space")
    case = st.selectbox("Case", ["No Change", "UPPERCASE", "lowercase", "Title Case"])

    st.divider()
    
    st.header("🎨 Styling")
    f_family = st.selectbox("Font", ["Sans-Serif (Arial)", "Serif (Times)", "Monospace (Courier)"])
    f_size = st.slider("Size", 8, 32, 12)

# 3. Main Input Area
user_input = st.text_area(
    "Content:", 
    height=300, 
    key="content_area"
)

# If the user clears the text manually, reset the generation state
if not user_input:
    st.session_state.pdf_generated = False

# --- Enhanced Insights Section ---
if user_input:
    stats = AppUtils.get_stats(user_input)
    
    with st.expander("📊 View Content Insights", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Reading Grade", f"Lvl {stats['grade']}")
            st.caption("Flesch-Kincaid Scale")
            
        with col2:
            st.metric("Reading Time", f"{stats['reading_time']} min")
            st.caption("Based on 200 WPM")
            
        with col3:
            st.metric("Word Count", stats['words'])
            st.caption(f"{stats['chars']} Characters")

        with col4:
            sent = stats['sentiment']
            st.metric("Sentiment", sent['label'], delta=sent['score'])
            # st.caption(f"Subjectivity: {sent['subjectivity']}")  
        
        st.divider()

        st.write("**Top Themes:**")
        if stats['top_terms']:
            cols = st.columns(len(stats['top_terms']))
            for idx, (word, count) in enumerate(stats['top_terms']):
                cols[idx].info(f"{word.capitalize()} ({count}x)")
        else:
            st.write("Enter more text to extract themes.")
else:
    st.caption("Enter text above to unlock document intelligence.")

# 4. Conversion / Action Buttons
col_submit, col_clear = st.columns(2)

with col_submit:
    submit_disabled = not user_input or st.session_state.pdf_generated
    if st.button("Submit", type="primary", use_container_width=True, disabled=submit_disabled):
        st.session_state.pdf_generated = True
        st.rerun()

with col_clear:
    clear_disabled = not user_input
    st.button(
        "Clear All", 
        use_container_width=True, 
        disabled=clear_disabled, 
        on_click=AppUtils.clear_text
    )

progress_placeholder = st.empty()

# 5. Processing Logic
if st.session_state.pdf_generated and user_input:
    with st.status("Finalizing document...", expanded=True) as status:
        st.write("Processing text intelligence...")
        clean_text = AppUtils.process_text(user_input, trim_space=trim, case_style=case)
        
        delay = random.uniform(3, 8)
        time.sleep(delay) 
        
        st.write("Building PDF structure...")
        success, msg = PDFGenerator.convert_text_to_pdf(
            clean_text,
            filename,
            f_family,
            f_size,
        )
        
        if success:
            status.update(label="Conversion Complete", state="complete", expanded=False)
            st.success(f"✨ {msg}! Your file '{filename}' is ready.")
            
            with open(filename, "rb") as f:
                pdf_data = f.read()
            
            st.download_button(
                label="📥 Download PDF Now",
                data=pdf_data,
                file_name=filename,
                mime="application/pdf",
                use_container_width=True
            )
        else:
            status.update(label="Conversion Failed", state="error")
            st.error(msg)
            st.session_state.pdf_generated = False
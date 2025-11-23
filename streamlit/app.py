import streamlit as st
import requests
from pdf2image import convert_from_bytes
from PIL import Image
import io
import base64
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import markdown2
import pynvml
import gc
import json
import os
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Custom CSS for polished UI
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; padding: 30px; }
    .stButton>button { 
        background-color: #28a745; 
        color: white; 
        border-radius: 8px; 
        padding: 8px 16px; 
        font-size: 16px; 
        margin-top: 10px; 
    }
    .stTextArea>label { 
        font-weight: bold; 
        color: #1a3c34; 
        font-size: 16px; 
    }
    .stImage>img { 
        border: 1px solid #e0e0e0; 
        border-radius: 8px; 
        margin-bottom: 15px; 
    }
    .sidebar .sidebar-content { 
        background-color: #ffffff; 
        padding: 20px; 
        border-right: 1px solid #e0e0e0; 
    }
    h1, h2, h3 { 
        color: #1a3c34; 
        font-family: 'Helvetica Neue', Arial, sans-serif; 
        margin-bottom: 20px; 
    }
    .stProgress .st-bo { 
        background-color: #28a745; 
    }
    .debug-expander { 
        background-color: #e6f3fa; 
        border-radius: 8px; 
        padding: 15px; 
        margin-bottom: 15px; 
    }
    .markdown-preview { 
        background-color: #ffffff; 
        padding: 20px; 
        border: 1px solid #e0e0e0; 
        border-radius: 8px; 
        min-height: 400px; 
        font-family: 'Helvetica Neue', Arial, sans-serif; 
        font-size: 16px; 
    }
    .stContainer { 
        margin-bottom: 20px; 
        padding: 15px; 
        border-radius: 8px; 
        background-color: #ffffff; 
    }
    </style>
""", unsafe_allow_html=True)

# vLLM API endpoint
VLLM_URL = "http://qwen-vlm:8000/v1/chat/completions"

# Session state persistence file
SESSION_FILE = "/app/data/session_state.json"

# Initialize session state
if "file_results" not in st.session_state:
    st.session_state.file_results = {}
if "file_times" not in st.session_state:
    st.session_state.file_times = {}
if "debug_info" not in st.session_state:
    st.session_state.debug_info = {}
if "processed" not in st.session_state:
    st.session_state.processed = False
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

# Ensure /app/data is writable
os.makedirs("/app/data", exist_ok=True)

# Load session state from file
def load_session_state():
    if os.path.exists(SESSION_FILE):
        try:
            with open(SESSION_FILE, 'r') as f:
                data = json.load(f)
                st.session_state.file_results = data.get('file_results', {})
                st.session_state.file_times = {k: float(v) for k, v in data.get('file_times', {}).items()}
                st.session_state.debug_info = data.get('debug_info', {})
                st.session_state.processed = data.get('processed', False)
                st.session_state.uploaded_files = [
                    {"name": f["name"], "data": base64.b64decode(f["data"])} for f in data.get('uploaded_files', [])
                ]
        except Exception as e:
            st.error(f"Failed to load session state: {str(e)}")

# Save session state to file
def save_session_state():
    try:
        with open(SESSION_FILE, 'w') as f:
            json.dump({
                'file_results': st.session_state.file_results,
                'file_times': st.session_state.file_times,
                'debug_info': st.session_state.debug_info,
                'processed': st.session_state.processed,
                'uploaded_files': [
                    {"name": f["name"], "data": base64.b64encode(f["data"]).decode("utf-8")} for f in st.session_state.uploaded_files
                ]
            }, f)
    except Exception as e:
        st.error(f"Failed to save session state: {str(e)}")

# Load session state on app start
load_session_state()

# Dynamic workers based on GPU memory
try:
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    total_free_mem = 0
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        total_free_mem += mem_info.free / 1024**3  # Sum free VRAM
    MAX_WORKERS = min(4, max(1, int(total_free_mem // 3)))  # ~3GB per request
    pynvml.nvmlShutdown()
except Exception:
    MAX_WORKERS = 2  # Fallback if pynvml fails

# Wait for vLLM server to be ready
def wait_for_vllm():
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    session.mount('http://', HTTPAdapter(max_retries=retries))
    start_time = time.time()
    while time.time() - start_time < 90:
        try:
            response = session.get("http://qwen-vlm:8000/health", timeout=5)
            if response.status_code == 200:
                return True
        except Exception:
            time.sleep(5)
    return False

if not wait_for_vllm():
    st.error("vLLM server not ready after 90 seconds. Please check logs and try again.")
    st.stop()

st.title("📄 Qwen3-VL OCR on PDFs")

# Sidebar for file selection and status
with st.sidebar:
    st.header("Uploaded Files")
    uploaded_files = st.file_uploader("Upload PDFs (max 20 files, 200MB each)", type=["pdf"], accept_multiple_files=True, key="file_uploader")
    if uploaded_files:
        st.write(f"Files uploaded: {len(uploaded_files)}")
        if len(uploaded_files) > 20:
            st.warning("Maximum 20 files allowed. Please upload fewer files.")
    status_placeholder = st.empty()
    
    # File deletion
    if st.session_state.uploaded_files:
        st.subheader("Manage Files")
        for i, file in enumerate(st.session_state.uploaded_files):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(file["name"])
            with col2:
                if st.button("Delete", key=f"delete_{i}", type="secondary"):
                    file_name = file["name"]
                    st.session_state.uploaded_files.pop(i)
                    if file_name in st.session_state.file_results:
                        del st.session_state.file_results[file_name]
                    if file_name in st.session_state.file_times:
                        del st.session_state.file_times[file_name]
                    if file_name in st.session_state.debug_info:
                        del st.session_state.debug_info[file_name]
                    st.session_state.processed = False
                    save_session_state()
                    st.rerun()

def process_page(file_name, page, page_idx):
    """Process a single page, returning OCR result and timing."""
    page_start_time = time.time()
    try:
        # Simple preprocessing
        if page.mode != 'RGB':
            page = page.convert('RGB')
        
        # Resize to reasonable dimensions
        max_dimension = 800
        ratio = min(max_dimension / page.width, max_dimension / page.height)
        new_size = (int(page.width * ratio), int(page.height * ratio))
        if ratio < 1:
            page = page.resize(new_size, Image.LANCZOS)
        
        buf = io.BytesIO()
        page.save(buf, format="JPEG", quality=85)
        img_bytes = buf.getvalue()
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
        
        # Clean up
        buf.close()
        page.close()
        del page, buf
        gc.collect()
        
        # Direct prompt for OCR
        payload = {
            "model": "Qwen/Qwen3-VL-2B-Instruct-FP8",  # UPDATED MODEL
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": "Extract all text from this image exactly as it appears. Preserve line breaks, formatting, and punctuation. Output only the text content."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_b64}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 2048,
            "temperature": 0.0,
            "top_p": 1.0,
            "repetition_penalty": 1.2
        }
        
        session = requests.Session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
        session.mount('http://', HTTPAdapter(max_retries=retries))
        
        response = session.post(VLLM_URL, json=payload, headers={"Content-Type": "application/json"}, timeout=120)
        if response.status_code == 200:
            text = response.json()["choices"][0]["message"]["content"].strip()
        else:
            text = f"Error: {response.status_code} - {response.text}"
            
    except Exception as e:
        text = f"Exception: {str(e)}"
    
    page_time = time.time() - page_start_time
    
    # Clean up
    if 'img_b64' in locals():
        del img_b64
    if 'payload' in locals():
        del payload
    gc.collect()
    
    return file_name, page_idx, text, page_time

def convert_pdf_to_pages(uploaded_file):
    """Convert PDF to pages."""
    file_name = uploaded_file["name"]
    try:
        pages = convert_from_bytes(uploaded_file["data"], dpi=300)  # High DPI for quality
        return file_name, pages
    except Exception as e:
        return file_name, str(e)

# Process files only if new uploads or not yet processed
uploaded_file_names = [f.name for f in uploaded_files] if uploaded_files else []
current_file_names = [f["name"] for f in st.session_state.uploaded_files]

if uploaded_files and (not st.session_state.processed or uploaded_file_names != current_file_names):
    with status_placeholder.container():
        st.info("Processing PDFs in parallel with Qwen3-VL...")
        progress_bar = st.progress(0)
        progress_text = st.empty()
    
    # Update uploaded files in session state
    st.session_state.uploaded_files = [{"name": f.name, "data": f.read()} for f in uploaded_files]
    for f in uploaded_files:
        f.seek(0)  # Reset file pointers
    
    # Step 1: Convert PDFs to pages in parallel
    pdf_pages = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(convert_pdf_to_pages, uploaded_file): uploaded_file["name"] for uploaded_file in st.session_state.uploaded_files}
        for future in as_completed(futures):
            file_name = futures[future]
            try:
                fname, result = future.result()
                if isinstance(result, str):
                    st.error(f"Failed to convert {file_name}: {result}")
                else:
                    pdf_pages[fname] = result
            except Exception as e:
                st.error(f"Failed to convert {file_name}: {str(e)}")
            gc.collect()
    
    # Step 2: Process all pages in parallel
    all_pages = []
    total_pages = 0
    st.session_state.file_results = {}
    st.session_state.file_times = {}
    st.session_state.debug_info = {}
    
    for file_name, pages in pdf_pages.items():
        file_start_time = time.time()
        all_pages.extend([(file_name, page, i+1) for i, page in enumerate(pages)])
        total_pages += len(pages)
        st.session_state.file_times[file_name] = file_start_time
        st.session_state.file_results[file_name] = []
        st.session_state.debug_info[file_name] = {
            "pdf_size": len([f for f in st.session_state.uploaded_files if f["name"] == file_name][0]["data"]),
            "page_count": len(pages)
        }
    
    if all_pages:
        with status_placeholder.container():
            progress_text.text(f"Processing {total_pages} pages with {MAX_WORKERS} concurrent requests...")
        
        processed_pages = 0
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_page = {executor.submit(process_page, file_name, page, page_idx): (file_name, page_idx) for file_name, page, page_idx in all_pages}
            for future in as_completed(future_to_page):
                file_name, page_idx = future_to_page[future]
                try:
                    file_name, page_idx, text, page_time = future.result()
                    processed_pages += 1
                    progress_bar.progress(processed_pages / total_pages)
                    progress_text.text(f"Processed {processed_pages}/{total_pages} pages")
                    st.session_state.file_results[file_name].append({
                        "page": page_idx,
                        "text": text,
                        "page_time": page_time
                    })
                except Exception as e:
                    st.error(f"Failed to process {file_name} - Page {page_idx}: {str(e)}")
                gc.collect()
        
        # Sort results by page number
        for file_name in st.session_state.file_results:
            st.session_state.file_results[file_name].sort(key=lambda x: x["page"])
        
        st.session_state.processed = True
        save_session_state()
    
    progress_text.text("Processing complete!")
    st.rerun()

# Sidebar: File selection and summary
if st.session_state.file_results:
    total_pages = sum(st.session_state.debug_info[file]["page_count"] for file in st.session_state.debug_info)
    st.sidebar.write(f"Total files: {len(st.session_state.file_results)}")
    st.sidebar.write(f"Total pages: {total_pages}")
    selected_file = st.sidebar.selectbox("Select a file to preview", options=list(st.session_state.file_results.keys()), index=0)

# Debug info
with st.expander("Debug Information"):
    for file_name, info in st.session_state.debug_info.items():
        st.write(f"PDF: {file_name}")
        st.write(f"Size: {info['pdf_size']} bytes")
        st.write(f"Pages: {info['page_count']}")
    
    st.write("Session state keys:", list(st.session_state.keys()))
    st.write("Processed status:", st.session_state.processed)

# Main content: two columns
if st.session_state.file_results:
    col1, col2 = st.columns([2, 3])
    
    with col1.container():
        st.header("Page Details")
        if selected_file:
            for result in st.session_state.file_results[selected_file]:
                with st.container():
                    st.subheader(f"Page {result['page']}")
                    st.text_area(f"OCR Result - Page {result['page']}", result["text"], height=150, key=f"text_{selected_file}_{result['page']}")
                    with st.expander("Page Debug Info"):
                        st.write(f"Processing time: {result['page_time']:.2f} seconds")

    with col2.container():
        st.header("Markdown Preview")
        if selected_file:
            markdown_content = f"# OCR Results for {selected_file}\n\n"
            for result in st.session_state.file_results[selected_file]:
                markdown_content += f"## Page {result['page']}\n\n"
                markdown_content += f"**Processing Time**: {result['page_time']:.2f} seconds\n\n"
                markdown_content += f"```text\n{result['text']}\n```\n\n"
            html_content = markdown2.markdown(markdown_content, extras=["fenced-code-blocks", "tables"])
            st.markdown(f'<div class="markdown-preview">{html_content}</div>', unsafe_allow_html=True)
            
            st.download_button(
                label=f"Download Markdown for {selected_file}",
                data=markdown_content,
                file_name=f"ocr_{selected_file}.md",
                mime="text/markdown"
            )

# Processing summary and combined download
with st.expander("Processing Summary"):
    for file_name, start_time in st.session_state.file_times.items():
        file_time = time.time() - start_time
        st.write(f"Total processing time for {file_name}: {file_time:.2f} seconds")
    
    all_results = []
    for file_name in st.session_state.file_results:
        for result in st.session_state.file_results[file_name]:
            all_results.append(f"File: {file_name} - Page {result['page']}\n{result['text']}")
    if all_results:
        full_text = "\n\n".join(all_results)
        st.download_button(
            label="Download Combined OCR Text",
            data=full_text,
            file_name="combined_ocr_output.txt",
            mime="text/plain"
        )

# Clear session state button
with st.expander("Advanced Options"):
    if st.button("Clear Session State and Start Over"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        if os.path.exists(SESSION_FILE):
            os.remove(SESSION_FILE)
        st.rerun()
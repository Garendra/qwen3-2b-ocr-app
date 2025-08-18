import streamlit as st
import requests
from pdf2image import convert_from_bytes
from PIL import Image
import io
import base64

# vLLM API endpoint
VLLM_URL = "http://qwen-vlm:8000/v1/chat/completions"

st.title("📄 Qwen2.5-VL OCR on PDFs")
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    st.info("Processing PDF pages...")
    pages = convert_from_bytes(uploaded_file.read(), dpi=200)
    results = []
    for i, page in enumerate(pages):
        st.image(page, caption=f"Page {i+1}", use_container_width=True)
        # Convert PIL image to PNG bytes
        buf = io.BytesIO()
        page.save(buf, format="PNG")
        img_bytes = buf.getvalue()
        # Encode image as base64
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
        # Prepare payload for Qwen2.5-VL
        payload = {
            "model": "Qwen/Qwen2.5-VL-3B-Instruct-AWQ",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Please extract all text from this image."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_b64}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 4096,
            "temperature": 0.0
        }
        try:
            response = requests.post(VLLM_URL, json=payload, headers={"Content-Type": "application/json"})
            if response.status_code == 200:
                text = response.json()["choices"][0]["message"]["content"]
            else:
                text = f"Error: {response.status_code} - {response.text}"
        except Exception as e:
            text = f"Exception: {e}"
        st.text_area(f"OCR Result - Page {i+1}", text, height=200)
        results.append(text)
    # Download full OCR text
    full_text = "\n\n".join(results)
    st.download_button(
        "Download Full OCR Text",
        full_text,
        file_name="ocr_output.txt",
        mime="text/plain"
    )

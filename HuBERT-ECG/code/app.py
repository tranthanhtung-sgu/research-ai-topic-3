#!/usr/bin/env python3
"""
Streamlit app for ECG diagnosis using HuBERT-ECG model
"""
import os, io, base64, tempfile, pathlib
import numpy as np
import streamlit as st
import re

# Fix for PyTorch module watching issue
import sys
from streamlit.web.server import Server
Server._watch_for_local_sources_changes = lambda *args, **kwargs: None

from PIL import Image
from fpdf import FPDF
import matplotlib.pyplot as plt

# Import backend modules
from backend import eda, report
from validate_six_conditions import predict

# Default model path
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                 "outputs/six_conditions_model_anti_overfitting_best/best_model_accuracy.pt")

# ────────────────────────────────── UI SETUP ─────────────────────────────────
st.set_page_config(page_title="🫀 ECG Diagnosis Demo", layout="centered")
st.markdown("""
<style>
    h1{color:#8B0000;}
    .stButton>button{border-radius:8px;font-weight:600;}
    [data-testid="stSidebar"]{background:#F0F8FF;}
    pre {
        background-color: #f5f5f5;
        padding: 10px;
        border-radius: 5px;
        white-space: pre-wrap;
        font-family: monospace;
    }
</style>""", unsafe_allow_html=True)
st.title("🫀 ECG Condition Detection — HuBERT-ECG")

OPENAI_API_KEY = (
    st.sidebar.text_input("🔑 Paste your OpenAI key", type="password")
    or os.getenv("OPENAI_API_KEY", "")
)

# Display a warning if no API key is provided
if not OPENAI_API_KEY:
    st.sidebar.warning("⚠️ No OpenAI API key provided. The app will run without generating detailed reports.")

# Add model selection in sidebar
model_path = st.sidebar.text_input("Model Path", value=DEFAULT_MODEL_PATH)

st.sidebar.markdown("""
**Flow**

1. Upload **validation.npy** file  
2. Press **Start**  
3. Download PDF report
""")

# ───────────────────────────── helper: PDF builder ───────────────────────────
def build_pdf(title, cond, imgs, rpt, detailed_results=None):
    pdf = FPDF(); pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page(); pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, title, ln=1, align="C"); pdf.ln(4)
    pdf.set_font("Helvetica", "", 12)
    pdf.multi_cell(0, 8, f"Predicted Condition: {cond}"); pdf.ln(2)
    
    # Add detailed prediction results if available
    if detailed_results:
        # Replace Unicode characters with ASCII alternatives
        detailed_results = detailed_results.replace("✓", "YES")
        detailed_results = detailed_results.replace("✗", "NO")
        
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "Detailed Prediction Results:", ln=1); pdf.ln(2)
        pdf.set_font("Courier", "", 10)  # Use monospaced font for table
        
        # Split the detailed results into lines
        result_lines = detailed_results.split('\n')
        for line in result_lines:
            pdf.cell(0, 5, line, ln=1)
        pdf.ln(5)
    
    # Add plain language report - sanitize for PDF compatibility
    pdf.set_font("Helvetica", "", 12)
    
    # Replace problematic Unicode characters with ASCII equivalents
    if rpt:
        # Common replacements for PDF compatibility
        rpt = rpt.replace('\u2013', '-')       # en dash
        rpt = rpt.replace('\u2014', '-')       # em dash
        rpt = rpt.replace('\u2018', "'")       # left single quote
        rpt = rpt.replace('\u2019', "'")       # right single quote
        rpt = rpt.replace('\u201c', '"')       # left double quote
        rpt = rpt.replace('\u201d', '"')       # right double quote
        rpt = rpt.replace('\u2022', '*')       # bullet
        rpt = rpt.replace('\u2026', '...')     # ellipsis
        rpt = rpt.replace('\u00a0', ' ')       # non-breaking space
        rpt = rpt.replace('\u00b0', ' degrees') # degree symbol
        rpt = rpt.replace('\u00b1', '+/-')     # plus-minus sign
        rpt = rpt.replace('\u03bc', 'u')       # micro symbol
        rpt = rpt.replace('\u2264', '<=')      # less than or equal
        rpt = rpt.replace('\u2265', '>=')      # greater than or equal
        
        # Remove any remaining non-Latin-1 characters
        rpt = ''.join(c if ord(c) < 256 else '?' for c in rpt)
    
    pdf.multi_cell(0, 7, rpt or "—")
    
    # Add ECG visualizations
    for im in imgs:
        pdf.add_page()
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        
        # Check if im is a file path (string) or a PIL Image
        if isinstance(im, str):
            # If it's a path, just copy the file
            with open(im, 'rb') as src_file:
                with open(tmp.name, 'wb') as dest_file:
                    dest_file.write(src_file.read())
        else:
            # Otherwise, assume it's a PIL Image and save it
            im.save(tmp.name)
            
        tmp.close()
        pdf.image(tmp.name, w=180)
        os.unlink(tmp.name)
        
    out = pathlib.Path(tempfile.gettempdir()) / "ecg_report.pdf"
    
    # Use try-except to catch encoding errors
    try:
        pdf.output(out)
    except UnicodeEncodeError as e:
        print(f"Warning: Unicode encoding issue detected: {e}")
        # Fallback to a more restricted ASCII version
        try:
            # Create a new PDF with even more restricted content
            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.add_page()
            pdf.set_font("Helvetica", "B", 16)
            pdf.cell(0, 10, "ECG Report", ln=1, align="C")
            pdf.set_font("Helvetica", "", 12)
            pdf.cell(0, 10, f"Condition: {cond}", ln=1)
            pdf.multi_cell(0, 10, "The detailed report contains characters that cannot be encoded in the PDF. Please refer to the on-screen report.")
            
            # Still include the images
            for im in imgs:
                pdf.add_page()
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                if isinstance(im, str):
                    with open(im, 'rb') as src_file:
                        with open(tmp.name, 'wb') as dest_file:
                            dest_file.write(src_file.read())
                else:
                    im.save(tmp.name)
                tmp.close()
                pdf.image(tmp.name, w=180)
                os.unlink(tmp.name)
                
            pdf.output(out)
        except Exception as e2:
            print(f"Failed to create fallback PDF: {e2}")
            return None
            
    return out

# ───────────────────────────────── main UI ───────────────────────────────────
uploaded = st.file_uploader("📂 Upload validation.npy", type=["npy"])

def process_ecg(ecg_data, sample_name="Sample", model_path=DEFAULT_MODEL_PATH):
    """Process a single ECG trace"""
    # Print diagnostic information about the data
    print(f"Data shape: {ecg_data.shape}")
    print(f"Data type: {ecg_data.dtype}")
    print(f"Data min/max: {ecg_data.min():.4f}/{ecg_data.max():.4f}")
    print(f"Data mean/std: {ecg_data.mean():.4f}/{ecg_data.std():.4f}")
    
    # Check if model path exists
    if not os.path.exists(model_path):
        print(f"Warning: Model file {model_path} does not exist. Trying default path.")
        model_path = DEFAULT_MODEL_PATH
        if not os.path.exists(model_path):
            print(f"Error: Default model file {model_path} also does not exist.")
    
    # Pass model_path to predict function
    cond, fs, detailed_results = predict(ecg_data, sample_name, model_path)
    
    # Make a deep copy of the data to avoid modifying the original
    ecg_data_copy = ecg_data.copy()
    
    # Print data shape before passing to make_figs
    print(f"Data shape before passing to make_figs: {ecg_data_copy.shape}")
    
    figs, report_text = eda.make_figs(ecg_data_copy, 
                                      case_condition_short_label=cond,
                                      ref_npy_path_relative="./backend/validation01/validation01.npy")
    print(figs)
    
    # Check if API key is provided
    if not OPENAI_API_KEY:
        # Use default report if no API key is available
        rpt = "# API Key Required\n\nPlease provide an OpenAI API key in the sidebar to generate a detailed report.\n\n## ECG Summary\n" + report_text
    else:
        # Generate report using OpenAI
        rpt = report.write(cond, figs, report_text, OPENAI_API_KEY)
        
    return cond, figs, rpt, detailed_results

if st.button("🚀 Start", type="primary"):
    if uploaded is None:
        st.warning("Upload a .npy file first")
        st.stop()
        
    try:
        # Load the data
        ecg_data = np.load(uploaded)
        
        # Print diagnostic information about the loaded file
        print(f"Loaded file: {uploaded.name}")
        print(f"Original data shape: {ecg_data.shape}")
        print(f"Original data type: {ecg_data.dtype}")
        
        # Format data correctly based on dimensions
        if ecg_data.ndim == 1:
            # Single lead data, reshape to (1, samples)
            ecg_data = ecg_data.reshape(1, -1)
        elif ecg_data.ndim == 2:
            # Check if we need to transpose
            if ecg_data.shape[0] > ecg_data.shape[1]:
                # Likely (samples, leads), transpose to (leads, samples)
                ecg_data = ecg_data.T
        elif ecg_data.ndim == 3:
            # Multiple ECGs, take only the first one
            st.info("File contains multiple ECGs. Only processing the first one.")
            ecg_data = ecg_data[0]
            
            # Check shape again after taking first sample
            if ecg_data.ndim == 1:
                ecg_data = ecg_data.reshape(1, -1)
            elif ecg_data.shape[0] > ecg_data.shape[1]:
                ecg_data = ecg_data.T
        
        # Get filename without extension for sample name
        sample_name = os.path.splitext(os.path.basename(uploaded.name))[0]
        
        # Process the ECG
        with st.spinner("Processing ECG..."):
            cond, imgs, rpt, detailed_results = process_ecg(ecg_data, sample_name, model_path)
        
        # Display results
        st.markdown(f"### 🩺 Predicted Condition\n**{cond}**")
        
        # Display detailed prediction results
        st.markdown("### 📊 Detailed Prediction Results")
        st.code(detailed_results, language=None)
        
        # Display ECG visualizations
        st.markdown("### 📈 ECG Visualizations")
        for im in imgs:
            # Check if im is a file path (string) or a PIL Image
            if isinstance(im, str):
                # If it's a path, load it as a PIL Image first
                try:
                    pil_image = Image.open(im)
                    st.image(pil_image, use_column_width=True)
                except Exception as img_error:
                    st.error(f"Error displaying image {im}: {str(img_error)}")
            else:
                # Otherwise, assume it's a PIL Image
                st.image(im, use_column_width=True)
        
        # Display report
        st.markdown("### 📝 Plain-Language Report")
        st.markdown(rpt, unsafe_allow_html=True)
        
        # Create PDF report
        try:
            pdf_path = build_pdf("ECG Diagnosis Report", cond, imgs, rpt, detailed_results)
            with open(pdf_path, "rb") as f:
                st.download_button("📄 Download PDF", f,
                    file_name=f"ecg_report_{sample_name}.pdf",
                    mime="application/pdf")
        except UnicodeEncodeError as e:
            st.warning(f"Could not generate PDF due to encoding issues: {str(e)}")
            st.info("PDF generation failed, but you can still see all results on screen.")
                
    except Exception as e:
        st.error(f"Error processing ECG data: {str(e)}")
        st.exception(e)

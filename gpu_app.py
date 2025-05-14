
# This code is for running LLM model on local GPU to the server

from flask import Flask, request, jsonify, send_file, session
from werkzeug.datastructures import FileStorage
from utils import clean_csv_response
from flask_cors import CORS
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from dotenv import load_dotenv
import os
from supabase import create_client, Client
import logging
from io import BytesIO
import uuid
import pickle
import base64
import binascii
import pickle
import numpy as np
from mistralai import Mistral

# Load environment variables
load_dotenv()
load_dotenv()

# Tracks LLM models and who uses them 
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# Check CUDA availability
print("CUDA Version:", torch.version.cuda)  
print("CUDA Available:", torch.cuda.is_available())

# Automatically set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Use float16 for GPU, float32 for CPU
torch_dtype = torch.float16 if device == "cuda" else torch.float32

# **Your model name**
model_name = "mistralai/Mistral-7B-v0.3"

# 1) Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_name, 
    use_auth_token=HUGGINGFACE_TOKEN
)

# 2) Load model (quantized on GPU, full on CPU)
if device == "cuda":
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map="auto",
        use_auth_token=HUGGINGFACE_TOKEN,
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        use_auth_token=HUGGINGFACE_TOKEN,
    )
    model.to(device)

print(f"Loaded {model_name} on {device} with dtype {torch_dtype}")

# Function extract text from PDF
def extract_pdf_text(pdf_bytes: bytes) -> str:
    reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
    return "\n".join(page.extract_text() or "" for page in reader.pages)

def clean_csv_response(resp: str) -> str:
    # strip BOM, drop any “Unnamed” columns, remove stray quotes, etc.
    lines = [line.strip().strip('"') for line in resp.splitlines() if line.strip()]
    # optionally filter out header rows mentioning “Fact,Value”
    return "\n".join(lines)

@app.route("/restructure2", methods=["POST"])
def restructure2():
    try:
        # 1) Receive PDF
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400
        pdf_bytes = request.files["file"].read()

        # 2) Extract plain text
        document_text = extract_pdf_text(pdf_bytes)

        # 3) Build the prompt
        prompt = (
            "Extract all the key facts and their numeric values from the document below.\n"
            "Output only CSV with exactly two columns: Fact,Value. No headers, no commentary.\n"
            "If you cannot parse it, reply exactly “failed to structure”.\n\n"
            "-----DOCUMENT START-----\n"
            f"{document_text}\n"
            "-----DOCUMENT END-----"
        )

        # 4) Tokenize & move to device (GPU or CPU depending on `device`)
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # --- Optional CPU-only path (uncomment to use) ---
        # cpu_inputs = tokenizer(prompt, return_tensors="pt")
        # cpu_inputs = {k: v.to("cpu") for k, v in cpu_inputs.items()}
        # cpu_outputs = model.generate(**cpu_inputs, max_new_tokens=512)
        # cpu_response_text = tokenizer.decode(cpu_outputs[0], skip_special_tokens=True)

        # 5) Generate on your designated device
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
        )
        raw = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 6) Clean up CSV formatting
        cleaned = clean_csv_response(raw)

        # 7) Return as JSON
        return jsonify({"generated_text": cleaned}), 200

    except Exception as e:
        app.logger.exception("Error in /restructure2")
        return jsonify({"error": str(e)}), 500
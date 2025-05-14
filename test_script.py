#!/usr/bin/env python3
import requests
import base64
import os

BASE_URL = "http://127.0.0.1:5000"

def post_pdf_multipart(filepath):
    """Send the PDF as multipart/form-data (Flask: request.files['file'])."""
    filename = os.path.basename(filepath)
    with open(filepath, "rb") as f:
        files = {
            "file": (filename, f, "application/pdf")
        }
        resp = requests.post(f"{BASE_URL}/restructure", files=files)
    resp.raise_for_status()
    print("multipart/form-data:", resp.status_code, resp.json())

def post_pdf_base64(filepath):
    """Send the PDF base64-encoded inside JSON (Flask: request.json['file'])."""
    with open(filepath, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    payload = {
        "filename": os.path.basename(filepath),
        "file": b64
    }
    resp = requests.post(f"{BASE_URL}/restructure", json=payload)
    resp.raise_for_status()
    print("application/json:", resp.status_code, resp.json())

if __name__ == "__main__":
    pdf_path = r"D:\Mistral Ex\Mistral-Example\AAR-Chronology-Americas-Freight-Railroads-Fact-Sheet.pdf"

    # Option B: JSON + base64
    post_pdf_base64(pdf_path)

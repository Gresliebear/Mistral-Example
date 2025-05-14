import os
import base64
from io import BytesIO, StringIO
from dotenv import load_dotenv
from flask import Flask, request, jsonify
import pandas as pd
import PyPDF2
from mistralai import Mistral  # adjust if needed

load_dotenv()
app = Flask(__name__)
MISTRAL_API_KEY = os.environ["MISTRAL_API_KEY"]

def extract_pdf_text(pdf_bytes: bytes) -> str:
    reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
    return "\n".join(page.extract_text() or "" for page in reader.pages)

def clean_csv_response(resp: str) -> str:
    lines = [line.strip().strip('"') for line in resp.splitlines() if line.strip()]
    return "\n".join(lines)

@app.route("/restructure", methods=["POST"])
def restructure():
    try:
        # 1) get PDF bytes from either multipart or JSON-base64
        if "file" in request.files:
            pdf_bytes = request.files["file"].read()
        else:
            data = request.get_json(force=True)
            b64 = data.get("file") or data.get("base64")
            if not b64:
                return jsonify({"error": "No file provided"}), 400
            pdf_bytes = base64.b64decode(b64)

        # 2) extract text
        document_text = extract_pdf_text(pdf_bytes)

        # 3) build prompt
        prompt = (
            "Extract all the key facts and their numeric values from the document below. "
            "Output only CSV with exactly two columns: Year, Invention, Fact, Value. "
            "Include units with Value column"
            "No headers, no commentary. "
            "If you cannot parse it, reply exactly “failed to structure”.\n\n"
            "-----DOCUMENT START-----\n"
            f"{document_text}\n"
            "-----DOCUMENT END-----"
        )

        # 4) call Mistral
        client = Mistral(api_key=MISTRAL_API_KEY)
        chat = client.chat.complete(
            model="mistral-large-latest",
            messages=[{"role": "user", "content": prompt}],
        )
        raw_csv = chat.choices[0].message.content

        # 5) clean & parse CSV
        cleaned = clean_csv_response(raw_csv)
        df = pd.read_csv(
            StringIO(cleaned),
            names=["Year", "Invention", "Fact", "Value"],
            engine="python",         # more tolerant parser
            skip_blank_lines=True,
            skipinitialspace=True,
            on_bad_lines="skip"      # drop any row that doesn’t split into exactly 4 fields
        )

        # 6) write Excel directly to disk
        output_path = "key_facts.xlsx"
        with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
            df.to_excel(writer, index=False, sheet_name="KeyFacts")

        # 7) report success
        return jsonify({
            "message": "File saved successfully",
            "path": output_path
        }), 200

    except Exception as e:
        app.logger.exception("Restructure failed")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

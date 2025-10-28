import fitz
from pdf2image import convert_from_bytes
import pytesseract
import google.generativeai as genai

gemini_api_key = "AIzaSyD3Zlhgi_ElXTxzZgmA1EqI9ECroDhmjPM"
genai.configure(api_key=gemini_api_key)

def analyze_document(file_bytes: bytes, filename: str) -> str:
    content = ""
    try:
        if filename.endswith(".pdf"):
            pdf = fitz.open(stream=file_bytes, filetype="pdf")
            for page in pdf:
                content += page.get_text()
            pdf.close()
            if not content.strip():
                images = convert_from_bytes(file_bytes)
                for img in images:
                    content += pytesseract.image_to_string(img)
        else:
            content = file_bytes.decode("utf-8", errors="ignore")
    except Exception as e:
        return f"Error reading file: {e}"
    
    if not content.strip():
        return "Uploaded file is empty or unreadable."

    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(
            f"You are a document analysis assistant. Analyze and summarize this content: {content}"
        )
        return response.text
    except Exception as e:
        return f"Error during document analysis: {e}"

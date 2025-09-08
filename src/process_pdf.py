import fitz  # PyMuPDF
import google.generativeai as genai
from PIL import Image
import io
import json
import os

# ---------------------------
# Setup Gemini
# ---------------------------
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))  # set your key as env var
model = genai.GenerativeModel("gemini-1.5-flash")


# ---------------------------
# Extract text per page
# ---------------------------
def extract_text(pdf_path):
    doc = fitz.open(pdf_path)
    text_chunks = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")
        if text.strip():
            text_chunks.append({
                "chunk_type": "text",
                "page_number": page_num,
                "content": text
            })
    return text_chunks


# ---------------------------
# Extract page images (whole page snapshots)
# ---------------------------
def extract_images(pdf_path):
    doc = fitz.open(pdf_path)
    images = []
    for page_num, page in enumerate(doc, start=1):
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # high resolution
        img_bytes = pix.tobytes("png")
        images.append({
            "page_number": page_num,
            "image_bytes": img_bytes
        })
    return images


# ---------------------------
# Send image to Gemini for parsing
# ---------------------------
def parse_chart_with_gemini(img_bytes):
    img = Image.open(io.BytesIO(img_bytes))

    prompt = (
        "Extract chart data from this page image in JSON format with keys: "
        "title, x_axis, y_axis, legend, values. "
        "If no chart is present, return {}."
    )

    response = model.generate_content([prompt, img])
    try:
        parsed = json.loads(response.text)
    except Exception:
        parsed = {"raw_output": response.text}
    return parsed


# ---------------------------
# Master pipeline
# ---------------------------
def process_pdf_for_rag(pdf_path):
    chunks = []

    # Extract text
    chunks.extend(extract_text(pdf_path))

    # Extract page images & send to Gemini
    image_chunks = extract_images(pdf_path)
    for img in image_chunks:
        parsed_data = parse_chart_with_gemini(img["image_bytes"])
        if parsed_data:  # skip empty charts
            chunks.append({
                "chunk_type": "chart",
                "page_number": img["page_number"],
                "content": parsed_data
            })

    return chunks


# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    pdf_file = "sample.pdf"  # replace with your file
    chunks = process_pdf_for_rag(pdf_file)

    # Save as JSON
    with open("extracted_data.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    print("\n✅ Extraction complete. Saved to extracted_data.json")
    print("\n📌 Sample chunks:")
    for c in chunks[:5]:
        print(json.dumps(c, indent=2, ensure_ascii=False), "\n")

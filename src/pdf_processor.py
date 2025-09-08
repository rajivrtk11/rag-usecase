import os
from typing import List, Dict
from PyPDF2 import PdfReader
import uuid
import fitz  # PyMuPDF
from PIL import Image
import io
import json
import google.generativeai as genai
from src.config import settings

class PDFProcessor:
    """Process PDF documents and extract text chunks."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Initialize Gemini model for image processing
        if settings.GOOGLE_API_KEY:
            genai.configure(api_key=settings.GOOGLE_API_KEY)
            self.gemini_model = genai.GenerativeModel("gemini-1.5-flash")
        else:
            self.gemini_model = None
            print("Warning: GOOGLE_API_KEY not set. Image processing functionality will be limited.")
    
    def extract_text(self, pdf_path: str) -> str:
        """Extract text from a PDF file."""
        try:
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            raise Exception(f"Error reading PDF file {pdf_path}: {str(e)}")
    
    def extract_images(self, pdf_path: str) -> List[Dict]:
        """Extract page images (whole page snapshots) from a PDF file."""
        try:
            doc = fitz.open(pdf_path)
            images = []
            for page_num, page in enumerate(doc, start=1):
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # high resolution
                img_bytes = pix.tobytes("png")
                images.append({
                    "page_number": page_num,
                    "image_bytes": img_bytes
                })
            doc.close()
            return images
        except Exception as e:
            raise Exception(f"Error extracting page images from PDF file {pdf_path}: {str(e)}")
    
    def parse_chart_with_gemini(self, img_bytes: bytes) -> List[Dict]:
        """Send image to Gemini for parsing multiple charts."""
        if not self.gemini_model:
            return [{"error": "Gemini model not initialized due to missing API key"}]
            
        try:
            img = Image.open(io.BytesIO(img_bytes))

            prompt = (
                "Analyze this page image and identify ALL charts, graphs, or visual data representations. "
                "For each chart found, extract the data in JSON format. "
                "Return an array of chart objects, where each chart has keys: title, x_axis, y_axis, legend, values. "
                "If no charts are found, return an empty array []. "
                "Example format: "
                '[{"title": "Chart 1 Title", "x_axis": [...], "y_axis": "...", "legend": [...], "values": [...]}, '
                '{"title": "Chart 2 Title", "x_axis": [...], "y_axis": "...", "legend": [...], "values": [...]}]'
            )

            response = self.gemini_model.generate_content([prompt, img])
            try:
                # Clean the response text to remove markdown code blocks
                response_text = response.text.strip()
                if response_text.startswith("```json"):
                    response_text = response_text[7:]  # Remove ```json
                if response_text.endswith("```"):
                    response_text = response_text[:-3]  # Remove ```
                response_text = response_text.strip()
                
                parsed = json.loads(response_text)
                print(f"Parsed image content: {parsed}")
                
                # Ensure it's a list
                if isinstance(parsed, list):
                    return parsed
                else:
                    return [parsed]  # Wrap single chart in array
                    
            except Exception as e:
                parsed = [{"raw_output": response.text}]
                print(f"Raw image content: {response.text}")
                return parsed
        except Exception as e:
            return [{"error": f"Error parsing image with Gemini: {str(e)}"}]
    
    def chunk_text(self, text: str, source: str = "") -> List[Dict]:
        """Split text into chunks with metadata."""
        chunks = []
        words = text.split()
        
        # Create overlapping chunks
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            if not chunk_words:
                break
                
            chunk_text = " ".join(chunk_words)
            chunk_id = str(uuid.uuid4())
            
            chunk = {
                "id": chunk_id,
                "text": chunk_text,
                "source": source,
                "chunk_index": len(chunks)
            }
            chunks.append(chunk)
            
            # Break if we've reached the end
            if i + self.chunk_size >= len(words):
                break
                
        return chunks
    
    def process_pdf(self, pdf_path: str) -> List[Dict]:
        """Process a PDF file and return text and chart chunks."""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
        chunks = []

        # Extract text
        text = self.extract_text(pdf_path)
        text_chunks = self.chunk_text(text, os.path.basename(pdf_path))
        chunks.extend(text_chunks)

        # Extract page images & send to Gemini
        image_chunks = self.extract_images(pdf_path)
        for img in image_chunks:
            charts_data = self.parse_chart_with_gemini(img["image_bytes"])
            
            # Process each chart found on this page
            for chart_index, chart_data in enumerate(charts_data):
                if chart_data and chart_data != {}:  # skip empty charts
                    chunks.append({
                        "id": str(uuid.uuid4()),
                        "chunk_type": "chart",
                        "page_number": img["page_number"],
                        "chart_index": chart_index + 1,  # 1-based index
                        "content": chart_data,
                        "source": os.path.basename(pdf_path)
                    })
                    print(f"✅ Added chart {chart_index + 1} from page {img['page_number']}")

        return chunks
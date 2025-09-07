import os
from typing import List, Dict
from PyPDF2 import PdfReader
import uuid

class PDFProcessor:
    """Process PDF documents and extract text chunks."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
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
        """Process a PDF file and return text chunks."""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
        text = self.extract_text(pdf_path)
        chunks = self.chunk_text(text, os.path.basename(pdf_path))
        return chunks
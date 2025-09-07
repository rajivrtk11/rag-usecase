from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import List
import os
import tempfile
from src.rag_system import RAGSystem

# Create the app instance
app = FastAPI(title="RAG System API", description="RAG system for PDF question answering")

# Global RAG system instance
rag_system = RAGSystem()

@app.on_event("startup")
async def startup_event():
    """Initialize the RAG system on startup."""
    success = await rag_system.initialize()
    if not success:
        raise RuntimeError("Failed to initialize RAG system")

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "RAG System API is running"}

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and process a PDF file."""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        # Process PDF
        success = await rag_system.process_pdf(tmp_file_path)
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        if success:
            return {"message": "PDF processed successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to process PDF")
            
    except Exception as e:
        # Clean up temporary file if it exists
        if 'tmp_file_path' in locals():
            try:
                os.unlink(tmp_file_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

@app.post("/query/")
async def query(query_text: str = Form(...)):
    """Query the RAG system."""
    try:
        response = await rag_system.query(query_text)
        return {"query": query_text, "response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/health/")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}
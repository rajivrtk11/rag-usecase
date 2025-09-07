from typing import List, Dict, Optional
import os
from src.pdf_processor import PDFProcessor
from src.vector_store import VectorStore
from src.llm import llm_provider
from src.config import settings

class RAGSystem:
    """Main RAG system that integrates all components."""
    
    def __init__(self):
        self.pdf_processor = PDFProcessor(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )
        self.vector_store = VectorStore()
        
    async def initialize(self) -> bool:
        """Initialize the RAG system components."""
        print("Initializing RAG system...")
        
        # Initialize vector store
        success = await self.vector_store.initialize()
        if not success:
            print("Failed to initialize vector store")
            return False
            
        print("RAG system initialized successfully!")
        return True
    
    async def process_pdf(self, pdf_path: str) -> bool:
        """Process a PDF file and add it to the vector store."""
        try:
            print(f"Processing PDF: {pdf_path}")
            
            # Process PDF
            chunks = self.pdf_processor.process_pdf(pdf_path)
            print(f"Extracted {len(chunks)} chunks from PDF")
            
            # Add to vector store
            success = await self.vector_store.add_documents(chunks)
            if success:
                print(f"Successfully added PDF chunks to vector store")
            else:
                print("Failed to add PDF chunks to vector store")
                
            return success
        except Exception as e:
            print(f"Error processing PDF: {str(e)}")
            return False
    
    async def query(self, query_text: str) -> str:
        """Query the RAG system and generate a response."""
        try:
            print(f"Processing query: {query_text}")
            
            # Search for relevant documents
            results = await self.vector_store.search(query_text, n_results=3)
            
            if not results:
                return "No relevant documents found."
            
            # Create context from search results
            context = "\n\n".join([f"Document: {result['metadata']['source']}\nContent: {result['text']}" 
                                 for result in results])
            
            # Create prompt for LLM
            prompt = f"""
            Use the following context to answer the question at the end. 
            If you don't know the answer, just say that you don't know, don't try to make up an answer.
            
            Context:
            {context}
            
            Question: {query_text}
            
            Answer:
            """
            
            # Generate response using LLM
            response = await llm_provider.generate_response(prompt)
            
            if response:
                return response
            else:
                return "Failed to generate response from LLM."
                
        except Exception as e:
            print(f"Error processing query: {str(e)}")
            return f"Error processing query: {str(e)}"
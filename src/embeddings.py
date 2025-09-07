import google.generativeai as genai
from typing import List, Optional
import asyncio
from src.config import settings

class EmbeddingProvider:
    """Provider for generating text embeddings using Google Generative AI."""
    
    def __init__(self):
        self.model_name = "models/text-embedding-004"
        if settings.GOOGLE_API_KEY:
            genai.configure(api_key=settings.GOOGLE_API_KEY)
            self.model = genai.embed_content
        else:
            self.model = None
            print("Warning: GOOGLE_API_KEY not set. Embedding functionality will be limited.")
    
    async def get_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for a text string."""
        if not self.model:
            # Return a dummy embedding if no API key is set
            # In a real implementation, you might use a local model here
            return [0.0] * 768
            
        try:
            # Google's embedding API is synchronous, so we run it in a thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, 
                lambda: genai.embed_content(
                    model=self.model_name,
                    content=text,
                    task_type="retrieval_document"
                )
            )
            return result['embedding']
        except Exception as e:
            print(f"Error generating embedding: {str(e)}")
            return None

# Global instance
embedding_provider = EmbeddingProvider()
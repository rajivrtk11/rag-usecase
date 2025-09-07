import google.generativeai as genai
from typing import Optional
import asyncio
from src.config import settings

class LLMProvider:
    """Provider for generating responses using Google Generative AI (Gemini)."""
    
    def __init__(self):
        if settings.GOOGLE_API_KEY:
            genai.configure(api_key=settings.GOOGLE_API_KEY)
            # Use a more current model name
            self.model = genai.GenerativeModel('gemini-1.5-flash')
        else:
            self.model = None
            print("Warning: GOOGLE_API_KEY not set. LLM functionality will be limited.")
    
    async def generate_response(self, prompt: str) -> Optional[str]:
        """Generate a response using the LLM."""
        if not self.model:
            return "LLM functionality is not available due to missing API key."
            
        try:
            # Generate content asynchronously
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.model.generate_content(prompt)
            )
            return response.text
        except Exception as e:
            print(f"Error generating LLM response: {str(e)}")
            return None

# Global instance
llm_provider = LLMProvider()
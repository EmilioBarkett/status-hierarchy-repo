# api_client.py

import asyncio
from openai import AsyncOpenAI
from config import OPENAI_API_KEY, MODEL_NAME, USE_DIFFERENT_MODELS, M1_MODEL, M2_MODEL


class ModelClient:
    """
    Handles API calls to OpenAI models.
    """
    
    def __init__(self, model_name=None):
        self.client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        self.model_name = model_name if model_name else MODEL_NAME
    
    async def get_rating(self, system_prompt, user_prompt):
        """
        Gets a rating from the model.
        
        Args:
            system_prompt: System prompt establishing model's characteristics
            user_prompt: User prompt with specific task
            
        Returns:
            Float rating between 0 and 1
        """
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=10
            )
            
            # Extract rating from response
            rating_text = response.choices[0].message.content.strip()
            rating = float(rating_text)
            
            # Ensure rating is between 0 and 1
            rating = max(0.0, min(1.0, rating))
            
            return rating
            
        except Exception as e:
            print(f"Error getting rating: {e}")
            return None
    
    async def process_conversation(self, system_prompt, messages):
        """
        Processes a multi-turn conversation with the model.
        
        Args:
            system_prompt: System prompt for the model
            messages: List of message dictionaries
            
        Returns:
            Model's response
        """
        try:
            full_messages = [{"role": "system", "content": system_prompt}] + messages
            
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=full_messages,
                temperature=0.7,
                max_tokens=10
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error in conversation: {e}")
            return None
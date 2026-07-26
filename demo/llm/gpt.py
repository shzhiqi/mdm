"""
This module contains the functions to generate responses from the LLM.
"""
from openai import OpenAI
from llm.base import BaseLLM
import base64
from PIL import Image
import io

class OpenAILLM(BaseLLM):
    """
    A class that wraps the OpenAI llm.
    """
    def __init__(self, model: str):
        """
        Initialize the OpenAI client.
        """
        self.client = OpenAI()
        self.model = model

    def generate_response(self, user: str, **kwargs) -> str:
        """
        Generate a response from the LLM.
        """
        if "images" in kwargs:
            images = kwargs["images"]
            texts = user.split("<image_placeholder>")
            content = []
            for text_idx, text in enumerate(texts[:-1]):
                content.append({"type": "text", "text": texts[text_idx]})
                image_base64 = iamge_to_base64(images[text_idx])
                content.append({"type": "image_url", "image_url": {"url": image_base64}})
            if len(texts) > len(images):
                content.append({"type": "text", "text": texts[-1]})
            messages = [
                {
                    "role": "user",
                    "content": content
                }
            ]
        else:
            messages = [
                {
                    "role": "user",
                    "content": user
                }
            ]
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=messages
        )
        return response.choices[0].message.content


def iamge_to_base64(image_path: str) -> str:
    """
    Convert an image to base64.
    """
    # Load or create your image
    image = Image.open(image_path)  # or use Image.new()

    # Convert to JPEG base64
    buffered = io.BytesIO()
    image.convert("RGB").save(buffered, format="JPEG")  # Ensure it's in RGB for JPEG
    img_bytes = buffered.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")

    # Optional: for HTML embedding
    img_base64_str = f"data:image/jpeg;base64,{img_base64}"
    return img_base64_str

if __name__ == "__main__":
    llm = OpenAILLM(model="gpt-4o-mini")
    response = llm.generate_response("Hello, how are you?")
    print(response)
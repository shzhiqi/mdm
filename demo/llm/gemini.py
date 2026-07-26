import os
from google import genai
from google.genai import types
from PIL import Image
import io
from llm.base import BaseLLM


class GeminiLLM(BaseLLM):
    def __init__(self, model: str):
        self.model = model
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    def generate_response(self, user: str, **kwargs) -> str:
        contents = []
        if "images" in kwargs:
            images = kwargs["images"]
            texts = user.split("<image_placeholder>")
            texts = [x for x in texts if x]
            for text_idx, text in enumerate(texts):
                contents.append(text)
                if text_idx < len(images):
                    contents.append(self._image_to_part(images[text_idx]))
        else:
            contents = [user]

        response = self.client.models.generate_content(
            model=self.model,
            contents=contents
        )
        return response.text

    def _image_to_part(self, image_path: str) -> types.Part:
        image = Image.open(image_path)
        buffered = io.BytesIO()
        image.convert("RGB").save(buffered, format="JPEG")
        return types.Part.from_bytes(data=buffered.getvalue(), mime_type="image/jpeg")


if __name__ == "__main__":
    llm = GeminiLLM(model="gemini-2.5-flash")
    response = llm.generate_response(
        user="What are in this image? <image_placeholder>",
        images=[os.path.join(os.path.dirname(__file__), "..", "tests", "dataset", "test_images", "Snopes", "1321_spc.jpg")]
    )
    print(response)

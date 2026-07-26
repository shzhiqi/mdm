import base64
import io
from PIL import Image
import anthropic
import os
from llm.base import BaseLLM


class ClaudeLLM(BaseLLM):
    """
    A class that wraps the Claude llm.
    """
    def __init__(self, model: str):
        self.model = model
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def generate_response(self, user: str, **kwargs) -> str:
        if "images" in kwargs:
            images = kwargs["images"]
            texts = user.split("<image_placeholder>")
            texts = [x for x in texts if x]
            content = []
            for text_idx, text in enumerate(texts):
                content.append({"type": "text", "text": text})
                if text_idx < len(images):
                    image_base64 = image_to_base64(images[text_idx])
                    content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_base64
                        }
                    })
            messages = [{"role": "user", "content": content}]
        else:
            messages = [{"role": "user", "content": user}]

        response = self.client.messages.create(
            model=self.model,
            max_tokens=8096,
            messages=messages
        )
        return response.content[0].text


def image_to_base64(image_path: str) -> str:
    image = Image.open(image_path)
    image = image.resize((640, 480))
    buffered = io.BytesIO()
    image.convert("RGB").save(buffered, format="JPEG")
    img_bytes = buffered.getvalue()
    return base64.b64encode(img_bytes).decode("utf-8")


if __name__ == "__main__":
    llm = ClaudeLLM(model="claude-opus-4-8")
    images = [os.path.join(os.path.dirname(__file__), "..", "tests", "dataset", "test_images", "Snopes", "1321_spc.jpg")]
    response = llm.generate_response(user="What is the capital of France? <image_placeholder>", images=images)
    print(response)

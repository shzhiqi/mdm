"""

"""
import sys
import os

from mcp.server.fastmcp import FastMCP
from jinja2 import Template
import logging
import json
from openai import OpenAI
from pathlib import Path
import torch
import clip
from PIL import Image
from typing import Any, Dict
from models import MultiTaskModelV3_2

# save file log to a file in current directory
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler(os.path.join(os.path.dirname(__file__), "mdm_emotion_tools.log"))
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s\n")
handler.setFormatter(formatter)
logger.addHandler(handler)

# Initialize FastMCP server
mcp = FastMCP("mdm_emotion_tools")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = MultiTaskModelV3_2(device=device)
model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), 'model_2.pt'), map_location=device))
model.eval()
_, preprocess = clip.load('ViT-B/32', device=device, jit=False)

@mcp.tool()
async def get_human_perceptions(news_image_path: str, news_text: str) -> dict[str, Any] | None:
    """
    Get the human perception of News, the News includes an image and a short text. return scores scale from 0 to 1.
    Args:
        news_image_path: str, the path to the news image
        news_text: str, the text of the news
    Returns:
        return a dict include the human perceptions of the news:
        1. pred_ai_likelihood: the human guess of the news, indicating human's guess of the news is true or false
        2. pred_dissemination: the human sharing of the news, indicating how human will share the news
        3. pred_belief: the human believability of the news, indicating how human will believe the news
    """
    img = Image.open(news_image_path)
    text = clip.tokenize(news_text).to(device)
    image = preprocess(img).to(device).unsqueeze(dim=0)
    pred_guess, pred_sharing, pred_believable, emo_consist, sem_consist = model(image, text)
    return {
        "pred_ai_likelihood": pred_guess.item(),
        "pred_dissemination": pred_sharing.item(),
        "pred_belief": pred_believable.item()
    }

# @mcp.tool()
# def get_emotion_consistency(image_path: str, text: str):
#     """
#     Get the emotion consistency of the image and text. return scores scale from 0 to 1.
#     Args:
#         image: PIL.Image, the image of the news
#         text: str, the text of the news
#     """
#     img = Image.open(image_path)
#     text = clip.tokenize(text).to(device)
#     image = preprocess(img).to(device).unsqueeze(dim=0)
#     pred_guess, pred_sharing, pred_believable, emo_consist, sem_consist = model(image, text)
#     return {
#         "emo_consist": emo_consist.item()
#     }

# @mcp.tool()
# def get_semantic_consistency(image_path: str, text: str):
#     """
#     Get the semantic consistency of the image and text. return scores scale from 0 to 1.
#     Args:
#         image: PIL.Image, the image of the news
#         text: str, the text of the news
#     """
#     img = Image.open(image_path)
#     text = clip.tokenize(text).to(device)
#     image = preprocess(img).to(device).unsqueeze(dim=0)
#     pred_guess, pred_sharing, pred_believable, emo_consist, sem_consist = model(image, text)
#     return {    
#         "sem_consist": sem_consist.item()
#     }

if __name__ == "__main__":
    mcp.run()
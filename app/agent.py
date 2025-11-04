import os
import random
import logging

import google.auth
from google.adk.agents import Agent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

THEMES = [
    "Chaotic", "Nonsensical", "Mundane", "Vaguely religious",
    "Self Discovery", "Prophetically hopeful", "Prophetically dark"
]

_, project_id = google.auth.default()
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", project_id)
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "global")
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "True")

def get_vision_themes() -> str:
    """Selects two random themes for a vision."""
    themes = ", ".join(random.sample(THEMES, 2))
    logger.info(f"Selected themes: {themes}")
    return themes
    
root_agent = Agent(
    name="Oracle",
    model="gemini-2.5-flash",
    instruction="""You are an Oracle. You produce visions for pilgrims.
The visions are vivid and should remain open to the user's interpretation.
You provide visions not advice or interpretation.
The text description of the vision should always rhyme.
Use the `get_vision_themes` tool to select two themes for the vision.
""",
    tools=[get_vision_themes],
)

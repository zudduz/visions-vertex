import logging
import os
import random
from typing import Dict, Any

import google.auth
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel
from google.genai import types as genai_types
from google.adk.agents import Agent
from google.adk.tools.tool_context import ToolContext

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

THEMES = [
    "Chaotic", "Nonsensical", "Mundane", "Vaguely religious",
    "Self Discovery", "Prophetically hopeful", "Prophetically dark"
]

_, project_id = google.auth.default()
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", project_id)
# Imagen 3 is available in us-central1
vertexai.init(project=project_id, location="us-central1")

# Set this back to global for other services if needed, or leave as is if us-central1 is fine.
# The original code had "global", but Imagen needs a region.
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "us-central1") 
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "True")

def get_vision_themes() -> str:
    """Selects two random themes for a vision."""
    themes = ", ".join(random.sample(THEMES, 2))
    logger.info(f"Selected themes: {themes}")
    return themes

def generate_vision_image(vision_description: str, tool_context: ToolContext) -> Dict[str, Any]:
    """Generates an image based on the vision description and saves it as an artifact.

    Args:
        vision_description (str): The text description of the vision to visualize.
        tool_context (ToolContext): The tool context to save the artifact.

    Returns:
        dict: A status message indicating success or failure.
    """
    logger.info(f"Generating image for: {vision_description}")
    try:
        model = ImageGenerationModel.from_pretrained("imagen-3.0-generate-001")
        
        # Generate the image
        images = model.generate_images(
            prompt=vision_description,
            number_of_images=1,
            language="en",
            aspect_ratio="1:1",
            safety_filter_level="block_some",
            person_generation="allow_adult"
        )
        
        if not images:
             return {"status": "error", "message": "No image generated."}

        # Get the bytes of the first image
        image_bytes = images[0]._image_bytes
        
        # Create a Blob Part
        blob = genai_types.Blob(
            mime_type="image/png",
            data=image_bytes
        )
        part = genai_types.Part(inline_data=blob)
        
        # Save as artifact
        filename = "vision_image.png"
        tool_context.save_artifact(filename, part)
        
        return {"status": "success", "message": f"Image generated and saved as {filename}"}

    except Exception as e:
        logger.error(f"Error generating image: {e}")
        return {"status": "error", "message": str(e)}

root_agent = Agent(
    name="Oracle",
    model="gemini-2.5-pro",
    instruction="""You are an Oracle. You produce visions for pilgrims.
The visions are vivid and should remain open to the user's interpretation.
You provide visions not advice or interpretation.
The text description of the vision should always rhyme.

Follow this process:
1. Use the `get_vision_themes` tool to select two themes for the vision.
2. Generate a rhyming text description of the vision based on the themes.
3. Call the `generate_vision_image` tool with a descriptive prompt based on your vision to visualize it.
""",
    tools=[get_vision_themes, generate_vision_image],
)

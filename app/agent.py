import logging
import os
import random
import uuid
from typing import Dict, Any

import google.auth
import google.cloud.storage as storage
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel
from google.genai import types as genai_types
from google.adk.agents import Agent, SequentialAgent
from google.adk.tools.tool_context import ToolContext
from pydantic import BaseModel, Field

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

# Bucket name for storing images
VISION_BUCKET_NAME = f"{project_id}-oracle-visions"

def get_vision_themes() -> str:
    """Selects two random themes for a vision."""
    themes = ", ".join(random.sample(THEMES, 2))
    logger.info(f"Selected themes: {themes}")
    return themes

def generate_vision_image(vision_description: str, tool_context: ToolContext) -> Dict[str, Any]:
    """Generates an image based on the vision description and uploads it to GCS.

    Args:
        vision_description (str): The text description of the vision to visualize.
        tool_context (ToolContext): The tool context to save state.

    Returns:
        dict: A status message containing the public URL of the image.
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
        
        # Upload to GCS
        storage_client = storage.Client(project=os.environ.get("GOOGLE_CLOUD_PROJECT"))
        bucket = storage_client.bucket(VISION_BUCKET_NAME)
        
        # Ensure bucket exists (simple check, or rely on setup)
        if not bucket.exists():
            bucket = storage_client.create_bucket(VISION_BUCKET_NAME, location="us-central1")
            # Make bucket public readable if required, or just the objects
            # For simplicity, we'll try to make the object public. 
            # Note: Public access prevention on the bucket might block this.
        
        blob_name = f"visions/{uuid.uuid4()}.png"
        blob = bucket.blob(blob_name)
        blob.upload_from_string(image_bytes, content_type="image/png")
        
        # Make the blob publicly accessible (Optional: depends on security policy)
        # For a truly public app without auth, this is needed.
        # Use try-except as this might fail if "Domain Restricted Sharing" or "Public Access Prevention" is on.
        try:
            blob.make_public()
            public_url = blob.public_url
        except Exception as auth_e:
            logger.warning(f"Could not make blob public: {auth_e}")
            # Fallback to authenticated link or signed URL if needed, but for now return the direct link
            # which works if the bucket is public or user has access.
            public_url = f"https://storage.googleapis.com/{VISION_BUCKET_NAME}/{blob_name}"

        # Save URL to state for the formatter agent
        tool_context.state["generated_image_url"] = public_url
        
        return {"status": "success", "url": public_url}

    except Exception as e:
        logger.error(f"Error generating/uploading image: {e}")
        return {"status": "error", "message": str(e)}

# 1. Define the Output Schema
class OracleResponse(BaseModel):
    vision_text: str = Field(description="The rhyming text description of the vision.")
    image_url: str = Field(description="The public URL of the generated vision image.")

# 2. Define the Generator Agent (Uses Tools)
vision_generator = Agent(
    name="vision_generator",
    model="gemini-2.5-pro",
    instruction="""You are an Oracle's creative mind.
    1. Use `get_vision_themes` to pick themes.
    2. Generate a rhyming vision description based on them.
    3. Use `generate_vision_image` to visualize that EXACT description.
    
    Output ONLY the rhyming vision text. Do not output JSON.
    """,
    tools=[get_vision_themes, generate_vision_image],
    output_key="vision_text_content" # Save raw text to state
)

# 3. Define the Formatter Agent (Enforces Schema)
vision_formatter = Agent(
    name="vision_formatter",
    model="gemini-2.5-flash",
    instruction="""You are a formatter. 
    Construct a response using the data provided in the session state.
    - vision_text: {vision_text_content}
    - image_url: {generated_image_url}
    """,
    output_schema=OracleResponse
)

# 4. Define the Sequential Pipeline
root_agent = SequentialAgent(
    name="Oracle",
    description="Generates a rhyming vision and a matching visualization.",
    sub_agents=[vision_generator, vision_formatter]
)

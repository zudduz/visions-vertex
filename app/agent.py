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

# --- Configuration and Initialization ---

location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")

_, project_id = google.auth.default()
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", project_id)
vertexai.init(project=project_id, location=location)

os.environ.setdefault("GOOGLE_CLOUD_LOCATION", location)
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "True")

VISION_BUCKET_NAME = f"{project_id}-oracle-visions"

def initialize_gcs():
    """Checks for GCS bucket and creates it with a lifecycle rule if it doesn't exist."""
    logger.info(f"Initializing GCS bucket: {VISION_BUCKET_NAME}")
    try:
        storage_client = storage.Client(project=project_id)
        bucket = storage_client.bucket(VISION_BUCKET_NAME)
        
        if not bucket.exists():
            logger.info(f"Bucket {VISION_BUCKET_NAME} not found. Creating...")
            new_bucket = storage_client.create_bucket(VISION_BUCKET_NAME, location=location)
            logger.info(f"Bucket {VISION_BUCKET_NAME} created in {location}.")
            
            # Set the 7-day deletion lifecycle rule on the new bucket
            new_bucket.lifecycle_rules = [
                {"action": {"type": "Delete"}, "condition": {"age": 7}}
            ]
            new_bucket.patch()
            logger.info("7-day expiration rule set for the bucket.")
        else:
            logger.info(f"Bucket {VISION_BUCKET_NAME} already exists.")
            
    except Exception as e:
        # Log the error and continue; the tool call will likely fail, but we don't want to crash at startup.
        logger.error(f"Failed to initialize GCS bucket: {e}")

# Run GCS initialization at startup
initialize_gcs()

# --- Tools ---

def get_vision_themes() -> str:
    """Selects two random themes for a vision."""
    themes = ", ".join(random.sample(THEMES, 2))
    logger.info(f"Selected themes: {themes}")
    return themes

def generate_vision_image(vision_description: str, tool_context: ToolContext) -> Dict[str, Any]:
    """Generates an image and uploads it to GCS."""
    logger.info(f"Generating image for: {vision_description}")
    try:
        model = ImageGenerationModel.from_pretrained("imagen-3.0-generate-001")
        images = model.generate_images(prompt=vision_description, number_of_images=1)
        
        if not images:
             return {"status": "error", "message": "No image generated."}

        image_bytes = images[0]._image_bytes
        
        storage_client = storage.Client(project=project_id)
        bucket = storage_client.bucket(VISION_BUCKET_NAME)

        blob_name = f"visions/{uuid.uuid4()}.png"
        blob = bucket.blob(blob_name)
        blob.upload_from_string(image_bytes, content_type="image/png")
        
        try:
            blob.make_public()
            public_url = blob.public_url
        except Exception as auth_e:
            logger.warning(f"Could not make blob public: {auth_e}")
            public_url = f"https://storage.googleapis.com/{VISION_BUCKET_NAME}/{blob_name}"

        tool_context.state["generated_image_url"] = public_url
        return {"status": "success", "url": public_url}

    except Exception as e:
        logger.error(f"Error in generate_vision_image: {e}")
        return {"status": "error", "message": str(e)}

# --- Agent Definitions ---

class OracleResponse(BaseModel):
    vision_text: str = Field(description="The rhyming text description of the vision.")
    image_url: str = Field(description="The public URL of the generated vision image.")

text_generator = Agent(
    name="text_generator",
    model="gemini-2.5-pro",
    instruction="""You are an Oracle's creative mind.\n1. Use `get_vision_themes` to pick your themes.\n2. Generate a 4-line rhyming vision description based on the themes.\nOutput ONLY the rhyming vision text.""",
    tools=[get_vision_themes],
    output_key="vision_text"
)

image_generator = Agent(
    name="image_generator",
    model="gemini-2.5-flash",
    instruction="""You are an image generation specialist.\nUse the `generate_vision_image` tool.\nUse the vision text provided in `{vision_text}` as the `vision_description` for the tool.""",
    tools=[generate_vision_image],
)

vision_formatter = Agent(
    name="vision_formatter",
    model="gemini-2.5-flash",
    instruction="""You are a formatter. \nConstruct a JSON response using the data provided in the session state.\n- vision_text: {vision_text}\n- image_url: {generated_image_url}""",
    output_schema=OracleResponse,
    include_contents="none"
)

root_agent = SequentialAgent(
    name="Oracle",
    description="Generates a rhyming vision and a matching visualization.",
    sub_agents=[text_generator, image_generator, vision_formatter]
)

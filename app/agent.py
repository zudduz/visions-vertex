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
from google.adk.agents.callback_context import CallbackContext

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

THEMES = [
    "Chaotic", "Nonsensical", "Mundane", "Vaguely religious", "Self Discovery",
    "Prophetically hopeful", "Prophetically dark"
]

# --- Configuration and Initialization ---

location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")

# Hardcode the project ID to prevent issues where the environment
# provides a project number instead of the string ID.
project_id = "sandbox-456821"
creds, _ = google.auth.default()
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", project_id)
vertexai.init(project=project_id, location=location, credentials=creds)

os.environ.setdefault("GOOGLE_CLOUD_LOCATION", location)
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "True")

# IMPORTANT: This bucket must be created manually before running the agent.
# See the README.md file for setup instructions.
VISION_BUCKET_NAME = f"{project_id}-oracle-visions"

# --- Tools ---


def get_vision_themes() -> str:
    """Selects two random themes for a vision."""
    themes = ", ".join(random.sample(THEMES, 2))
    logger.info(f"Selected themes: {themes}")
    return themes


def generate_vision_image(vision_description: str,
                          tool_context: ToolContext) -> Dict[str, Any]:
    """Generates an image, uploads it to GCS, and saves the public URL to session state."""
    logger.info(f"Attempting to generate image for: {vision_description}")
    try:
        model = ImageGenerationModel.from_pretrained("imagen-3.0-generate-001")
        images = model.generate_images(prompt=vision_description,
                                       number_of_images=1)

        if not images:
            logger.error("Image generation failed, no images returned.")
            tool_context.state["generated_image_url"] = ""
            return {"status": "error", "message": "No image was generated."}

        image_bytes = images[0]._image_bytes

        storage_client = storage.Client(project=project_id)
        bucket = storage_client.bucket(VISION_BUCKET_NAME)

        blob_name = f"visions/{uuid.uuid4()}.png"
        blob = bucket.blob(blob_name)

        blob.upload_from_string(image_bytes, content_type="image/png")

        public_url = f"https://storage.googleapis.com/{VISION_BUCKET_NAME}/{blob_name}"

        tool_context.state["generated_image_url"] = public_url

        return {"status": "success", "url": public_url}

    except Exception as e:
        logger.error(f"An exception occurred in generate_vision_image: {e}",
                     exc_info=True)
        tool_context.state["generated_image_url"] = ""
        return {"status": "error", "message": str(e)}


# --- Callbacks for Logging ---


def log_state_callback(callback_context: CallbackContext) -> None:
    """Logs the current state before an agent runs."""
    state = callback_context.state
    agent_name = callback_context._invocation_context.agent.name
    logger.info(f"--- Running agent: {agent_name} ---")

    if "vision_text" in state:
        logger.info(f"State['vision_text']: {state.get('vision_text')}")
    else:
        logger.info("State['vision_text']: Not found")

    if "generated_image_url" in state:
        logger.info(
            f"State['generated_image_url']: {state.get('generated_image_url')}"
        )
    else:
        logger.info("State['generated_image_url']: Not found")
    logger.info("-----------------------------------------")


# --- Agent Definitions ---


class OracleResponse(BaseModel):
    vision_text: str = Field(
        description="The rhyming text description of the vision.")
    image_url: str = Field(
        default="",
        description="The public URL of the generated vision image.")


text_generator = Agent(name="text_generator",
                       model="gemini-2.5-pro",
                       instruction="""You are an Oracle's creative mind.
1. Use `get_vision_themes` to pick your themes.
2. Generate a 4-line rhyming vision description based on the themes.
Output ONLY the rhyming vision text.""",
                       tools=[get_vision_themes],
                       output_key="vision_text",
                       before_agent_callback=log_state_callback)

image_generator = Agent(name="image_generator",
                        model="gemini-2.5-flash",
                        instruction="""You are an image generation specialist.
Use the `generate_vision_image` tool with the vision text provided in `{vision_text}`.
Acknowledge completion of the task.
""",
                        tools=[generate_vision_image],
                        before_agent_callback=log_state_callback)

vision_formatter = Agent(name="vision_formatter",
                         model="gemini-2.5-flash",
                         instruction="""You are a formatter.
Construct a JSON response using the data provided in the session state.
CRITICAL: You MUST preserve the line breaks from the `vision_text` by using a `\n` in the final JSON string.
- vision_text: {vision_text}
- image_url: {generated_image_url}
""",
                         output_schema=OracleResponse,
                         include_contents="none",
                         before_agent_callback=log_state_callback)

root_agent = SequentialAgent(
    name="Oracle",
    description="Generates a rhyming vision and a matching visualization.",
    sub_agents=[text_generator, image_generator, vision_formatter])

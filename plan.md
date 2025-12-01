# Plan: Structured Output and Image Serving for Oracle Agent

To provide a consistently parsable response and serve the generated image to users, we need to upload the image to a publicly accessible Google Cloud Storage (GCS) bucket instead of just saving it as a local session artifact.

## Steps

1.  **Define Output Schema**
    *   Create a Pydantic model `OracleResponse` with fields:
        *   `vision_text` (str): The rhyming prophecy.
        *   `image_url` (str): The public URL of the generated image.

2.  **Implement GCS Upload Logic**
    *   Modify `generate_vision_image` (or create a new helper/tool) to:
        *   Initialize the Vertex AI Image Generation Model.
        *   Generate the image.
        *   Initialize a GCS client.
        *   Define a bucket name (e.g., `{project_id}-oracle-visions`).
        *   Ensure the bucket exists (use `app.utils.gcs.create_bucket_if_not_exists` if appropriate, or assume setup).
        *   Upload the image bytes to a blob in the bucket (e.g., `visions/{uuid}.png`).
        *   **Important**: Make the blob publicly accessible (or use signed URLs, but public read is simpler for "anonymous web user").
        *   Return the public URL (`https://storage.googleapis.com/{bucket}/{blob_name}`).

3.  **Refactor Agents**
    *   **Agent 1: `vision_generator`**
        *   Type: `Agent` (LlmAgent).
        *   Tools: `get_vision_themes`, `generate_vision_image` (updated to return URL).
        *   Instruction: Generate themes, generate image using the tool, and output the rhyming text. The tool call will return the URL.
        *   `output_key`: `"vision_text_content"` (Saves the rhyme).
    *   **Agent 2: `vision_formatter`**
        *   Type: `Agent` (LlmAgent).
        *   Tools: None.
        *   `output_schema`: `OracleResponse`.
        *   Instruction: Construct the response. Use `{vision_text_content}` for the text. Access the image URL from the tool response history or explicit state passing.
        *   *Refinement*: To make state passing cleaner, `vision_generator`'s tool `generate_vision_image` should probably save the URL to `tool_context.state['generated_image_url']` in addition to returning it. The formatter can then read `{generated_image_url}`.

4.  **Create Pipeline**
    *   `SequentialAgent`: `[vision_generator, vision_formatter]`.

5.  **Configuration**
    *   Need `GOOGLE_CLOUD_PROJECT` available.
    *   Permissions: The service account running the agent needs `Storage Object Admin` or similar on the bucket.

## Modified `generate_vision_image` Tool Logic
*   Takes `vision_description` and `tool_context`.
*   Generates image.
*   Uploads to GCS.
*   Sets `tool_context.state['generated_image_url'] = public_url`.
*   Returns `{"status": "success", "url": public_url}`.

## Formatter Instruction
*   "Construct the response using: vision_text='{vision_text_content}', image_url='{generated_image_url}'."

This ensures the user gets a valid JSON with a clickable URL.

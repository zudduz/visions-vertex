# Visionary Oracle Agent

This agent generates rhyming, prophetic visions and creates an accompanying image.

## Prerequisites

Before running this agent, you must manually create and configure a Google Cloud Storage (GCS) bucket for the images to be stored. This is a one-time setup.

### 1. Create the GCS Bucket

First, you need to create a GCS bucket. The name used in the agent is based on your Google Cloud Project ID.

1.  **Find your Project ID:** You can find this in the Google Cloud Console dashboard. For this project, it is `sandbox-456821`.
2.  **Construct the Bucket Name:** The bucket name must be `[YOUR_PROJECT_ID]-oracle-visions`. For this project, the name will be `sandbox-456821-oracle-visions`.
3.  **Create the Bucket:** Use the `gsutil` command-line tool to create the bucket.

    ```bash
    gsutil mb -p sandbox-456821 -l us-central1 gs://sandbox-456821-oracle-visions
    ```

### 2. Set Public Access for the Bucket

For the generated image URLs to be viewable, you must make the objects in the bucket publicly readable.

*   **Run the following `gsutil` command:**

    ```bash
    gsutil iam ch allUsers:objectViewer gs://sandbox-456821-oracle-visions
    ```

### 3. Grant Permissions to the Agent's Service Account

The agent runs on Vertex AI Agent Engine and uses a specific service account to interact with other Google Cloud services. You must grant this service account permission to upload files to the bucket.

1.  **Identify the Service Account:** The service account email address is constructed based on your project number. You can find this by running the agent once and viewing the error logs, or by finding the service account in the IAM section of the Google Cloud Console. For this project, the service account is:
    `service-171510694317@gcp-sa-aiplatform-re.iam.gserviceaccount.com`

2.  **Grant the `Storage Object Creator` Role:** In the IAM section of the Google Cloud Console, find this service account and grant it the **`Storage Object Creator`** role. If you cannot see the service account, make sure to check the box that says **"Include Google-provided role grants"**.

## Running the Agent

Once the GCS bucket and permissions are configured, you can deploy and run the agent as usual. The agent will now use the bucket you created to store and serve the generated images.

import logging
import click
import google.auth
import vertexai
from google.adk.artifacts import GcsArtifactService
from vertexai._genai.types import AgentEngine, AgentEngineConfig
from vertexai.agent_engines.templates.adk import AdkApp

from app.agent import root_agent
from app.utils.deployment import (
    parse_env_vars,
    print_deployment_success,
    write_deployment_metadata,
)
from app.utils.gcs import create_bucket_if_not_exists


@click.command()
@click.option(
    "--project",
    default=None,
    help="GCP project ID (defaults to application default credentials)",
)
@click.option(
    "--location",
    default="us-central1",
    help="GCP region (defaults to us-central1)",
)
@click.option(
    "--agent-name",
    default="oracle",
    help="Name for the agent engine",
)
@click.option(
    "--requirements-file",
    default=".requirements.txt",
    help="Path to requirements.txt file",
)
@click.option(
    "--extra-packages",
    multiple=True,
    default=["./app"],
    help="Additional packages to include",
)
@click.option(
    "--set-env-vars",
    default=None,
    help="Comma-separated list of environment variables in KEY=VALUE format",
)
@click.option(
    "--service-account",
    default=None,
    help="Service account email to use for the agent engine",
)
@click.option(
    "--staging-bucket-uri",
    default=None,
    help="GCS bucket URI for staging files (defaults to gs://{project}-agent-engine)",
)
@click.option(
    "--artifacts-bucket-name",
    default=None,
    help="GCS bucket name for artifacts (defaults to gs://{project}-agent-engine)",
)
def deploy_agent_engine_app(
    project: str | None,
    location: str,
    agent_name: str,
    requirements_file: str,
    extra_packages: tuple[str, ...],
    set_env_vars: str | None,
    service_account: str | None,
    staging_bucket_uri: str | None,
    artifacts_bucket_name: str | None,
) -> AgentEngine:
    """Deploy the agent engine app to Vertex AI."""

    logging.basicConfig(level=logging.INFO)

    # Parse environment variables if provided
    env_vars = parse_env_vars(set_env_vars)

    if not project:
        _, project = google.auth.default()
    if not staging_bucket_uri:
        staging_bucket_uri = f"gs://{project}-agent-engine"
    if not artifacts_bucket_name:
        artifacts_bucket_name = f"{project}-agent-engine"
    create_bucket_if_not_exists(
        bucket_name=artifacts_bucket_name, project=project, location=location
    )
    create_bucket_if_not_exists(
        bucket_name=staging_bucket_uri, project=project, location=location
    )

    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║   🤖 DEPLOYING AGENT TO VERTEX AI AGENT ENGINE 🤖         ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """)

    extra_packages_list = list(extra_packages)

    # Initialize vertexai client
    client = vertexai.Client(
        project=project,
        location=location,
    )
    vertexai.init(project=project, location=location)

    # Read requirements
    with open(requirements_file) as f:
        requirements = f.read().strip().split("\n")
    
    # Use the standard AdkApp directly
    agent_engine = AdkApp(
        agent=root_agent,
        enable_tracing=True,
        artifact_service_builder=lambda: GcsArtifactService(
            bucket_name=artifacts_bucket_name
        ),
    )
    # Set worker parallelism to 0
    env_vars["NUM_WORKERS"] = "0"

    # Common configuration for both create and update operations
    labels: dict[str, str] = {}

    config = AgentEngineConfig(
        display_name=agent_name,
        description="A base ReAct agent built with Google's Agent Development Kit (ADK)",
        extra_packages=extra_packages_list,
        env_vars=env_vars,
        service_account=service_account,
        requirements=requirements,
        staging_bucket=staging_bucket_uri,
        labels=labels,
        gcs_dir_name=agent_name,
    )

    agent_config = {
        "agent": agent_engine,
        "config": config,
    }
    logging.info(f"Agent config: {agent_config}")

    # Check if an agent with this name already exists
    existing_agents = list(client.agent_engines.list())
    matching_agents = [
        agent
        for agent in existing_agents
        if agent.api_resource.display_name == agent_name
    ]

    if matching_agents:
        # Update the existing agent with new configuration
        logging.info(f"\n📝 Updating existing agent: {agent_name}")
        remote_agent = client.agent_engines.update(
            name=matching_agents[0].api_resource.name, **agent_config
        )
    else:
        # Create a new agent if none exists
        logging.info(f"\n🚀 Creating new agent: {agent_name}")
        remote_agent = client.agent_engines.create(**agent_config)

    write_deployment_metadata(remote_agent)
    print_deployment_success(remote_agent, location, project)

    return remote_agent


if __name__ == "__main__":
    deploy_agent_engine_app()

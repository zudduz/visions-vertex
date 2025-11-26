import logging
import vertexai
from vertexai.preview import reasoning_engines

# Set up logging
logging.basicConfig(level=logging.INFO)

def main():
    """Runs the Oracle agent against the deployed Vertex AI Agent Engine."""
    print("--- Starting Oracle Agent Remote Test ---")

    # Configuration from user request
    project_id = "sandbox-456821"
    location = "us-central1"
    reasoning_engine_id = "1352192593978458112"
    
    resource_name = f"projects/{project_id}/locations/{location}/reasoningEngines/{reasoning_engine_id}"

    print(f"Connecting to Agent Engine: {resource_name}")
    
    # Initialize Vertex AI SDK
    vertexai.init(project=project_id, location=location)
    
    try:
        # Get the remote agent instance
        remote_agent = reasoning_engines.ReasoningEngine(resource_name)
        
        # print(f"Remote Agent object: {remote_agent}")
        # print(f"Available attributes: {dir(remote_agent)}")
        
        query_text = "I seek a vision for my future."
        print(f"Pilgrim asks: '{query_text}'")
        print("Oracle is thinking (remotely)...")
        
        # Use query(input=...) which we have explicitly implemented and registered
        if hasattr(remote_agent, 'query'):
            response = remote_agent.query(input=query_text)
            print(f"\nOracle's Vision:\n{response}")
        else:
            print("ERROR: No 'query' method found on remote agent.")
            print(f"Available methods: {[m for m in dir(remote_agent) if not m.startswith('_')]}")

            
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()

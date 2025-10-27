"""
Simple test script to check MCP connection and debug environment variables.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add the parent directory to the Python path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# Load environment variables
load_dotenv('.env')

def check_environment():
    """Check if all required environment variables are set."""
    print("=== Environment Variable Check ===")
    
    # Check GitHub token
    token1 = os.getenv('GITHUB_PERSONAL_ACCESS_TOKEN')
    token2 = os.getenv('GITHUB_TOKEN')
    
    print(f"GITHUB_PERSONAL_ACCESS_TOKEN: {'Set' if token1 else 'Not set'}")
    if token1:
        print(f"Token starts with: {token1[:10]}...")
    
    print(f"GITHUB_TOKEN: {'Set' if token2 else 'Not set'}")
    if token2:
        print(f"Token starts with: {token2[:10]}...")
    
    # Check other variables
    print(f"GITHUB_OWNER: {os.getenv('GITHUB_OWNER', 'Not set')}")
    print(f"GITHUB_REPO: {os.getenv('GITHUB_REPO', 'Not set')}")
    print(f"GITHUB_BRANCH: {os.getenv('GITHUB_BRANCH', 'Not set')}")
    
    return token1 or token2

async def test_basic_mcp():
    """Test basic MCP connection with debug info."""
    
    if not check_environment():
        print("\nError: No GitHub token found. Please check your .env file.")
        return
    
    try:
        from connectors.mcp_github_connector import create_remote_config, create_mcp_client
        
        print("\n=== Testing MCP Connection ===")
        
        # Get token (remove quotes if present)
        token = os.getenv('GITHUB_PERSONAL_ACCESS_TOKEN') or os.getenv('GITHUB_TOKEN')
        token = token.strip('"\'') if token else None
        
        if not token:
            print("No valid token found")
            return
            
        print(f"Using token: {token[:10]}...")
        
        # Create config
        config = create_remote_config(
            github_token=token,
            toolsets=["repos"],  # Start with just repos
            read_only=True
        )
        
        print(f"Config created: {config.connection_type} mode")
        
        # Try to connect
        client = await create_mcp_client(config)
        print("✅ Successfully connected to MCP server!")
        
        # Test basic functionality
        print("\n=== Testing Basic Functionality ===")
        
        # List available tools
        tools = await client.list_available_tools()
        print(f"Available tools: {len(tools)}")
        for tool in tools[:5]:  # Show first 5
            print(f"  - {tool.get('name', 'Unknown')}")
        
        # Get your repository info
        owner = os.getenv('GITHUB_OWNER', 'framsouza')
        repo = os.getenv('GITHUB_REPO', 'vault-storage-migration-on-k8s')
        
        print(f"\n=== Testing Repository Access: {owner}/{repo} ===")
        
        try:
            repo_info = await client.get_repository_info(owner, repo)
            print("✅ Repository info retrieved successfully!")
            
            # Print some basic info if available
            if 'content' in repo_info:
                content = repo_info['content']
                print(f"Repository name: {content.get('name', 'Unknown')}")
                print(f"Description: {content.get('description', 'No description')}")
                print(f"Stars: {content.get('stargazers_count', 0)}")
                print(f"Language: {content.get('language', 'Unknown')}")
        except Exception as e:
            print(f"❌ Failed to get repository info: {e}")
        
        await client.disconnect()
        print("\n✅ Test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during MCP test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_basic_mcp())

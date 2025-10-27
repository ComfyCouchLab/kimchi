"""
Quick test using local Docker GitHub MCP server.
"""

import os
import sys
import asyncio
from pathlib import Path
from dotenv import load_dotenv

# Add the parent directory to the Python path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# Load environment variables
load_dotenv('.env')

async def test_local_docker_mcp():
    """Test local Docker MCP server connection."""
    
    print("=== Testing Local Docker GitHub MCP Server ===")
    
    # Check if we have a GitHub token
    token = os.getenv('GITHUB_PERSONAL_ACCESS_TOKEN') or os.getenv('GITHUB_TOKEN')
    if not token:
        print("Error: No GitHub token found")
        return
    
    # Remove quotes if present
    token = token.strip('"\'')
    print(f"Using token: {token[:10]}...")
    
    try:
        from connectors.mcp_github_connector import create_local_config, create_mcp_client
        
        # Create local Docker config
        config = create_local_config(
            github_token=token,
            toolsets=["repos", "issues"],  # Start with basic toolsets
            read_only=True,
            use_docker=True
        )
        
        print("Starting Docker MCP server...")
        client = await create_mcp_client(config)
        print("✅ Successfully connected to local MCP server!")
        
        # Test basic functionality
        print("\n=== Testing Tools ===")
        tools = await client.list_available_tools()
        print(f"Available tools: {len(tools)}")
        
        for tool in tools[:10]:  # Show first 10
            name = tool.get('name', 'Unknown')
            description = tool.get('description', 'No description')
            print(f"  - {name}: {description[:60]}...")
        
        # Test repository access
        owner = os.getenv('GITHUB_OWNER', 'framsouza')
        repo = os.getenv('GITHUB_REPO', 'vault-storage-migration-on-k8s')
        
        print(f"\n=== Testing Repository: {owner}/{repo} ===")
        
        try:
            repo_info = await client.get_repository_info(owner, repo)
            print("✅ Repository info retrieved!")
            
            if 'content' in repo_info:
                content = repo_info['content']
                print(f"Name: {content.get('name', 'Unknown')}")
                print(f"Description: {content.get('description', 'No description')}")
                print(f"Stars: {content.get('stargazers_count', 0)}")
        except Exception as e:
            print(f"❌ Repository info failed: {e}")
        
        # Test file listing
        try:
            print(f"\n=== Testing File Listing ===")
            files = await client.list_repository_files(owner, repo, path="")
            print("✅ File listing retrieved!")
            
            if 'content' in files:
                for file in files['content'][:5]:  # Show first 5
                    name = file.get('name', 'Unknown')
                    file_type = file.get('type', 'unknown')
                    print(f"  - {name} ({file_type})")
        except Exception as e:
            print(f"❌ File listing failed: {e}")
        
        await client.disconnect()
        print("\n✅ Test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_local_docker_mcp())

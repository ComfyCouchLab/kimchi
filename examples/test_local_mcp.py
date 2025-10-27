#!/usr/bin/env python3
"""
Test the local GitHub MCP server with Docker.
"""

import os
import sys
import asyncio
from pathlib import Path

# Add the parent directory to Python path so we can import connectors
sys.path.insert(0, str(Path(__file__).parent.parent))

from connectors.mcp_github_connector import (
    create_local_config,
    create_mcp_client,
    MCPGitHubConnectorError
)
from dotenv import load_dotenv

async def test_local_mcp():
    """Test local Docker MCP server."""
    
    print("=== Local Docker MCP Server Test ===")
    
    # Load environment variables
    load_dotenv()
    
    # Check token
    github_token = os.getenv('GITHUB_PERSONAL_ACCESS_TOKEN')
    if not github_token:
        print("❌ Error: GITHUB_PERSONAL_ACCESS_TOKEN not found in environment")
        return
    
    print(f"✅ GitHub token found: {github_token[:10]}...")
    
    try:
        # Create local config for Docker
        config = create_local_config(
            github_token=github_token,
            toolsets=["repos", "issues"],  # Start with basic toolsets
            read_only=True,  # Safe mode
            use_docker=True
        )
        
        print("📦 Starting local MCP server with Docker...")
        print("This will pull the GitHub MCP server image if not already available")
        
        # Create and connect to MCP client
        client = await create_mcp_client(config)
        
        print("✅ Successfully connected to local MCP server!")
        
        # Test basic functionality
        print("\n=== Testing Available Tools ===")
        tools = await client.list_available_tools()
        print(f"Available tools: {len(tools)}")
        
        for i, tool in enumerate(tools[:5]):  # Show first 5 tools
            name = tool.get('name', 'Unknown')
            desc = tool.get('description', 'No description')
            print(f"{i+1}. {name}: {desc[:60]}...")
        
        # Test repository access
        owner = os.getenv('GITHUB_OWNER', 'framsouza')
        repo = os.getenv('GITHUB_REPO', 'vault-storage-migration-on-k8s')
        
        print(f"\n=== Testing Repository Access: {owner}/{repo} ===")
        
        try:
            repo_info = await client.get_repository_info(owner, repo)
            print("✅ Repository info retrieved successfully")
            
            # Try to get file contents
            files = await client.list_repository_files(owner, repo, path="")
            print(f"✅ Found {len(files.get('content', []))} files in root directory")
            
            # Try to get recent commits
            commits = await client.get_recent_commits(owner, repo, per_page=3)
            print(f"✅ Retrieved recent commits: {len(commits.get('content', []))}")
            
        except Exception as e:
            print(f"⚠️  Repository access error: {e}")
        
        # Clean up
        await client.disconnect()
        print("\n✅ Test completed successfully!")
        
    except MCPGitHubConnectorError as e:
        print(f"❌ MCP Error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

async def test_docker_availability():
    """Test if Docker is working properly."""
    
    print("=== Docker Availability Test ===")
    
    import subprocess
    
    try:
        # Test docker command
        result = subprocess.run(['docker', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"✅ Docker available: {result.stdout.strip()}")
        else:
            print(f"❌ Docker command failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Docker not available: {e}")
        return False
    
    try:
        # Test docker run with a simple command
        print("🧪 Testing Docker run capability...")
        result = subprocess.run(['docker', 'run', '--rm', 'hello-world'], 
                              capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print("✅ Docker run test successful")
            return True
        else:
            print(f"❌ Docker run failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Docker run test failed: {e}")
        return False

async def main():
    """Main test function."""
    
    # First test Docker
    docker_ok = await test_docker_availability()
    
    if docker_ok:
        print("\n" + "="*50)
        # Then test MCP
        await test_local_mcp()
    else:
        print("\n❌ Docker is not working properly. Please check Docker installation.")
        print("Make sure Docker Desktop is running.")

if __name__ == "__main__":
    asyncio.run(main())

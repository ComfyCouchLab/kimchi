"""
Example usage of MCPGitHubClient connector with GitHub's official MCP server.

This script demonstrates how to use the MCP GitHub client to interact
with a GitHub repository through the official GitHub MCP protocol.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the parent directory to the Python path so we can import connectors
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from connectors.mcp_github_connector import (
    MCPGitHubClient, 
    MCPConfig, 
    create_mcp_client,
    create_remote_config,
    create_local_config
)


async def test_remote_mcp_client():
    """Test the remote GitHub MCP client functionality."""
    
    print("=== Testing Remote GitHub MCP Server ===")
    
    # Example usage with remote server (hosted by GitHub)
    # Make sure you have this set in your .env file:
    # GITHUB_PERSONAL_ACCESS_TOKEN=your_github_token_here
    
    try:
        # Create remote config
        github_token = os.getenv('GITHUB_PERSONAL_ACCESS_TOKEN') or os.getenv('GITHUB_TOKEN')
        if not github_token:
            print("Error: GITHUB_PERSONAL_ACCESS_TOKEN or GITHUB_TOKEN environment variable required")
            return
        
        config = create_remote_config(
            github_token=github_token,
            toolsets=["repos", "issues", "pull_requests"],  # Specify what tools you want
            read_only=True  # Start with read-only for safety
        )
        
        # Create and connect to MCP client
        print("Connecting to remote MCP server...")
        client = await create_mcp_client(config)
        
        # List available tools
        print("\n=== Available MCP Tools ===")
        tools = await client.list_available_tools()
        for tool in tools:
            print(f"- {tool.get('name', 'Unknown')}: {tool.get('description', 'No description')[:100]}...")
        
        # Example repository (you can change these)
        owner = os.getenv('GITHUB_OWNER', 'microsoft')
        repo = os.getenv('GITHUB_REPO', 'vscode')
        
        print(f"\n=== Repository Info for {owner}/{repo} ===")
        
        # Get repository information
        try:
            repo_info = await client.get_repository_info(owner, repo)
            print(f"Repository: {repo_info.get('content', {}).get('name', 'Unknown')}")
            print(f"Description: {repo_info.get('content', {}).get('description', 'No description')}")
        except Exception as e:
            print(f"Could not get repository info: {e}")
        
        # Get recent commits
        try:
            print(f"\n=== Recent Commits ===")
            commits = await client.get_recent_commits(owner, repo, per_page=5)
            if 'content' in commits:
                for commit in commits['content'][:3]:  # Show first 3
                    message = commit.get('commit', {}).get('message', 'No message')
                    author = commit.get('commit', {}).get('author', {}).get('name', 'Unknown')
                    sha = commit.get('sha', 'Unknown')[:8]
                    print(f"- [{sha}] {message.split(chr(10))[0][:60]}... by {author}")
        except Exception as e:
            print(f"Could not get commits: {e}")
        
        # Get open issues
        try:
            print(f"\n=== Open Issues ===")
            issues = await client.get_repository_issues(owner, repo, per_page=3)
            if 'content' in issues:
                for issue in issues['content']:
                    title = issue.get('title', 'No title')
                    number = issue.get('number', 'N/A')
                    state = issue.get('state', 'unknown')
                    print(f"- #{number} [{state}]: {title}")
        except Exception as e:
            print(f"Could not get issues: {e}")
        
        # Get repository files (root directory)
        try:
            print(f"\n=== Root Directory Files ===")
            files = await client.list_repository_files(owner, repo, path="")
            if 'content' in files:
                for file in files['content'][:10]:  # Show first 10
                    name = file.get('name', 'Unknown')
                    file_type = file.get('type', 'file')
                    size = file.get('size', 0)
                    print(f"- {name} ({file_type}, {size} bytes)")
        except Exception as e:
            print(f"Could not list files: {e}")
        
        # Example: Get content of a specific file (README.md)
        try:
            print(f"\n=== README.md Content (first 200 chars) ===")
            readme = await client.get_file_content(owner, repo, "README.md")
            if 'content' in readme:
                # The content might be base64 encoded
                import base64
                content = readme['content'].get('content', '')
                if readme['content'].get('encoding') == 'base64':
                    content = base64.b64decode(content).decode('utf-8')
                print(f"{content[:200]}...")
        except Exception as e:
            print(f"Could not get README.md: {e}")
        
        # Disconnect
        await client.disconnect()
        print("\n=== Disconnected from remote MCP server ===")
        
    except Exception as e:
        print(f"Error: {e}")


async def test_local_mcp_client():
    """Test the local GitHub MCP client functionality."""
    
    print("\n=== Testing Local GitHub MCP Server ===")
    
    try:
        github_token = os.getenv('GITHUB_PERSONAL_ACCESS_TOKEN') or os.getenv('GITHUB_TOKEN')
        if not github_token:
            print("Error: GITHUB_PERSONAL_ACCESS_TOKEN or GITHUB_TOKEN environment variable required")
            return
        
        config = create_local_config(
            github_token=github_token,
            toolsets=["repos", "issues"],
            read_only=True,
            use_docker=True  # Set to False if you have the binary installed
        )
        
        print("Starting local MCP server (Docker)...")
        client = await create_mcp_client(config)
        
        # List available tools
        print("\n=== Local Server Available Tools ===")
        tools = await client.list_available_tools()
        for tool in tools[:5]:  # Show first 5
            print(f"- {tool.get('name', 'Unknown')}")
        
        await client.disconnect()
        print("=== Disconnected from local MCP server ===")
        
    except Exception as e:
        print(f"Local server error (this is expected if Docker isn't available): {e}")


async def test_mcp_search():
    """Test MCP search functionality."""
    
    print("\n=== Testing MCP Search ===")
    
    try:
        github_token = os.getenv('GITHUB_PERSONAL_ACCESS_TOKEN') or os.getenv('GITHUB_TOKEN')
        if not github_token:
            print("Error: GitHub token required for search")
            return
        
        config = create_remote_config(github_token=github_token, read_only=True)
        client = await create_mcp_client(config)
        
        owner = os.getenv('GITHUB_OWNER', 'microsoft')
        repo = os.getenv('GITHUB_REPO', 'vscode')
        
        print(f"=== Searching for 'authentication' in {owner}/{repo} ===")
        
        # Search for code containing 'authentication'
        search_results = await client.search_repository_code_in_repo(
            owner, repo, "authentication", per_page=3
        )
        
        if 'content' in search_results and 'items' in search_results['content']:
            for item in search_results['content']['items']:
                file_path = item.get('path', 'Unknown path')
                score = item.get('score', 0)
                print(f"- {file_path} (score: {score})")
        else:
            print("No search results found")
        
        await client.disconnect()
        
    except Exception as e:
        print(f"Search test error: {e}")


def main():
    """Main function to run MCP tests."""
    print("=== GitHub Official MCP Client Test ===")
    print("This tests both remote (GitHub-hosted) and local (Docker) MCP servers")
    
    # Test remote functionality (most common use case)
    asyncio.run(test_remote_mcp_client())
    
    # Test local functionality (if Docker is available)
    asyncio.run(test_local_mcp_client())
    
    # Test search functionality
    asyncio.run(test_mcp_search())


if __name__ == "__main__":
    main()

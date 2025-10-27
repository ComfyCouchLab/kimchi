"""
Discover actual tool names from the GitHub MCP server.
"""

import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the parent directory to the Python path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from connectors.mcp_github_connector import create_local_config, create_mcp_client


async def discover_tools():
    """Discover all available tools and their parameters."""
    
    try:
        github_token = os.getenv('GITHUB_PERSONAL_ACCESS_TOKEN') or os.getenv('GITHUB_TOKEN')
        if not github_token:
            print("Error: GitHub token required")
            return
        
        config = create_local_config(
            github_token=github_token,
            toolsets=["repos", "issues", "pull_requests"],
            read_only=True,
            use_docker=True
        )
        
        print("🔍 Discovering GitHub MCP Server Tools...")
        client = await create_mcp_client(config)
        
        # Get all available tools
        tools = await client.list_available_tools()
        
        print(f"\n📋 Found {len(tools)} tools:")
        print("=" * 80)
        
        # Group tools by category
        repo_tools = []
        issue_tools = []
        pr_tools = []
        other_tools = []
        
        for tool in tools:
            name = tool.get('name', 'Unknown')
            description = tool.get('description', 'No description')
            
            if 'repo' in name.lower() or 'file' in name.lower() or 'content' in name.lower():
                repo_tools.append((name, description))
            elif 'issue' in name.lower():
                issue_tools.append((name, description))
            elif 'pull' in name.lower() or 'pr' in name.lower():
                pr_tools.append((name, description))
            else:
                other_tools.append((name, description))
        
        # Print categorized tools
        print("\n🗂️  REPOSITORY TOOLS:")
        for name, desc in repo_tools:
            print(f"  • {name}: {desc[:80]}...")
        
        print("\n🐛 ISSUE TOOLS:")
        for name, desc in issue_tools:
            print(f"  • {name}: {desc[:80]}...")
        
        print("\n🔀 PULL REQUEST TOOLS:")
        for name, desc in pr_tools:
            print(f"  • {name}: {desc[:80]}...")
        
        print("\n🔧 OTHER TOOLS:")
        for name, desc in other_tools[:10]:  # Show first 10 others
            print(f"  • {name}: {desc[:80]}...")
        
        await client.disconnect()
        
        # Generate mapping for the connector
        print("\n" + "=" * 80)
        print("📝 Tool Mapping for Connector Update:")
        print("=" * 80)
        
        # Look for specific patterns
        for name, desc in repo_tools:
            if 'get' in name.lower() and 'repo' in name.lower():
                print(f"Repository info: {name}")
            elif 'file' in name.lower() and 'content' in name.lower():
                print(f"File content: {name}")
            elif 'list' in name.lower() and ('file' in name.lower() or 'content' in name.lower()):
                print(f"List files: {name}")
        
        for name, desc in other_tools:
            if 'commit' in name.lower():
                print(f"Commits: {name}")
            elif 'search' in name.lower():
                print(f"Search: {name}")
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(discover_tools())

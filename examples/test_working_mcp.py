"""
Test the updated MCP GitHub connector with correct tool names.
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


async def test_working_mcp():
    """Test the MCP connector with correct tool names."""
    
    try:
        github_token = os.getenv('GITHUB_PERSONAL_ACCESS_TOKEN') or os.getenv('GITHUB_TOKEN')
        owner = os.getenv('GITHUB_OWNER', 'framsouza')
        repo = os.getenv('GITHUB_REPO', 'vault-storage-migration-on-k8s')
        
        print(f"🧪 Testing MCP connector with {owner}/{repo}")
        
        config = create_local_config(
            github_token=github_token,
            toolsets=["repos", "issues", "pull_requests"],
            read_only=True,
            use_docker=True
        )
        
        client = await create_mcp_client(config)
        print("✅ Connected to MCP server")
        
        # Test 1: Get file content
        print(f"\n📄 Testing file content retrieval...")
        try:
            readme = await client.get_file_content(owner, repo, "README.md")
            if 'content' in readme:
                content = readme['content']
                if isinstance(content, str):
                    print(f"✅ README.md content: {content[:100]}...")
                else:
                    print(f"✅ README.md metadata: {type(content)}")
            else:
                print(f"📄 README.md response: {readme}")
        except Exception as e:
            print(f"❌ File content error: {e}")
        
        # Test 2: List repository files
        print(f"\n📁 Testing repository file listing...")
        try:
            files = await client.list_repository_files(owner, repo, path="")
            if 'content' in files:
                content = files['content']
                if isinstance(content, list):
                    print(f"✅ Found {len(content)} files in root directory")
                    for file in content[:3]:
                        print(f"  - {file.get('name', 'Unknown')}")
                else:
                    print(f"✅ Files response type: {type(content)}")
            else:
                print(f"📁 Files response: {files}")
        except Exception as e:
            print(f"❌ File listing error: {e}")
        
        # Test 3: Get commits
        print(f"\n📈 Testing commit retrieval...")
        try:
            commits = await client.get_recent_commits(owner, repo, per_page=3)
            if 'content' in commits:
                content = commits['content']
                if isinstance(content, list):
                    print(f"✅ Found {len(content)} recent commits")
                    for commit in content:
                        message = commit.get('commit', {}).get('message', 'No message')
                        sha = commit.get('sha', 'Unknown')[:8]
                        print(f"  - [{sha}] {message.split(chr(10))[0][:60]}...")
                else:
                    print(f"✅ Commits response type: {type(content)}")
            else:
                print(f"📈 Commits response: {commits}")
        except Exception as e:
            print(f"❌ Commits error: {e}")
        
        # Test 4: Get issues
        print(f"\n🐛 Testing issue retrieval...")
        try:
            issues = await client.get_repository_issues(owner, repo, per_page=3)
            if 'content' in issues:
                content = issues['content']
                if isinstance(content, list):
                    print(f"✅ Found {len(content)} issues")
                    for issue in content:
                        title = issue.get('title', 'No title')
                        number = issue.get('number', 'N/A')
                        state = issue.get('state', 'unknown')
                        print(f"  - #{number} [{state}]: {title}")
                else:
                    print(f"✅ Issues response type: {type(content)}")
            else:
                print(f"🐛 Issues response: {issues}")
        except Exception as e:
            print(f"❌ Issues error: {e}")
        
        # Test 5: Get current user
        print(f"\n👤 Testing user info...")
        try:
            user = await client.get_current_user()
            if 'content' in user:
                content = user['content']
                username = content.get('login', 'Unknown')
                name = content.get('name', 'Unknown')
                print(f"✅ Authenticated as: {username} ({name})")
            else:
                print(f"👤 User response: {user}")
        except Exception as e:
            print(f"❌ User info error: {e}")
        
        await client.disconnect()
        print("\n🎉 MCP connector test completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")


if __name__ == "__main__":
    asyncio.run(test_working_mcp())

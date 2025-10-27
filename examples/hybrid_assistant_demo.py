"""
Working example of hybrid RAG + MCP system.

This demonstrates how to use both your existing RAG system and the MCP connector
for a comprehensive GitHub assistant.
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
from connectors.github_connector import GitHubConnector


class HybridGitHubAssistant:
    """
    Hybrid assistant that combines RAG (static knowledge) with MCP (live data).
    """
    
    def __init__(self):
        self.rag_connector = None
        self.mcp_client = None
        self.is_mcp_available = False
    
    async def initialize(self):
        """Initialize both RAG and MCP connections."""
        
        print("🚀 Initializing Hybrid GitHub Assistant...")
        
        # Initialize RAG connector
        try:
            self.rag_connector = GitHubConnector()
            print("✅ RAG connector ready")
        except Exception as e:
            print(f"⚠️  RAG connector error: {e}")
        
        # Initialize MCP connector
        try:
            github_token = os.getenv('GITHUB_PERSONAL_ACCESS_TOKEN') or os.getenv('GITHUB_TOKEN')
            if github_token:
                config = create_local_config(
                    github_token=github_token,
                    toolsets=["repos", "issues", "pull_requests"],
                    read_only=True,
                    use_docker=True
                )
                self.mcp_client = await create_mcp_client(config)
                self.is_mcp_available = True
                print("✅ MCP connector ready")
            else:
                print("⚠️  No GitHub token - MCP disabled")
        except Exception as e:
            print(f"⚠️  MCP connector error: {e}")
            self.is_mcp_available = False
    
    def classify_query(self, query: str) -> str:
        """
        Classify query to determine routing strategy.
        
        Returns:
            'live_data' - for real-time information
            'knowledge' - for documentation/best practices
            'hybrid' - for both
        """
        query_lower = query.lower()
        
        # Live data indicators
        live_keywords = [
            'recent', 'latest', 'current', 'now', 'today', 'last week',
            'open issues', 'current state', 'live', 'active', 'new'
        ]
        
        # Knowledge indicators  
        knowledge_keywords = [
            'how to', 'best practice', 'guide', 'tutorial', 'example',
            'documentation', 'setup', 'configure', 'install', 'deploy'
        ]
        
        has_live = any(keyword in query_lower for keyword in live_keywords)
        has_knowledge = any(keyword in query_lower for keyword in knowledge_keywords)
        
        if has_live and not has_knowledge:
            return 'live_data'
        elif has_knowledge and not has_live:
            return 'knowledge'
        else:
            return 'hybrid'
    
    async def get_live_data(self, query: str, owner: str, repo: str) -> dict:
        """Get real-time data using MCP."""
        
        if not self.is_mcp_available:
            return {"error": "MCP not available", "data": None}
        
        try:
            query_lower = query.lower()
            
            if 'commit' in query_lower or 'recent changes' in query_lower:
                result = await self.mcp_client.get_recent_commits(owner, repo, per_page=5)
                return {"type": "commits", "data": result}
            
            elif 'issue' in query_lower:
                result = await self.mcp_client.get_repository_issues(owner, repo, per_page=10)
                return {"type": "issues", "data": result}
            
            elif 'file' in query_lower and 'content' in query_lower:
                # Extract filename from query (simplified)
                if 'readme' in query_lower:
                    result = await self.mcp_client.get_file_content(owner, repo, "README.md")
                    return {"type": "file_content", "data": result}
                else:
                    result = await self.mcp_client.list_repository_files(owner, repo)
                    return {"type": "file_list", "data": result}
            
            elif 'user' in query_lower or 'who am i' in query_lower:
                result = await self.mcp_client.get_current_user()
                return {"type": "user_info", "data": result}
            
            else:
                # Default to repository files
                result = await self.mcp_client.list_repository_files(owner, repo)
                return {"type": "general", "data": result}
                
        except Exception as e:
            return {"error": str(e), "data": None}
    
    def get_knowledge_data(self, query: str) -> dict:
        """Get static knowledge using RAG."""
        
        if not self.rag_connector:
            return {"error": "RAG not available", "data": None}
        
        try:
            # This would integrate with your existing RAG system
            # For now, return mock data
            return {
                "type": "knowledge",
                "data": f"RAG knowledge response for: {query}",
                "source": "documentation/runbooks"
            }
        except Exception as e:
            return {"error": str(e), "data": None}
    
    async def answer_question(self, query: str, owner: str = None, repo: str = None) -> dict:
        """
        Main method to answer questions using hybrid approach.
        """
        
        # Use defaults from environment if not provided
        owner = owner or os.getenv('GITHUB_OWNER', 'framsouza')
        repo = repo or os.getenv('GITHUB_REPO', 'vault-storage-migration-on-k8s')
        
        print(f"🤔 Question: {query}")
        print(f"📍 Target: {owner}/{repo}")
        
        # Classify the query
        classification = self.classify_query(query)
        print(f"🧠 Classification: {classification}")
        
        response = {
            "query": query,
            "classification": classification,
            "live_data": None,
            "knowledge_data": None,
            "answer": None
        }
        
        # Route based on classification
        if classification == 'live_data':
            response["live_data"] = await self.get_live_data(query, owner, repo)
            response["answer"] = f"Live data: {response['live_data']}"
        
        elif classification == 'knowledge':
            response["knowledge_data"] = self.get_knowledge_data(query)
            response["answer"] = f"Knowledge: {response['knowledge_data']}"
        
        else:  # hybrid
            response["live_data"] = await self.get_live_data(query, owner, repo)
            response["knowledge_data"] = self.get_knowledge_data(query)
            response["answer"] = f"Hybrid: Live={response['live_data']}, Knowledge={response['knowledge_data']}"
        
        return response
    
    async def cleanup(self):
        """Clean up connections."""
        if self.mcp_client:
            await self.mcp_client.disconnect()


async def demo():
    """Demonstrate the hybrid assistant."""
    
    assistant = HybridGitHubAssistant()
    await assistant.initialize()
    
    # Test different types of queries
    test_queries = [
        "What are the recent commits?",  # live_data
        "How do I setup CI/CD?",  # knowledge  
        "Show me current issues and best practices for issue management",  # hybrid
        "What files are in the repository?",  # live_data
        "Who am I?",  # live_data
    ]
    
    print("\n" + "="*60)
    print("🎭 HYBRID ASSISTANT DEMO")
    print("="*60)
    
    for query in test_queries:
        print(f"\n{'='*20}")
        response = await assistant.answer_question(query)
        print(f"💬 Response: {response['classification']}")
        
        if response['live_data']:
            print(f"📊 Live Data: {response['live_data'].get('type', 'unknown')}")
        
        if response['knowledge_data']:
            print(f"📚 Knowledge: {response['knowledge_data'].get('type', 'unknown')}")
    
    await assistant.cleanup()
    print("\n✅ Demo completed!")


if __name__ == "__main__":
    asyncio.run(demo())

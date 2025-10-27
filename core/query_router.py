"""
QueryRouter - Intelligent routing between RAG and MCP systems.

This module uses AI to analyze user queries and route them to the appropriate
data source (RAG for knowledge, MCP for live data, or both for hybrid responses).
"""

import os
import asyncio
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class QueryType(Enum):
    """Types of queries for routing decisions."""
    LIVE_DATA = "live_data"      # Real-time GitHub data via MCP
    KNOWLEDGE = "knowledge"      # Static documentation via RAG
    HYBRID = "hybrid"           # Both sources needed
    GENERAL = "general"         # General questions


@dataclass
class RoutingDecision:
    """Result of query routing analysis."""
    query_type: QueryType
    confidence: float
    reasoning: str
    mcp_tools: List[str]        # Which MCP tools to use
    rag_topics: List[str]       # Which RAG topics to search
    priority: str               # Which source to prioritize if hybrid


class QueryRouter:
    """
    Intelligent router that analyzes queries and determines optimal data sources.
    
    Uses OpenAI to understand query intent and route to:
    - MCP: For live GitHub data (commits, issues, files, etc.)
    - RAG: For documentation, tutorials, best practices
    - Hybrid: For complex queries needing both
    """
    
    def __init__(self, openai_api_key: Optional[str] = None):
        """
        Initialize the QueryRouter.
        
        Args:
            openai_api_key: OpenAI API key (or loads from environment)
        """
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        if not self.openai_api_key:
            raise ValueError("OpenAI API key required for QueryRouter")
        
        openai.api_key = self.openai_api_key
        
        # Define routing patterns and examples
        self.routing_examples = self._get_routing_examples()
        self.mcp_tool_mapping = self._get_mcp_tool_mapping()
    
    def _get_routing_examples(self) -> Dict[str, List[str]]:
        """Get example queries for each routing type."""
        return {
            "live_data": [
                "What are the recent commits?",
                "Show me open issues",
                "Who are the contributors?",
                "What's the current state of pull requests?",
                "When was the last commit?",
                "List the latest releases",
                "What branches exist?",
                "Are there any security issues?",
                "Show me recent pull request activity",
                "What are the latest commit messages?"
            ],
            "knowledge": [
                "What's this repo about?",
                "What does this repository do?",
                "Explain the project architecture",
                "How do I set up CI/CD?",
                "What are the best practices for testing?",
                "How to configure deployment?",
                "Explain the architecture pattern",
                "What is the recommended workflow?",
                "How to handle error logging?",
                "Best practices for security",
                "How to optimize performance?",
                "Deployment strategies explained",
                "Code review guidelines",
                "Show me the README file content",
                "What is this project for?",
                "How does this system work?",
                "What are the main features?",
                "Describe the project structure"
            ],
            "hybrid": [
                "Analyze recent commits and suggest improvements",
                "Review current issues and provide solutions",
                "Check repository health and recommend best practices",
                "Compare current code with security guidelines",
                "Audit recent changes against standards",
                "Show current state and improvement recommendations",
                "Analyze workflow and suggest optimizations",
                "Review repository structure and best practices"
            ]
        }
    
    def _get_mcp_tool_mapping(self) -> Dict[str, List[str]]:
        """Map query intents to specific MCP tools."""
        return {
            "commits": ["list_commits", "get_commit"],
            "issues": ["list_issues", "issue_read", "search_issues"],
            "pull_requests": ["list_pull_requests", "pull_request_read"],
            "files": ["get_file_contents", "list_repository_files"],
            "repository": ["get_repository", "search_repositories"],
            "user": ["get_me", "get_contributors"],
            "releases": ["get_latest_release", "get_release_by_tag"],
            "branches": ["list_branches", "create_branch"],
            "search": ["search_code", "search_issues", "search_repositories"]
        }
    
    async def analyze_query(self, query: str, context: Optional[Dict] = None) -> RoutingDecision:
        """
        Analyze a query and determine the optimal routing strategy.
        
        Args:
            query: User's question or request
            context: Optional context (repository info, user history, etc.)
            
        Returns:
            RoutingDecision with routing strategy and confidence
        """
        try:
            # Create prompt for OpenAI analysis
            prompt = self._create_routing_prompt(query, context)
            
            # Get AI analysis
            response = await self._get_openai_analysis(prompt)
            
            # Parse the response
            decision = self._parse_ai_response(response, query)
            
            return decision
            
        except Exception as e:
            # Fallback to rule-based routing if AI fails
            return self._fallback_routing(query)
    
    def _create_routing_prompt(self, query: str, context: Optional[Dict] = None) -> str:
        """Create a detailed prompt for OpenAI query analysis."""
        
        context_str = ""
        if context:
            context_str = f"\nContext: {json.dumps(context, indent=2)}"
        
        return f"""
Analyze this GitHub-related query and determine the optimal data routing strategy.

Query: "{query}"{context_str}

Available Data Sources:
1. LIVE_DATA (MCP): Real-time GitHub API data only
   - Recent commits, current issues, live repository metadata
   - Active pull requests, repository statistics, user info  
   - Current branches, releases, contributors, timestamps

2. KNOWLEDGE (RAG): Repository content and documentation
   - README content, documentation files, code explanations
   - Project descriptions, architecture explanations, setup guides
   - File contents, project structure explanations, tutorials
   - Best practices, workflows, troubleshooting guides

3. HYBRID: Both sources needed
   - Analysis combining current state with documentation
   - Recommendations based on live data and repository content
   - Comparisons between current state and documented standards

IMPORTANT ROUTING RULES:
- If asking WHAT a repository/project is about → ALWAYS use KNOWLEDGE (RAG has README/docs)
- If asking about repository PURPOSE, DESCRIPTION, or FUNCTIONALITY → use KNOWLEDGE  
- If asking WHO contributed recently → use LIVE_DATA (MCP has commit history)  
- If asking HOW to set something up → use KNOWLEDGE (RAG has documentation)
- If asking WHEN something happened → use LIVE_DATA (MCP has timestamps)
- If asking about RECENT/CURRENT state → use LIVE_DATA (MCP has live data)

Examples by Category:
KNOWLEDGE: "What's this repo about?", "What does this project do?", "Describe this repository", 
           "What are the features?", "How does this work?", "Show README content", "Project description"
LIVE_DATA: "Who committed recently?", "When was last commit?", "How many open issues?", 
           "List recent commits", "Show current pull requests", "What files are in the repo?"
HYBRID: "Analyze recent commits and suggest improvements", "Review issues and provide solutions"

Respond in JSON format:
{{
    "query_type": "LIVE_DATA|KNOWLEDGE|HYBRID",
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation of why this routing was chosen",
    "mcp_tools": ["tool1", "tool2"] or [],
    "rag_topics": ["topic1", "topic2"] or [],
    "priority": "mcp|rag|balanced"
}}

MCP Tools Available:
- commits: list_commits, get_commit
- issues: list_issues, issue_read, search_issues  
- files: get_file_contents, list_repository_files
- pull_requests: list_pull_requests, pull_request_read
- repository: get_repository, search_repositories
- user: get_me, get_contributors
- search: search_code, search_issues

RAG Topics Available:
- deployment, testing, security, architecture, workflows, best_practices, troubleshooting, setup, configuration

Analyze the query and provide routing decision:
"""
    
    async def _get_openai_analysis(self, prompt: str) -> str:
        """Get analysis from OpenAI."""
        try:
            client = openai.OpenAI(api_key=self.openai_api_key)
            
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing GitHub-related queries and routing them to optimal data sources. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.1
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            raise Exception(f"OpenAI analysis failed: {e}")
    
    def _parse_ai_response(self, response: str, original_query: str) -> RoutingDecision:
        """Parse OpenAI response into RoutingDecision."""
        try:
            # Clean the response (remove code blocks if present)
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].strip()
            
            data = json.loads(response)
            
            return RoutingDecision(
                query_type=QueryType(data.get("query_type", "general").lower()),
                confidence=float(data.get("confidence", 0.5)),
                reasoning=data.get("reasoning", "AI analysis"),
                mcp_tools=data.get("mcp_tools", []),
                rag_topics=data.get("rag_topics", []),
                priority=data.get("priority", "balanced")
            )
            
        except Exception as e:
            # Fallback parsing
            return self._fallback_routing(original_query)
    
    def _fallback_routing(self, query: str) -> RoutingDecision:
        """Fallback rule-based routing if AI analysis fails."""
        query_lower = query.lower()
        
        # Specific repository content patterns (high priority for knowledge)
        repo_content_patterns = [
            'what\'s this repo about',
            'what is this repo about', 
            'what does this repo do',
            'what does this repository do',
            'what is this project',
            'describe this project',
            'tell me about this repo',
            'explain this repository',
            'project description',
            'repository description'
        ]
        
        # Check for direct repository content questions first
        for pattern in repo_content_patterns:
            if pattern in query_lower:
                return RoutingDecision(
                    query_type=QueryType.KNOWLEDGE,
                    confidence=0.9,
                    reasoning="Direct repository content/description question - routed to RAG",
                    mcp_tools=[],
                    rag_topics=["architecture", "setup", "best_practices"],
                    priority="rag"
                )
        
        # Live data keywords (more specific matching)
        live_keywords = ['recent commits', 'latest commits', 'current issues', 'open issues', 'new issues', 'active pull', 'who committed', 'when was', 'list files', 'show files']
        
        # Knowledge keywords  
        knowledge_keywords = ['how to', 'setup', 'configure', 'best practice', 'guide', 'tutorial', 'deploy', 'install', 'architecture', 'pattern', 'what is', 'what does', 'explain', 'describe']
        
        # Hybrid keywords
        hybrid_keywords = ['analyze', 'review', 'recommend', 'suggest', 'improve', 'audit', 'compare', 'optimize']
        
        live_score = sum(1 for kw in live_keywords if kw in query_lower)
        knowledge_score = sum(1 for kw in knowledge_keywords if kw in query_lower)
        hybrid_score = sum(1 for kw in hybrid_keywords if kw in query_lower)
        
        if hybrid_score > 0:
            query_type = QueryType.HYBRID
            confidence = 0.7
        elif knowledge_score > live_score:
            query_type = QueryType.KNOWLEDGE
            confidence = 0.7
        elif live_score > knowledge_score:
            query_type = QueryType.LIVE_DATA
            confidence = 0.6
        else:
            query_type = QueryType.KNOWLEDGE  # Default to knowledge for ambiguous cases
            confidence = 0.5
        
        return RoutingDecision(
            query_type=query_type,
            confidence=confidence,
            reasoning="Fallback rule-based routing",
            mcp_tools=self._extract_mcp_tools(query_lower),
            rag_topics=self._extract_rag_topics(query_lower),
            priority="balanced"
        )
    
    def _extract_mcp_tools(self, query_lower: str) -> List[str]:
        """Extract relevant MCP tools from query."""
        tools = []
        
        if any(word in query_lower for word in ['commit', 'commits']):
            tools.extend(["list_commits", "get_commit"])
        if any(word in query_lower for word in ['issue', 'issues']):
            tools.extend(["list_issues", "issue_read"])
        if any(word in query_lower for word in ['file', 'files', 'content']):
            tools.extend(["get_file_contents", "list_repository_files"])
        if any(word in query_lower for word in ['pull request', 'pr', 'merge']):
            tools.extend(["list_pull_requests", "pull_request_read"])
        if any(word in query_lower for word in ['repository', 'repo']):
            tools.extend(["get_repository"])
        if any(word in query_lower for word in ['user', 'who', 'contributor']):
            tools.extend(["get_me", "get_contributors"])
        if any(word in query_lower for word in ['search', 'find']):
            tools.extend(["search_code", "search_issues"])
        
        return list(set(tools))  # Remove duplicates
    
    def _extract_rag_topics(self, query_lower: str) -> List[str]:
        """Extract relevant RAG topics from query."""
        topics = []
        
        if any(word in query_lower for word in ['deploy', 'deployment']):
            topics.append("deployment")
        if any(word in query_lower for word in ['test', 'testing']):
            topics.append("testing")
        if any(word in query_lower for word in ['security', 'secure']):
            topics.append("security")
        if any(word in query_lower for word in ['architecture', 'design', 'pattern']):
            topics.append("architecture")
        if any(word in query_lower for word in ['workflow', 'process']):
            topics.append("workflows")
        if any(word in query_lower for word in ['best practice', 'guidelines']):
            topics.append("best_practices")
        if any(word in query_lower for word in ['setup', 'configure', 'install']):
            topics.append("setup")
        if any(word in query_lower for word in ['troubleshoot', 'debug', 'error']):
            topics.append("troubleshooting")
        
        return topics
    
    async def route_query(self, query: str, context: Optional[Dict] = None) -> RoutingDecision:
        """
        Main routing method - analyze and route a query.
        
        Args:
            query: User's question
            context: Optional context information
            
        Returns:
            RoutingDecision with complete routing strategy
        """
        decision = await self.analyze_query(query, context)
        
        # Log the routing decision
        print(f"🧠 Query: {query}")
        print(f"📍 Route: {decision.query_type.value} (confidence: {decision.confidence:.2f})")
        print(f"💭 Reasoning: {decision.reasoning}")
        if decision.mcp_tools:
            print(f"🔧 MCP Tools: {', '.join(decision.mcp_tools)}")
        if decision.rag_topics:
            print(f"📚 RAG Topics: {', '.join(decision.rag_topics)}")
        
        return decision
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get statistics about routing patterns."""
        return {
            "supported_query_types": [qt.value for qt in QueryType],
            "available_mcp_tools": sum(len(tools) for tools in self.mcp_tool_mapping.values()),
            "example_queries": len(sum(self.routing_examples.values(), [])),
            "mcp_categories": list(self.mcp_tool_mapping.keys())
        }


# Example usage and testing
async def test_query_router():
    """Test the QueryRouter with various query types."""
    
    router = QueryRouter()
    
    test_queries = [
        # Live data queries
        "What are the recent commits in this repository?",
        "Show me all open issues",
        "List the files in the root directory",
        "Who are the contributors to this project?",
        
        # Knowledge queries  
        "How do I set up continuous integration?",
        "What are the best practices for code reviews?",
        "How should I structure my deployment pipeline?",
        "Explain the microservices architecture pattern",
        
        # Hybrid queries
        "Analyze the recent commits and suggest code quality improvements",
        "Review the current issues and provide solution strategies",
        "Check the repository structure against best practices",
        "Examine recent changes and recommend security enhancements"
    ]
    
    print("🔄 Testing QueryRouter with various query types:")
    print("=" * 60)
    
    for query in test_queries:
        print(f"\n{'=' * 20}")
        decision = await router.route_query(query)
        print(f"Priority: {decision.priority}")
        print()


if __name__ == "__main__":
    asyncio.run(test_query_router())

"""
Kimchi GitHub Assistant - Core AI-powered hybrid assistant.

This assistant combines RAG and MCP systems with intelligent routing
using OpenAI to determine the optimal data source for each query.
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the parent directory to the Python path for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# Configure logging to suppress verbose third-party messages early
try:
    from utils.logging import configure_third_party_loggers
    configure_third_party_loggers()
except ImportError:
    # Fallback if utils not available
    import logging
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("elastic_transport").setLevel(logging.WARNING)
    logging.getLogger("elasticsearch").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

from .query_router import QueryRouter, QueryType, RoutingDecision
from connectors.mcp_github_connector import create_local_config, create_mcp_client
from connectors.elasticsearch_connector import ElasticsearchConnector, ElasticsearchConfig
from config import load_config
import openai


class KimchiAssistant:
    """
    AI-powered hybrid GitHub assistant with intelligent query routing.
    
    Features:
    - Smart routing between RAG and MCP using OpenAI
    - Context-aware responses
    - Fallback mechanisms
    - Response synthesis using AI
    """
    
    def __init__(self):
        self.query_router = None
        self.rag_connector = None
        self.mcp_client = None
        self.openai_client = None
        self.is_mcp_available = False
        self.is_rag_available = False
    
    async def initialize(self):
        """Initialize all components of the assistant."""
        
        print("Initializing Kimchi GitHub Assistant...")
        
        # Initialize QueryRouter
        try:
            self.query_router = QueryRouter()
            print("QueryRouter ready (AI-powered routing)")
        except Exception as e:
            print(f"QueryRouter error: {e}")
        
        # Initialize OpenAI client for response synthesis
        try:
            openai_key = os.getenv('OPENAI_API_KEY')
            if openai_key:
                self.openai_client = openai.OpenAI(api_key=openai_key)
                print("OpenAI client ready")
            else:
                print("No OpenAI API key - response synthesis disabled")
        except Exception as e:
            print(f"OpenAI client error: {e}")
        
        # Initialize RAG connector
        try:
            # Load configuration for Elasticsearch RAG
            config = load_config()
            self.rag_connector = ElasticsearchConnector(
                config.elasticsearch,
                config.embedding_model
            )
            await self.rag_connector.initialize()
            self.is_rag_available = True
            print("RAG connector ready")
        except Exception as e:
            print(f"RAG connector error: {e}")
            self.is_rag_available = False
        
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
                print("MCP connector ready")
            else:
                print("No GitHub token - MCP disabled")
        except Exception as e:
            print(f"MCP connector error: {e}")
            self.is_mcp_available = False
        
        print(f"Assistant ready! MCP: {self.is_mcp_available}, RAG: {self.is_rag_available}")
    
    async def answer_question(self, 
                            query: str, 
                            owner: str = None, 
                            repo: str = None,
                            context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Answer a question using intelligent routing and AI synthesis.
        
        Args:
            query: User's question
            owner: Repository owner (defaults from env)
            repo: Repository name (defaults from env)
            context: Optional context information
            
        Returns:
            Comprehensive response with routing info and synthesized answer
        """
        
        # Use defaults from environment
        owner = owner or os.getenv('GITHUB_OWNER', 'microsoft')
        repo = repo or os.getenv('GITHUB_REPO', 'vscode')
        
        # Create response structure
        response = {
            "query": query,
            "repository": f"{owner}/{repo}",
            "routing_decision": None,
            "live_data": None,
            "knowledge_data": None,
            "synthesized_answer": None,
            "sources_used": [],
            "confidence": 0.0
        }
        
        try:
            # Step 1: Route the query
            if self.query_router:
                routing_decision = await self.query_router.route_query(query, context)
                response["routing_decision"] = {
                    "type": routing_decision.query_type.value,
                    "confidence": routing_decision.confidence,
                    "reasoning": routing_decision.reasoning,
                    "mcp_tools": routing_decision.mcp_tools,
                    "rag_topics": routing_decision.rag_topics,
                    "priority": routing_decision.priority
                }
            else:
                # Fallback routing
                routing_decision = self._simple_routing(query)
                response["routing_decision"] = {
                    "type": routing_decision.query_type.value,
                    "confidence": 0.5,
                    "reasoning": "Simple fallback routing",
                    "mcp_tools": [],
                    "rag_topics": [],
                    "priority": "balanced"
                }
            
            # Step 2: Gather data based on routing decision
            if routing_decision.query_type in [QueryType.LIVE_DATA, QueryType.HYBRID]:
                response["live_data"] = await self._get_live_data(query, owner, repo, routing_decision)
                if response["live_data"]:
                    response["sources_used"].append("MCP")
            
            if routing_decision.query_type in [QueryType.KNOWLEDGE, QueryType.HYBRID]:
                response["knowledge_data"] = await self._get_knowledge_data(query, routing_decision)
                if response["knowledge_data"]:
                    response["sources_used"].append("RAG")
            
            # Step 3: Synthesize final answer using AI
            if self.openai_client:
                response["synthesized_answer"] = await self._synthesize_answer(response)
                response["confidence"] = routing_decision.confidence
            else:
                # Simple concatenation if no OpenAI
                response["synthesized_answer"] = self._simple_answer_synthesis(response)
                response["confidence"] = routing_decision.confidence * 0.7
            
            return response
            
        except Exception as e:
            response["synthesized_answer"] = f"Error processing query: {e}"
            response["confidence"] = 0.0
            return response
    
    # ...existing methods... (keeping all the existing logic for now)
    
    async def _get_live_data(self, query: str, owner: str, repo: str, routing: RoutingDecision) -> Optional[Dict]:
        """Get live data using MCP based on routing decision."""
        
        if not self.is_mcp_available:
            return {"error": "MCP not available"}
        
        try:
            live_data = {}
            query_lower = query.lower()
            
            # Use routing decision to determine which tools to call
            if not routing.mcp_tools:
                # Fallback to query analysis
                if any(word in query_lower for word in ['commit', 'commits']):
                    routing.mcp_tools = ["list_commits"]
                elif any(word in query_lower for word in ['issue', 'issues']):
                    routing.mcp_tools = ["list_issues"]
                elif any(word in query_lower for word in ['file', 'files']):
                    routing.mcp_tools = ["get_file_contents"]
                else:
                    routing.mcp_tools = ["get_file_contents"]  # Default
            
            # Execute MCP calls based on identified tools
            for tool in routing.mcp_tools:
                try:
                    if tool == "list_commits":
                        result = await self.mcp_client.get_recent_commits(owner, repo, per_page=5)
                        live_data["commits"] = result
                    
                    elif tool == "list_issues":
                        result = await self.mcp_client.get_repository_issues(owner, repo, per_page=10)
                        live_data["issues"] = result
                    
                    elif tool == "get_file_contents":
                        if 'readme' in query_lower:
                            result = await self.mcp_client.get_file_content(owner, repo, "README.md")
                            live_data["readme"] = result
                        else:
                            result = await self.mcp_client.list_repository_files(owner, repo)
                            live_data["files"] = result
                    
                    elif tool == "list_pull_requests":
                        result = await self.mcp_client.get_pull_requests(owner, repo, per_page=5)
                        live_data["pull_requests"] = result
                    
                    elif tool == "get_me":
                        result = await self.mcp_client.get_current_user()
                        live_data["user"] = result
                
                except Exception as e:
                    live_data[f"{tool}_error"] = str(e)
            
            return live_data if live_data else None
            
        except Exception as e:
            return {"error": f"MCP error: {e}"}
    
    async def _get_knowledge_data(self, query: str, routing: RoutingDecision) -> Optional[Dict]:
        """Get knowledge data using RAG based on routing decision."""
        
        if not self.is_rag_available:
            return {"error": "RAG not available"}
        
        try:
            knowledge_data = {}
            
            # Try to use actual RAG connector if available
            if hasattr(self.rag_connector, 'search_documents'):
                try:
                    # Enable detailed observability for debugging
                    enable_debug = os.getenv('ELASTICSEARCH_DEBUG', 'false').lower() == 'true'
                    
                    # Use routing topics as search terms
                    search_terms = routing.rag_topics if routing.rag_topics else [query]
                    
                    for term in search_terms:
                        results = self.rag_connector.search_documents(term, k=3, debug=enable_debug)
                        if results:
                            knowledge_data[term] = results
                    
                    # If no specific topics yielded results, do a general search
                    if not knowledge_data:
                        results = self.rag_connector.search_documents(query, k=5, debug=enable_debug)
                        if results:
                            knowledge_data["search_results"] = results
                
                except Exception as e:
                    print(f"RAG search error: {e}")
                    # Fall back to topic-based responses
                    knowledge_data = self._get_fallback_knowledge(routing.rag_topics, query)
            else:
                # Use fallback knowledge
                knowledge_data = self._get_fallback_knowledge(routing.rag_topics, query)
            
            return knowledge_data if knowledge_data else {"general": "No specific knowledge found for this query."}
            
        except Exception as e:
            return {"error": f"RAG error: {e}"}
    
    def _get_fallback_knowledge(self, topics: List[str], query: str) -> Dict[str, str]:
        """Get fallback knowledge responses when RAG is not fully available."""
        knowledge_data = {}
        
        for topic in topics:
            if topic == "deployment":
                knowledge_data["deployment"] = "Best practices for deployment include automated CI/CD, blue-green deployments, proper monitoring, health checks, and gradual rollouts."
            elif topic == "testing":
                knowledge_data["testing"] = "Testing best practices include unit tests, integration tests, code coverage, automated testing, test-driven development, and comprehensive test suites."
            elif topic == "security":
                knowledge_data["security"] = "Security best practices include dependency scanning, secret management, HTTPS enforcement, regular updates, access controls, and security audits."
            elif topic == "architecture":
                knowledge_data["architecture"] = "Architecture patterns include microservices, event-driven, layered architecture, clean architecture, and proper separation of concerns."
            elif topic == "ci/cd" or "ci" in topic.lower():
                knowledge_data["ci_cd"] = "CI/CD best practices include automated builds, testing pipelines, deployment automation, environment promotion, and monitoring."
            elif topic == "configuration":
                knowledge_data["configuration"] = "Configuration management includes environment-specific configs, secret management, configuration validation, and infrastructure as code."
            elif topic == "best_practices":
                knowledge_data["best_practices"] = "General best practices include code reviews, documentation, testing, monitoring, security, and continuous improvement."
            else:
                knowledge_data[topic] = f"Best practices and guidance for {topic} based on industry standards and documentation."
        
        # If no specific topics, provide query-specific knowledge
        if not knowledge_data:
            query_lower = query.lower()
            if any(word in query_lower for word in ['vault', 'hashicorp', 'secret']):
                knowledge_data["vault"] = "HashiCorp Vault best practices include proper unsealing, backup strategies, secret rotation, access policies, and storage backend configuration."
            elif any(word in query_lower for word in ['kubernetes', 'k8s']):
                knowledge_data["kubernetes"] = "Kubernetes best practices include resource management, security policies, monitoring, logging, and proper deployment strategies."
            elif any(word in query_lower for word in ['migration', 'migrate']):
                knowledge_data["migration"] = "Migration best practices include thorough planning, testing, backup strategies, gradual rollouts, and rollback procedures."
            else:
                knowledge_data["general"] = "General software development and infrastructure best practices."
        
        return knowledge_data
    
    async def _synthesize_answer(self, response_data: Dict) -> str:
        """Synthesize final answer using OpenAI."""
        
        try:
            # Create synthesis prompt
            query = response_data['query']
            repository = response_data['repository']
            routing = response_data.get('routing_decision', {})
            live_data = response_data.get('live_data', {})
            knowledge_data = response_data.get('knowledge_data', {})
            
            # Determine what data is actually available
            has_live_data = live_data and not live_data.get('error') and len(live_data) > 0
            has_knowledge_data = knowledge_data and not knowledge_data.get('error') and len(knowledge_data) > 0
            
            if has_live_data and has_knowledge_data:
                data_context = f"Live GitHub data: {live_data}\n\nKnowledge/Best Practices: {knowledge_data}"
                instruction = "Combine the live repository data with the best practices knowledge to provide a comprehensive answer."
            elif has_live_data:
                data_context = f"Live GitHub data: {live_data}"
                instruction = "Use the live repository data to answer the question. If additional context or best practices would be helpful, mention them generally."
            elif has_knowledge_data:
                data_context = f"Knowledge/Best Practices: {knowledge_data}"
                instruction = "Use the best practices and knowledge to answer the question. Be specific and actionable."
            else:
                data_context = "No specific data available for this repository."
                instruction = "Provide a helpful general answer based on common best practices and industry standards."
            
            prompt = f"""
You are a helpful GitHub and software development assistant. Answer the user's question clearly and comprehensively.

Question: {query}
Repository Context: {repository}
Routing: {routing.get('reasoning', 'General query')}

Available Information:
{data_context}

Instructions: {instruction}

Requirements:
1. Answer the user's question directly
2. Be practical and actionable
3. Use specific information when available
4. Provide context and reasoning
5. Maintain a helpful, professional tone

Answer:
"""
            
            ai_response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful GitHub assistant that provides clear, accurate answers by combining live repository data with knowledge and best practices."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.3
            )
            
            return ai_response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"AI synthesis error: {e}. " + self._simple_answer_synthesis(response_data)
    
    def _simple_answer_synthesis(self, response_data: Dict) -> str:
        """Simple fallback answer synthesis without AI."""
        
        answer_parts = []
        
        # Add routing info
        routing = response_data.get('routing_decision', {})
        answer_parts.append(f"Query type: {routing.get('type', 'unknown')}")
        
        # Add live data summary
        if response_data.get('live_data'):
            live_data = response_data['live_data']
            if 'commits' in live_data:
                answer_parts.append("Recent commits data available")
            if 'issues' in live_data:
                answer_parts.append("Current issues data available")
            if 'files' in live_data:
                answer_parts.append("Repository files data available")
        
        # Add knowledge summary
        if response_data.get('knowledge_data'):
            knowledge = response_data['knowledge_data']
            topics = list(knowledge.keys())
            answer_parts.append(f"Knowledge available on: {', '.join(topics)}")
        
        return " | ".join(answer_parts)
    
    def _simple_routing(self, query: str) -> RoutingDecision:
        """Simple fallback routing without AI."""
        
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['recent', 'current', 'latest', 'open']):
            return RoutingDecision(
                query_type=QueryType.LIVE_DATA,
                confidence=0.6,
                reasoning="Simple routing: detected live data keywords",
                mcp_tools=[],
                rag_topics=[],
                priority="mcp"
            )
        elif any(word in query_lower for word in ['how to', 'best practice', 'guide']):
            return RoutingDecision(
                query_type=QueryType.KNOWLEDGE,
                confidence=0.6,
                reasoning="Simple routing: detected knowledge keywords",
                mcp_tools=[],
                rag_topics=[],
                priority="rag"
            )
        else:
            return RoutingDecision(
                query_type=QueryType.HYBRID,
                confidence=0.5,
                reasoning="Simple routing: default to hybrid",
                mcp_tools=[],
                rag_topics=[],
                priority="balanced"
            )
    
    async def cleanup(self):
        """Clean up connections and resources."""
        try:
            # Clean up MCP client
            if self.mcp_client:
                try:
                    await self.mcp_client.disconnect()
                except Exception as e:
                    print(f"Warning: MCP cleanup error: {e}")
            
            # Clean up RAG connector
            if self.rag_connector:
                try:
                    self.rag_connector.close()
                except Exception as e:
                    print(f"Warning: RAG cleanup error: {e}")
            
            # Clean up OpenAI client (most OpenAI clients don't need async cleanup)
            if self.openai_client:
                try:
                    # Only try to close if it has an async close method
                    if hasattr(self.openai_client, 'close') and asyncio.iscoroutinefunction(self.openai_client.close):
                        await self.openai_client.close()
                except Exception as e:
                    print(f"Warning: OpenAI cleanup error: {e}")
                    
        except Exception as e:
            print(f"Warning: General cleanup error: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the assistant."""
        return {
            "query_router": self.query_router is not None,
            "openai_client": self.openai_client is not None,
            "mcp_available": self.is_mcp_available,
            "rag_available": self.is_rag_available,
            "sources_ready": self.is_mcp_available or self.is_rag_available
        }


# Alias for backward compatibility
EnhancedGitHubAssistant = KimchiAssistant


# Demo function
async def demo_kimchi_assistant():
    """Demonstrate the Kimchi assistant capabilities."""
    
    assistant = KimchiAssistant()
    await assistant.initialize()
    
    # Test queries with different routing needs
    test_queries = [
        "What are the recent commits in this repository?",
        "How should I set up continuous integration for this project?", 
        "Analyze the current repository state and suggest improvements",
        "Show me the open issues and provide troubleshooting guidance",
        "What files are in the repository and how should they be organized?"
    ]
    
    print("\n" + "="*70)
    print("KIMCHI HYBRID ASSISTANT DEMO")
    print("="*70)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*20} Query {i} {'='*20}")
        
        response = await assistant.answer_question(query)
        
        print(f"Query: {response['query']}")
        print(f"Repository: {response['repository']}")
        print(f"Routing: {response['routing_decision']['type']} ({response['routing_decision']['confidence']:.2f})")
        print(f"Sources: {', '.join(response['sources_used']) if response['sources_used'] else 'None'}")
        print(f"Answer: {response['synthesized_answer'][:200]}...")
        print(f"Confidence: {response['confidence']:.2f}")
    
    await assistant.cleanup()
    print(f"\nKimchi assistant demo completed!")
    print(f"Status: {assistant.get_status()}")


if __name__ == "__main__":
    asyncio.run(demo_kimchi_assistant())

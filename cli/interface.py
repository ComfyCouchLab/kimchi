"""
Command-line interface for the Kimchi GitHub Assistant.

This module provides both interactive and single-query modes for the assistant.
"""

import asyncio
import sys
import traceback
from typing import Optional
from pathlib import Path
import logging

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging quietly (this will be set up in main.py, but adding as fallback)
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

from core.assistant import KimchiAssistant


class KimchiCLI:
    """Command-line interface for the Kimchi GitHub Assistant."""
    
    def __init__(self):
        self.assistant = KimchiAssistant()
        self.initialized = False
    
    async def initialize(self) -> bool:
        """Initialize the assistant and return success status."""
        try:
            await self.assistant.initialize()
            self.initialized = True
            return True
        except Exception as e:
            print(f"Failed to initialize assistant: {e}")
            print("Please check your configuration and try again.")
            return False
    
    async def run_single_query(self, query: str) -> None:
        """Run a single query and exit."""
        if not self.initialized:
            if not await self.initialize():
                sys.exit(1)
        
        print(f"\nQuery: {query}")
        print("=" * 60)
        
        try:
            response_data = await self.assistant.answer_question(query)
            # Extract the synthesized answer from the response
            response = response_data.get("synthesized_answer", "No response generated")
            print(f"\nResponse:\n{response}")
            
            # Show additional info in debug mode
            if "--debug" in sys.argv:
                print(f"\nDebug Info:")
                routing_type = response_data.get('routing_decision', {}).get('type', 'unknown')
                print(f"Routing: {routing_type}")
                print(f"Confidence: {response_data.get('confidence', 'unknown')}")
                print(f"Sources used: {response_data.get('sources_used', [])}")
        except Exception as e:
            print(f"Error processing query: {e}")
            if "--debug" in sys.argv:
                traceback.print_exc()
            sys.exit(1)
        finally:
            # Cleanup after single query
            try:
                await self.assistant.cleanup()
            except Exception as e:
                print(f"Warning: Cleanup error: {e}")
    
    async def run_interactive(self) -> None:
        """Run in interactive mode with continuous query processing."""
        if not self.initialized:
            if not await self.initialize():
                sys.exit(1)
        
        self._print_welcome()
        
        try:
            while True:
                try:
                    # Get user input
                    query = input("\nAsk me about GitHub (or 'quit'/'exit'): ").strip()
                    
                    # Handle exit commands
                    if query.lower() in ['quit', 'exit', 'q']:
                        print("\nGoodbye!")
                        break
                    
                    # Handle empty input
                    if not query:
                        continue
                    
                    # Handle help
                    if query.lower() in ['help', '?']:
                        self._print_help()
                        continue
                    
                    # Handle status check
                    if query.lower() in ['status', 'info']:
                        await self._print_status()
                        continue
                    
                    # Process the query
                    print(f"\nProcessing: {query}")
                    print("-" * 50)
                    
                    response_data = await self.assistant.answer_question(query)
                    # Extract the synthesized answer from the response
                    response = response_data.get("synthesized_answer", "No response generated")
                    print(f"\nResponse:\n{response}")
                    
                    # Show routing info if available
                    routing = response_data.get('routing_decision', {})
                    if routing and hasattr(routing, 'query_type'):
                        print(f"\nRouted to: {routing.query_type.value} (confidence: {routing.confidence:.2f})")
                    
                except KeyboardInterrupt:
                    print("\n\nInterrupted. Goodbye!")
                    break
                except EOFError:
                    print("\n\nEOF detected. Goodbye!")
                    break
                except Exception as e:
                    print(f"\nError: {e}")
                    if "--debug" in sys.argv:
                        traceback.print_exc()
        
        finally:
            # Cleanup
            try:
                await self.assistant.cleanup()
            except:
                pass
    
    def _print_welcome(self) -> None:
        """Print welcome message and capabilities."""
        print("\n" + "=" * 60)
        print("Welcome to Kimchi - Your Hybrid GitHub Assistant!")
        print("=" * 60)
        print("\nCapabilities:")
        print("• Knowledge queries (best practices, tutorials)")
        print("• Live GitHub data (commits, issues, files)")
        print("• Intelligent routing for optimal responses")
        print("• Hybrid analysis combining both sources")
        
        # Show available systems
        systems = []
        if self.assistant.is_rag_available:
            systems.append("RAG (Knowledge)")
        if self.assistant.is_mcp_available:
            systems.append("MCP (Live Data)")
        
        if systems:
            print(f"\n✅ Available systems: {', '.join(systems)}")
        else:
            print("\n⚠️  No systems available - check configuration")
        
        print("\nCommands:")
        print("• Type your question naturally")
        print("• 'help' or '?' - Show this help")
        print("• 'status' - Show system status")
        print("• 'quit' or 'exit' - Exit the assistant")
        print("=" * 60)
    
    def _print_help(self) -> None:
        """Print help information."""
        print("\n📖 Help - Example Queries:")
        print("-" * 30)
        print("\n🔄 Live Data Queries (MCP):")
        print("• What are the recent commits?")
        print("• Show me open issues")
        print("• What files are in the main branch?")
        print("• Who are the contributors?")
        print("• Show me the README content")
        
        print("\n📚 Knowledge Queries (RAG):")
        print("• How do I set up CI/CD?")
        print("• What are testing best practices?")
        print("• Explain deployment strategies")
        print("• Security best practices")
        print("• Code review guidelines")
        
        print("\n🔀 Hybrid Queries (Both):")
        print("• Analyze recent commits and suggest improvements")
        print("• Review current issues and provide solutions")
        print("• Check repository health and recommend practices")
    
    async def _print_status(self) -> None:
        """Print current system status."""
        print("\n📊 System Status:")
        print("-" * 20)
        print(f"RAG System: {'✅ Available' if self.assistant.is_rag_available else '❌ Unavailable'}")
        print(f"MCP System: {'✅ Available' if self.assistant.is_mcp_available else '❌ Unavailable'}")
        
        if hasattr(self.assistant, 'query_router') and self.assistant.query_router:
            print("Query Router: ✅ Available")
        else:
            print("Query Router: ❌ Unavailable")
        
        if hasattr(self.assistant, 'openai_client') and self.assistant.openai_client:
            print("OpenAI Client: ✅ Available")
        else:
            print("OpenAI Client: ❌ Unavailable")

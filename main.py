#!/usr/bin/env python3
"""
Kimchi - Hybrid RAG + MCP GitHub Assistant

Main entry point for the intelligent GitHub assistant that combines:
- RAG (Retrieval Augmented Generation) for static knowledge
- MCP (Model Context Protocol) for live GitHub data
- AI-powered routing for optimal responses

Usage:
    python main.py                           # Interactive mode
    python main.py "What are recent commits?" # Single query mode
    python main.py --help                    # Show help
"""

import asyncio
import argparse
import sys
from pathlib import Path
import traceback

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Set up quiet logging early to suppress verbose messages
from utils.logging import setup_quiet_logging
setup_quiet_logging()

from cli.interface import KimchiCLI


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Kimchi - Hybrid RAG + MCP GitHub Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Interactive mode
  %(prog)s "What are recent commits?"         # Single query
  %(prog)s "How do I set up CI/CD?" --debug   # Single query with debug
  
Environment Variables:
  OPENAI_API_KEY          # Required for AI routing and synthesis
  GITHUB_TOKEN           # Optional for enhanced GitHub access
  ELASTICSEARCH_URL      # Optional for RAG functionality
        """
    )
    
    parser.add_argument(
        'query',
        nargs='?',
        help='Single query to process (if not provided, enters interactive mode)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug output with full error traces'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='Kimchi 1.0.0 - Hybrid GitHub Assistant'
    )
    
    return parser


async def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Create CLI instance
    cli = KimchiCLI()
    
    try:
        if args.query:
            # Single query mode
            await cli.run_single_query(args.query)
        else:
            # Interactive mode
            await cli.run_interactive()
    
    except KeyboardInterrupt:
        print("\nInterrupted. Goodbye!")
    except Exception as e:
        print(f"Unexpected error: {e}")
        if args.debug:
            traceback.print_exc()
    finally:
        # Ensure cleanup always happens
        try:
            if cli.initialized and cli.assistant:
                await cli.assistant.cleanup()
        except Exception as e:
            print(f"Warning: Cleanup error: {e}")


if __name__ == "__main__":
    # Set up proper event loop for asyncio
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        print(f"Fatal error: {e}")
    finally:
        # Give a moment for cleanup
        import time
        time.sleep(0.1)

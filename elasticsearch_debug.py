#!/usr/bin/env python3
"""
Elasticsearch Observability CLI Tool

This script provides various observability features for your Elasticsearch RAG system:
- Query analysis and debugging
- Index inspection
- Performance monitoring
- Query comparison
- Result quality assessment

Usage:
    python elasticsearch_debug.py --query "What's EIS?"
    python elasticsearch_debug.py --analyze-index
    python elasticsearch_debug.py --compare "query1" "query2"
    python elasticsearch_debug.py --demo
"""

import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Verify OpenAI API key is loaded
if not os.getenv('OPENAI_API_KEY'):
    print("❌ OPENAI_API_KEY not found in environment variables")
    print("Please check your .env file and ensure OPENAI_API_KEY is set")
    sys.exit(1)

# Add parent directory to path for imports
parent_dir = Path(__file__).parent
sys.path.insert(0, str(parent_dir))

from connectors.elasticsearch_connector import ElasticsearchConnector
from utils.elasticsearch_observer import ElasticsearchObserver, create_observability_demo
from config import load_config


def setup_elasticsearch_connector():
    """Initialize and return an Elasticsearch connector."""
    try:
        config = load_config()
        connector = ElasticsearchConnector(config.elasticsearch, config.embedding_model)
        
        # Connect synchronously
        store = connector.connect()
        if store:
            print("✅ Connected to Elasticsearch")
            return connector
        else:
            print("❌ Failed to connect to Elasticsearch")
            return None
            
    except Exception as e:
        print(f"❌ Error setting up Elasticsearch: {e}")
        return None


def analyze_single_query(connector, query: str, k: int = 5):
    """Analyze a single query with detailed observability."""
    print(f"\n🔍 ANALYZING QUERY: '{query}'")
    print("="*60)
    
    # Enable query logging
    connector.enable_elasticsearch_query_logging()
    
    # Create observer and analyze
    observer = ElasticsearchObserver(connector)
    analysis = observer.analyze_query(query, k)
    
    # Get suggestions
    suggestions = observer.suggest_improvements(analysis)
    
    print(f"\n💡 SUGGESTIONS:")
    for suggestion in suggestions:
        print(f"  {suggestion}")
    
    return analysis


def analyze_index(connector):
    """Perform comprehensive index analysis."""
    print(f"\n📊 INDEX ANALYSIS")
    print("="*60)
    
    # Get index statistics
    print("Getting index statistics...")
    stats = connector.get_index_stats()
    
    # Get index mapping
    print("\nGetting index mapping...")
    mapping = connector.get_index_mapping()
    
    # Sample documents
    print("\nSampling documents...")
    samples = connector.sample_documents(3)
    
    return {
        "stats": stats,
        "mapping": mapping,
        "samples": samples
    }


def compare_queries(connector, query1: str, query2: str, k: int = 5):
    """Compare two queries to understand differences."""
    print(f"\n🔄 COMPARING QUERIES")
    print("="*60)
    
    observer = ElasticsearchObserver(connector)
    comparison = observer.compare_queries(query1, query2, k)
    
    return comparison


def run_demo(connector):
    """Run a comprehensive observability demo."""
    print(f"\n🎯 RUNNING OBSERVABILITY DEMO")
    print("="*60)
    
    create_observability_demo(connector)


def interactive_mode(connector):
    """Run in interactive mode for continuous query analysis."""
    print(f"\n🖥️  INTERACTIVE MODE")
    print("="*60)
    print("Enter queries to analyze (type 'quit' to exit)")
    print("Commands:")
    print("  analyze <query>     - Analyze a query")
    print("  compare <q1> <q2>   - Compare two queries")
    print("  index               - Analyze index")
    print("  demo                - Run demo")
    print("  quit                - Exit")
    
    observer = ElasticsearchObserver(connector)
    
    while True:
        try:
            user_input = input("\n> ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            elif user_input.lower() == 'index':
                analyze_index(connector)
            elif user_input.lower() == 'demo':
                run_demo(connector)
            elif user_input.startswith('analyze '):
                query = user_input[8:].strip()
                if query:
                    analyze_single_query(connector, query)
            elif user_input.startswith('compare '):
                parts = user_input[8:].split(' ', 1)
                if len(parts) == 2:
                    # Try to split on quotes or just split in half
                    if '"' in parts[1]:
                        queries = [q.strip('"').strip() for q in parts[1].split('"') if q.strip('"').strip()]
                        if len(queries) >= 2:
                            compare_queries(connector, queries[0], queries[1])
                        else:
                            print("Please provide two queries in quotes")
                    else:
                        # Split in half
                        mid = len(parts[1]) // 2
                        q1 = parts[1][:mid].strip()
                        q2 = parts[1][mid:].strip()
                        compare_queries(connector, q1, q2)
                else:
                    print("Usage: compare <query1> <query2>")
            elif user_input:
                # Default to analyze
                analyze_single_query(connector, user_input)
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n👋 Goodbye!")


def main():
    parser = argparse.ArgumentParser(description="Elasticsearch Observability Tool")
    parser.add_argument("--query", "-q", type=str, help="Query to analyze")
    parser.add_argument("--k", type=int, default=5, help="Number of results to retrieve")
    parser.add_argument("--analyze-index", action="store_true", help="Analyze index structure and stats")
    parser.add_argument("--compare", nargs=2, metavar=("QUERY1", "QUERY2"), help="Compare two queries")
    parser.add_argument("--demo", action="store_true", help="Run observability demo")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
    parser.add_argument("--enable-debug", action="store_true", help="Enable detailed ES query logging")
    
    args = parser.parse_args()
    
    # Set debug environment variable if requested
    if args.enable_debug:
        os.environ['ELASTICSEARCH_DEBUG'] = 'true'
    
    # Setup connector
    connector = setup_elasticsearch_connector()
    if not connector:
        print("❌ Cannot proceed without Elasticsearch connection")
        sys.exit(1)
    
    try:
        if args.query:
            analyze_single_query(connector, args.query, args.k)
        elif args.analyze_index:
            analyze_index(connector)
        elif args.compare:
            compare_queries(connector, args.compare[0], args.compare[1], args.k)
        elif args.demo:
            run_demo(connector)
        elif args.interactive:
            interactive_mode(connector)
        else:
            # Default to interactive mode
            print("No specific command provided. Starting interactive mode...")
            interactive_mode(connector)
            
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        if connector:
            connector.close()


if __name__ == "__main__":
    main()

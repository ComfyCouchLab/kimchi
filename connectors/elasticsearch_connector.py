"""
Elasticsearch Connector - Handles document parsing, embedding generation, and ES operations.

This module provides functionality to:
- Parse documents using various parsers (Markdown, Code, JSON, etc.)
- Generate embeddings using OpenAI
- Store documents in Elasticsearch with vector search capabilities
"""

import os
import time
import glob
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

from llama_index.core.node_parser import (
    CodeSplitter, MarkdownNodeParser, JSONNodeParser, 
    SentenceSplitter, TokenTextSplitter
)
from llama_index.core import Document, Settings, SimpleDirectoryReader
from llama_index.vector_stores.elasticsearch import ElasticsearchStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.ingestion import IngestionPipeline
from elasticsearch import AsyncElasticsearch
import elastic_transport
import nest_asyncio
from dotenv import load_dotenv


@dataclass
class ElasticsearchConfig:
    """Configuration for Elasticsearch operations."""
    
    cloud_id: str
    username: str
    password: str
    index_name: str
    batch_size: int = 100
    request_timeout: int = 120
    max_retries: int = 5
    connection_retries: int = 20
    connection_retry_delay: int = 15


@dataclass
class ParserConfig:
    """Configuration for document parsers."""
    
    chunk_size: int = 750
    chunk_overlap: int = 50
    max_chars: int = 1500


class ElasticsearchConnectorError(Exception):
    """Custom exception for Elasticsearch connector operations."""
    pass


class DocumentProcessor:
    """Handles document parsing and processing operations."""
    
    def __init__(self, parser_config: Optional[ParserConfig] = None):
        """
        Initialize document processor.
        
        Args:
            parser_config: Configuration for parsers. Uses defaults if None.
        """
        self.config = parser_config or ParserConfig()
        self.parsers_and_extensions = self._setup_parsers()
    
    def _setup_parsers(self) -> List[Tuple[Any, List[str]]]:
        """Setup document parsers with their corresponding file extensions."""
        try:
            parsers = [
                (SentenceSplitter(
                    chunk_size=self.config.chunk_size, 
                    chunk_overlap=self.config.chunk_overlap
                ), [".md"]),
                (MarkdownNodeParser(), [".md"]),
                # Uncomment and configure as needed:
                # (CodeSplitter(language='python', max_chars=self.config.max_chars), [".py", ".ipynb"]),
                # (CodeSplitter(language='typescript', max_chars=self.config.max_chars), [".ts"]),
                # (CodeSplitter(language='go', max_chars=self.config.max_chars), [".go"]),
                # (CodeSplitter(language='javascript', max_chars=self.config.max_chars), [".js"]),
                # (CodeSplitter(language='bash', max_chars=self.config.max_chars), [".bash", ".sh"]),
                # (CodeSplitter(language='yaml', max_chars=self.config.max_chars), [".yaml", ".yml"]),
                # (JSONNodeParser(), [".json"]),
            ]
            return parsers
        except Exception as e:
            print(f"⚠️  Error setting up parsers: {e}")
            # Return a minimal fallback configuration
            return [
                (SentenceSplitter(chunk_size=500, chunk_overlap=20), [".md"]),
            ]
    
    def parse_documents(self, repo_path: str, verbose: bool = True) -> List[Any]:
        """
        Parse documents from the repository path.
        
        Args:
            repo_path: Path to the repository containing documents.
            verbose: Whether to print detailed information during processing.
            
        Returns:
            List of parsed nodes.
        """
        nodes = []
        file_summary = []
        
        if verbose:
            self._print_parser_configurations()
        
        for parser, extensions in self.parsers_and_extensions:
            matching_files = []
            for ext in extensions:
                matching_files.extend(glob.glob(f"{repo_path}/**/*{ext}", recursive=True))
            
            if len(matching_files) > 0:
                file_summary.append(
                    f"Found {len(matching_files)} {', '.join(extensions)} files in the repository."
                )
                
                loader = SimpleDirectoryReader(
                    input_dir=repo_path, 
                    required_exts=extensions, 
                    recursive=True
                )
                docs = loader.load_data()
                parsed_nodes = parser.get_nodes_from_documents(docs)
                
                if verbose:
                    parser_name = type(parser).__name__
                    self._print_individual_chunk_sizes(
                        parsed_nodes, f"{parser_name} ({', '.join(extensions)})"
                    )
                    self._print_chunk_size_summary(
                        parsed_nodes, f"{parser_name} ({', '.join(extensions)})", parser
                    )
                    self._print_docs_and_nodes(docs, parsed_nodes)
                
                nodes.extend(parsed_nodes)
            else:
                file_summary.append(f"No {', '.join(extensions)} files found in the repository.")
        
        if verbose:
            self._collect_and_print_file_summary(file_summary)
            if nodes:
                self._print_chunk_size_summary(nodes, "All Parsers Combined")
        
        return nodes
    
    def _print_docs_and_nodes(self, docs: List[Document], nodes: List[Any]) -> None:
        """Print information about documents and nodes."""
        print("\n=== Documents ===\n")
        for doc in docs:
            print(f"Document ID: {doc.doc_id}")
            print(f"Document file name: {doc.metadata.get('file_name', 'N/A')}")
            print(f"Document Content:\n{doc.text}\n\n---\n")

        print("\n=== Nodes ===\n")
        for node in nodes:
            print(f"Node ID: {node.id_}")
            print(f"Node metadata: {node.metadata}")
            print(f"Node Content Length: {len(node.text)} characters")
            print(f"Node Content:\n{node.text}\n\n---\n")
    
    def _print_chunk_size_summary(self, nodes: List[Any], parser_type: str, parser: Any = None) -> None:
        """Print summary of chunk sizes for nodes created by a specific parser."""
        if not nodes:
            print(f"No nodes created by {parser_type}")
            return
        
        chunk_sizes = [len(node.text) for node in nodes]
        print(f"\n=== Chunk Size Summary for {parser_type} ===")
        print(f"Total nodes: {len(nodes)}")
        print(f"Average chunk size: {sum(chunk_sizes) / len(chunk_sizes):.2f} characters")
        print(f"Min chunk size: {min(chunk_sizes)} characters")
        print(f"Max chunk size: {max(chunk_sizes)} characters")
        
        if parser:
            max_chars = getattr(parser, 'max_chars', None)
            chunk_overlap = getattr(parser, 'chunk_overlap', None)
            
            if max_chars:
                print(f"Parser max_chars setting: {max_chars}")
            if chunk_overlap:
                print(f"Parser chunk_overlap: {chunk_overlap}")
            
            if max_chars:
                max_chunk_size = max(chunk_sizes)
                avg_chunk_size = sum(chunk_sizes) / len(chunk_sizes)
                
                print(f"Context Window Analysis:")
                print(f"  - Max chunk utilization: {max_chunk_size}/{max_chars} ({(max_chunk_size/max_chars)*100:.1f}%)")
                print(f"  - Avg chunk utilization: {avg_chunk_size:.0f}/{max_chars} ({(avg_chunk_size/max_chars)*100:.1f}%)")

                if max_chunk_size > max_chars:
                    print(f"  ⚠️  WARNING: Some chunks exceed max_chars limit!")
                elif max_chunk_size < max_chars * 0.5:
                    print(f"  ℹ️  INFO: Chunks are using less than 50% of available context window")
            else:
                print(f"Parser type: {type(parser).__name__} (no explicit max_chars limit)")
        
        print(f"Chunk sizes: {chunk_sizes}")
        print("---\n")
    
    def _collect_and_print_file_summary(self, file_summary: List[str]) -> None:
        """Print file summary information."""
        print("\n=== File Summary ===\n")
        for summary in file_summary:
            print(summary)
    
    def _print_parser_configurations(self) -> None:
        """Print configuration details for all parsers."""
        print("\n=== Parser Configurations ===")
        for parser, extensions in self.parsers_and_extensions:
            parser_name = type(parser).__name__
            print(f"\n{parser_name} for {', '.join(extensions)}:")
            
            for attr in ['max_chars', 'chunk_overlap', 'language', 'chunk_size', 'chunk_overlap_ratio', 'max_chunk_size']:
                if hasattr(parser, attr):
                    print(f"  - {attr}: {getattr(parser, attr)}")
        print("=" * 40 + "\n")
    
    def _print_individual_chunk_sizes(self, nodes: List[Any], parser_name: str = "") -> None:
        """Print chunk size for each individual node."""
        print(f"\n=== Individual Chunk Sizes for {parser_name} ===")
        for i, node in enumerate(nodes, 1):
            file_name = node.metadata.get('file_name', 'Unknown')
            chunk_size = len(node.text)
            print(f"Node {i}: {chunk_size:,} chars from '{file_name}'")
        print()


class ElasticsearchConnector:
    """
    Handles Elasticsearch operations including document ingestion and vector storage.
    
    This connector is responsible for:
    - Connecting to Elasticsearch cluster
    - Creating and managing vector stores
    - Running ingestion pipelines
    - Managing embeddings
    """
    
    def __init__(self, config: Optional[ElasticsearchConfig] = None, embedding_model: str = "text-embedding-3-large"):
        """
        Initialize the Elasticsearch connector.
        
        Args:
            config: ElasticsearchConfig object. If None, will load from environment variables.
            embedding_model: OpenAI embedding model to use.
        """
        nest_asyncio.apply()
        load_dotenv('.env')
        
        # Verify OpenAI API key is available
        if not os.getenv('OPENAI_API_KEY'):
            raise ElasticsearchConnectorError(
                "OPENAI_API_KEY not found in environment variables. "
                "Please ensure it's set in your .env file."
            )
        
        self.config = config or self._load_config_from_env()
        self.embedding_model = embedding_model
        
        # Configure LlamaIndex settings
        try:
            Settings.embed_model = OpenAIEmbedding(model=self.embedding_model)
            print(f"✅ OpenAI embedding model '{self.embedding_model}' configured successfully")
        except Exception as e:
            raise ElasticsearchConnectorError(f"Failed to configure OpenAI embedding model: {e}")
        
        self.vector_store = None
        self.document_processor = DocumentProcessor()
    
    def _load_config_from_env(self) -> ElasticsearchConfig:
        """Load configuration from environment variables."""
        cloud_id = os.getenv("ELASTIC_CLOUD_ID")
        username = os.getenv("ELASTIC_USER")
        password = os.getenv("ELASTIC_PASSWORD")
        index_name = os.getenv("ELASTIC_INDEX")
        
        if not all([cloud_id, username, password, index_name]):
            raise ElasticsearchConnectorError(
                "Required Elasticsearch environment variables are missing: "
                "ELASTIC_CLOUD_ID, ELASTIC_USER, ELASTIC_PASSWORD, ELASTIC_INDEX"
            )
        
        return ElasticsearchConfig(
            cloud_id=cloud_id,
            username=username,
            password=password,
            index_name=index_name
        )
    
    def connect(self) -> ElasticsearchStore:
        """
        Establish connection to Elasticsearch and create vector store.
        
        Returns:
            ElasticsearchStore: Configured vector store.
            
        Raises:
            ElasticsearchConnectorError: If connection fails after retries.
        """
        print("Initializing Elasticsearch store...")
        
        es_client = AsyncElasticsearch(
            cloud_id=self.config.cloud_id,
            basic_auth=(self.config.username, self.config.password),
            request_timeout=self.config.request_timeout,
            retry_on_timeout=True,
            max_retries=self.config.max_retries,
        )
        
        for attempt in range(self.config.connection_retries):
            try:
                self.vector_store = ElasticsearchStore(
                    index_name=self.config.index_name,
                    es_client=es_client,
                    batch_size=self.config.batch_size
                )
                print("Elasticsearch store initialized successfully.")
                return self.vector_store
                
            except elastic_transport.ConnectionTimeout:
                print(f"Connection attempt {attempt + 1}/{self.config.connection_retries} timed out. Retrying...")
                time.sleep(self.config.connection_retry_delay)
        
        raise ElasticsearchConnectorError(
            f"Failed to initialize Elasticsearch store after {self.config.connection_retries} attempts"
        )
    
    def process_and_ingest_documents(self, repo_path: str, show_progress: bool = True, verbose: bool = True) -> None:
        """
        Process documents from repository and ingest into Elasticsearch.
        
        Args:
            repo_path: Path to the repository containing documents.
            show_progress: Whether to show ingestion progress.
            verbose: Whether to print detailed processing information.
            
        Raises:
            ElasticsearchConnectorError: If ingestion fails.
        """
        if not self.vector_store:
            self.connect()
        
        # Parse documents
        print(f"Processing documents from: {repo_path}")
        nodes = self.document_processor.parse_documents(repo_path, verbose=verbose)
        
        if not nodes:
            print("No documents found to process.")
            return
        
        print(f"Found {len(nodes)} nodes to ingest into Elasticsearch.")
        
        try:
            # Create and run ingestion pipeline
            pipeline = IngestionPipeline(vector_store=self.vector_store)
            pipeline.run(documents=nodes, show_progress=show_progress)
            print(f"Successfully ingested {len(nodes)} nodes into Elasticsearch index '{self.config.index_name}'.")
            
        except Exception as e:
            raise ElasticsearchConnectorError(f"Failed to ingest documents: {str(e)}")
    
    def search_documents(self, query: str, k: int = 5, debug: bool = True) -> List[Dict[str, Any]]:
        """
        Search for documents using vector similarity with detailed observability.
        
        Args:
            query: Search query string
            k: Number of results to return
            debug: Enable detailed logging of search process
            
        Returns:
            List of search results with content and metadata
        """
        try:
            if not self.vector_store:
                raise ElasticsearchConnectorError("Vector store not initialized. Call connect() first.")
            
            if debug:
                print(f"\n🔍 ELASTICSEARCH SEARCH DEBUG")
                print(f"{'='*50}")
                print(f"📝 Original Query: '{query}'")
                print(f"🎯 Results requested (k): {k}")
                print(f"📊 Index: {self.config.index_name}")
                print(f"🤖 Embedding Model: {self.embedding_model}")
            
            # Create a VectorStoreIndex from the existing store
            from llama_index.core import VectorStoreIndex
            from llama_index.core import Settings
            
            # Set up embedding model
            from llama_index.embeddings.openai import OpenAIEmbedding
            embed_model = OpenAIEmbedding(model=self.embedding_model)
            Settings.embed_model = embed_model
            
            # Generate embedding for the query to show what's being searched
            if debug:
                query_embedding = embed_model.get_text_embedding(query)
                print(f"🧮 Query Embedding Dimensions: {len(query_embedding)}")
                print(f"🧮 First 5 embedding values: {query_embedding[:5]}")
            
            # Create index
            index = VectorStoreIndex.from_vector_store(self.vector_store)
            
            # Create query engine with custom retriever for observability
            retriever = index.as_retriever(similarity_top_k=k)
            
            if debug:
                print(f"\n🔄 Performing vector similarity search...")
            
            # Perform retrieval to get detailed information
            nodes = retriever.retrieve(query)
            
            if debug:
                print(f"📊 Retrieved {len(nodes)} nodes")
                if hasattr(self.vector_store, '_client'):
                    # Try to get the last query executed (if available)
                    try:
                        # This is a workaround to show what's happening under the hood
                        print(f"📍 Vector Store Type: {type(self.vector_store).__name__}")
                        if hasattr(self.vector_store, '_index'):
                            print(f"📍 ES Index Name: {self.vector_store._index}")
                    except Exception as e:
                        print(f"📍 Could not retrieve ES details: {e}")
            
            # Format results with enhanced metadata
            formatted_results = []
            for i, node in enumerate(nodes):
                result = {
                    'content': node.node.text,
                    'metadata': node.node.metadata,
                    'score': node.score if hasattr(node, 'score') else 'N/A',
                    'node_id': node.node.node_id if hasattr(node.node, 'node_id') else f'node_{i}'
                }
                formatted_results.append(result)
                
                if debug:
                    print(f"\n📄 Result {i+1}:")
                    print(f"  🏷️  Score: {result['score']}")
                    print(f"  🆔 Node ID: {result['node_id']}")
                    print(f"  📁 Metadata: {result['metadata']}")
                    print(f"  📝 Content preview: {result['content'][:200]}...")
            
            if debug:
                print(f"\n{'='*50}")
                print(f"✅ Search completed successfully")
            
            return formatted_results
            
        except Exception as e:
            print(f"❌ Search error: {e}")
            import traceback
            traceback.print_exc()
            return []

    async def initialize(self) -> None:
        """Initialize the connector for use in async context."""
        try:
            self.vector_store = self.connect()
        except Exception as e:
            raise ElasticsearchConnectorError(f"Failed to initialize: {e}")

    def close(self) -> None:
        """Close Elasticsearch connection."""
        try:
            if self.vector_store:
                # Close the vector store if it has a close method
                if hasattr(self.vector_store, "close"):
                    self.vector_store.close()
                
                # Close the underlying Elasticsearch client
                if hasattr(self.vector_store, '_client') and self.vector_store._client:
                    try:
                        # Close async client if it exists
                        if hasattr(self.vector_store._client, 'close'):
                            import asyncio
                            try:
                                loop = asyncio.get_event_loop()
                                if loop.is_running():
                                    # Schedule close in the background
                                    asyncio.create_task(self.vector_store._client.close())
                                else:
                                    # Run close synchronously
                                    asyncio.run(self.vector_store._client.close())
                            except:
                                pass
                    except Exception as e:
                        print(f"Warning: ES client close error: {e}")
                
                self.vector_store = None
            print("Elasticsearch connection closed.")
        except Exception as e:
            print(f"Warning: Error closing Elasticsearch connection: {e}")
    
    def get_store_info(self) -> Dict[str, Any]:
        """
        Get information about the Elasticsearch configuration.
        
        Returns:
            dict: Store configuration information.
        """
        return {
            'index_name': self.config.index_name,
            'batch_size': self.config.batch_size,
            'embedding_model': self.embedding_model,
            'connected': self.vector_store is not None
        }
    
    def enable_elasticsearch_query_logging(self) -> None:
        """
        Enable detailed Elasticsearch query logging to see exactly what queries are executed.
        This will show the raw ES queries being sent to the cluster.
        """
        try:
            import logging
            import elasticsearch
            
            # Enable elasticsearch library debug logging
            es_logger = logging.getLogger('elasticsearch')
            es_logger.setLevel(logging.DEBUG)
            
            # Create console handler if not exists
            if not es_logger.handlers:
                console_handler = logging.StreamHandler()
                formatter = logging.Formatter('🔍 ES Query: %(message)s')
                console_handler.setFormatter(formatter)
                es_logger.addHandler(console_handler)
            
            # Enable trace logging for detailed request/response info
            trace_logger = logging.getLogger('elasticsearch.trace')
            trace_logger.setLevel(logging.DEBUG)
            
            if not trace_logger.handlers:
                trace_handler = logging.StreamHandler()
                trace_formatter = logging.Formatter('📡 ES Trace: %(message)s')
                trace_handler.setFormatter(trace_formatter)
                trace_logger.addHandler(trace_handler)
            
            print("✅ Elasticsearch query logging enabled")
            print("📋 You will now see raw ES queries in the console")
            
        except Exception as e:
            print(f"⚠️  Could not enable ES query logging: {e}")

    def get_index_mapping(self) -> Dict[str, Any]:
        """
        Get the current index mapping to understand the document structure.
        
        Returns:
            Dict containing the index mapping information
        """
        try:
            if not self.vector_store:
                return {"error": "Vector store not available"}
            
            print(f"\n📋 INDEX MAPPING FOR '{self.config.index_name}':")
            print("="*60)
            
            # Try to access the underlying Elasticsearch client
            if hasattr(self.vector_store, '_client'):
                client = self.vector_store._client
                try:
                    mapping = client.indices.get_mapping(index=self.config.index_name)
                    import json
                    print(json.dumps(mapping, indent=2))
                    return mapping
                except Exception as e:
                    print(f"Could not retrieve mapping via direct client: {e}")
            
            # Alternative approach - try to get basic info
            print("📍 Using LlamaIndex ElasticsearchStore")
            print(f"📍 Index Name: {self.config.index_name}")
            print(f"📍 Vector Store Type: {type(self.vector_store).__name__}")
            
            return {
                "index_name": self.config.index_name,
                "vector_store_type": type(self.vector_store).__name__,
                "note": "Direct mapping access not available through LlamaIndex abstraction"
            }
            
        except Exception as e:
            print(f"❌ Error getting index mapping: {e}")
            return {"error": str(e)}

    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get index statistics to understand the data volume and structure.
        
        Returns:
            Dict containing index statistics
        """
        try:
            if not self.vector_store:
                return {"error": "Vector store not available"}
            
            print(f"\n📊 INDEX STATISTICS FOR '{self.config.index_name}':")
            print("="*60)
            
            # Try to access the underlying Elasticsearch client
            if hasattr(self.vector_store, '_client'):
                client = self.vector_store._client
                try:
                    # Get index stats
                    stats = client.indices.stats(index=self.config.index_name)
                    
                    # Get document count
                    count_result = client.count(index=self.config.index_name)
                    doc_count = count_result['count']
                    
                    print(f"📄 Total documents: {doc_count}")
                    
                    if self.config.index_name in stats['indices']:
                        index_stats = stats['indices'][self.config.index_name]
                        total_size = index_stats['total']['store']['size_in_bytes']
                        print(f"💾 Total size: {total_size:,} bytes ({total_size/1024/1024:.2f} MB)")
                        print(f"🔍 Total searches: {index_stats['total']['search']['query_total']}")
                        print(f"⏱️  Search time: {index_stats['total']['search']['query_time_in_millis']} ms")
                    
                    return {
                        "document_count": doc_count,
                        "stats": stats,
                        "index_name": self.config.index_name
                    }
                    
                except Exception as e:
                    print(f"Could not retrieve stats via direct client: {e}")
            
            # Alternative approach
            print("📍 Using LlamaIndex ElasticsearchStore (limited stats)")
            print(f"📍 Index Name: {self.config.index_name}")
            print(f"📍 Vector Store Type: {type(self.vector_store).__name__}")
            
            return {
                "index_name": self.config.index_name,
                "vector_store_type": type(self.vector_store).__name__,
                "note": "Direct stats access not available through LlamaIndex abstraction"
            }
            
        except Exception as e:
            print(f"❌ Error getting index stats: {e}")
            return {"error": str(e)}

    def sample_documents(self, size: int = 5) -> List[Dict[str, Any]]:
        """
        Get a sample of documents from the index to understand the data structure.
        
        Args:
            size: Number of sample documents to retrieve
            
        Returns:
            List of sample documents
        """
        try:
            if not self.vector_store:
                return []
            
            print(f"\n📄 SAMPLE DOCUMENTS FROM '{self.config.index_name}':")
            print("="*60)
            
            # Try to access the underlying Elasticsearch client
            if hasattr(self.vector_store, '_client'):
                client = self.vector_store._client
                try:
                    # Search for sample documents
                    response = client.search(
                        index=self.config.index_name,
                        body={
                            "size": size,
                            "query": {"match_all": {}}
                        }
                    )
                    
                    documents = []
                    for i, hit in enumerate(response['hits']['hits']):
                        doc = hit['_source']
                        documents.append(doc)
                        
                        print(f"\n🔖 Document {i+1}:")
                        print(f"  🆔 ID: {hit['_id']}")
                        
                        # Show key fields
                        if 'content' in doc:
                            text_preview = doc['content'][:200] + "..." if len(doc['content']) > 200 else doc['content']
                            print(f"  📝 Content: {text_preview}")
                        
                        if 'text' in doc:
                            text_preview = doc['text'][:200] + "..." if len(doc['text']) > 200 else doc['text']
                            print(f"  📝 Text: {text_preview}")
                        
                        if 'metadata' in doc:
                            print(f"  📁 Metadata: {doc['metadata']}")
                        
                        if 'embedding' in doc:
                            print(f"  🧮 Has embedding: {len(doc['embedding'])} dimensions")
                        
                        # Show all available fields
                        print(f"  🔧 Available fields: {list(doc.keys())}")
                    
                    return documents
                    
                except Exception as e:
                    print(f"Could not retrieve samples via direct client: {e}")
            
            # Alternative approach - use search_documents to get samples
            print("📍 Using search_documents method to get samples")
            results = self.search_documents("*", k=size, debug=False)
            
            for i, result in enumerate(results[:size]):
                print(f"\n🔖 Document {i+1} (via search):")
                print(f"  🏆 Score: {result.get('score', 'N/A')}")
                
                content = result.get('content', '')
                if content:
                    preview = content[:200] + "..." if len(content) > 200 else content
                    print(f"  📝 Content: {preview}")
                
                if result.get('metadata'):
                    print(f"  📁 Metadata: {result['metadata']}")
            
            return results[:size]
            
        except Exception as e:
            print(f"❌ Error sampling documents: {e}")
            return []

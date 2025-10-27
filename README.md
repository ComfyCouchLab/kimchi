# Kimchi - Hybrid RAG + MCP GitHub Assistant

A sophisticated AI-powered assistant that combines Retrieval-Augmented Generation (RAG) with Model Context Protocol (MCP) to provide intelligent responses about GitHub repositories. The system intelligently routes queries between static knowledge (RAG) and live GitHub data (MCP) for optimal results.

## Overview

Kimchi is an advanced hybrid AI system that:

- **Intelligently routes queries** using OpenAI to determine the optimal data source
- **Combines RAG and MCP** for comprehensive repository analysis
- **Provides contextual responses** by synthesizing static documentation with live GitHub data
- **Supports private repositories** with proper authentication handling
- **Scales efficiently** using Elasticsearch for vector storage and Docker for MCP isolation

## Project Structure

```
kimchi/
├── main.py                          # CLI entry point
├── cli/
│   ├── __init__.py                  # CLI package
│   └── interface.py                 # Command-line interface logic
├── core/
│   ├── __init__.py                  # Core package exports
│   ├── assistant.py                 # Main KimchiAssistant class
│   └── query_router.py              # AI-powered query routing
├── connectors/
│   ├── __init__.py
│   ├── elasticsearch_connector.py   # RAG vector search
│   ├── github_connector.py          # GitHub API wrapper
│   └── mcp_github_connector.py      # MCP protocol handler
├── utils/
│   ├── __init__.py
│   ├── exceptions.py                # Custom exceptions
│   └── logging.py                   # Logging configuration
├── config.py                        # Configuration management
├── data_pipeline.py                 # RAG data ingestion
├── setup.py                         # Setup and validation script
└── tests/                           # Test suite
```

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Query    │───▶│  Query Router    │───▶│  Data Sources   │
│    (CLI)        │    │  (AI-powered)    │    │                 │
└─────────────────┘    └──────────────────┘    │  ┌─────────────┐ │
                                ▲               │  │    RAG      │ │
                                │               │  │(Elasticsearch)│ │
                       ┌──────────────────┐    │  └─────────────┘ │
                       │   Response       │    │  ┌─────────────┐ │
                       │  Synthesizer     │    │  │    MCP      │ │
                       │ (KimchiAssistant)│    │  │ (GitHub API)│ │
                       └──────────────────┘    │  └─────────────┘ │
                                │               └─────────────────┘
                                ▼
                       ┌──────────────────┐
                       │ Synthesized      │
                       │ Response         │
                       └──────────────────┘
```

### Key Components

- **CLI Interface** (`cli/interface.py`): User-facing command-line interface
- **Core Assistant** (`core/assistant.py`): Main `KimchiAssistant` orchestrating all operations  
- **Query Router** (`core/query_router.py`): AI-powered routing between RAG and MCP
- **Connectors**: Modular connectors for different data sources
- **Utils**: Shared utilities for logging, exceptions, and configuration

## Features

### Core Capabilities

- **Hybrid Intelligence**: Automatically determines whether to use static knowledge (RAG) or live data (MCP) based on query context
- **Private Repository Support**: Seamless access to private repositories using personal access tokens
- **Vector Search**: Elasticsearch-powered semantic search across ingested documentation
- **Live GitHub Data**: Real-time access to commits, issues, pull requests, and repository metadata
- **AI-Powered Synthesis**: OpenAI-based response generation combining multiple data sources

### Advanced Features

- **Intelligent Query Routing**: AI-powered decision making for optimal data source selection
- **Fallback Mechanisms**: Graceful degradation when services are unavailable
- **Comprehensive Error Handling**: Robust error recovery and logging
- **Async Architecture**: High-performance async/await implementation
- **Clean Resource Management**: Proper connection pooling and cleanup

## Quick Start

### Prerequisites

- Python 3.9+
- Docker (for MCP services)
- Elasticsearch Cloud account
- OpenAI API key
- GitHub Personal Access Token

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd kimchi
   ```

2. **Set up Python environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Run setup validation**
   ```bash
   python setup.py
   ```

4. **Configure environment**
   ```bash
   # Edit the generated .env file with your credentials
   nano .env
   ```

5. **Ingest repository data** (optional, for RAG functionality)
   ```bash
   python data_pipeline.py --owner elastic --repo eis-ray
   ```

6. **Start the assistant**
   ```bash
   python main.py
   ```

### Environment Configuration

The setup script creates a `.env` template file. Edit it with your credentials:

```bash
# Required: OpenAI API key for AI routing and response synthesis
OPENAI_API_KEY=sk-xxxxxxxxxxxx

# Optional but recommended: GitHub token for enhanced access
GITHUB_TOKEN=ghp_xxxxxxxxxxxx
GITHUB_PERSONAL_ACCESS_TOKEN=ghp_xxxxxxxxxxxx  # Alternative name

# Optional: Elasticsearch configuration for RAG functionality
ELASTICSEARCH_URL=http://localhost:9200
ELASTICSEARCH_INDEX=kimchi_index

# Optional: Default GitHub repository for queries
GITHUB_OWNER=your-username
GITHUB_REPO=your-repository
GITHUB_BRANCH=main

# Optional: Configuration flags
VERBOSE=true
SHOW_PROGRESS=true
```

### Minimal Configuration

For basic functionality, you only need:

```bash
OPENAI_API_KEY=sk-xxxxxxxxxxxx
```

The assistant will work in MCP-only mode for live GitHub data queries.

## Usage

### Interactive Mode

```bash
python main.py
```

Start an interactive session where you can:
- Ask questions about repositories
- Get routing explanations with `status` command
- Use `help` for example queries
- Exit with `quit` or `exit`

### Single Query Mode

```bash
python main.py "What are the recent commits?"
python main.py "How should I set up CI/CD for this project?"
```

Process a single query and exit - perfect for automation and scripting.

### Debug Mode

```bash
python main.py "your question" --debug
```

Shows detailed routing decisions, confidence scores, and error traces.

### Data Ingestion (RAG Setup)

```bash
# Ingest a specific repository for RAG
python data_pipeline.py --owner microsoft --repo vscode

# Force re-ingestion of existing data
python data_pipeline.py --force-reclone

# Update existing repository data
python data_pipeline.py --update-repo
```

## Query Types

The system automatically routes queries to appropriate data sources:

### Knowledge Queries (RAG)
- "How do I configure GPU support?"
- "What are the best practices for deployment?"
- "Explain the architecture of this system"

### Live Data Queries (MCP)
- "What are the recent commits?"
- "Show me open issues"
- "Who are the main contributors?"

### Hybrid Queries
- "Analyze the current repository state and suggest improvements"
- "Review recent changes and provide deployment guidance"

## System Components

### Query Router
- **Purpose**: Intelligent query classification and routing
- **Technology**: OpenAI GPT-4 with custom prompting
- **Function**: Determines optimal data source (RAG, MCP, or hybrid)

### RAG System
- **Vector Store**: Elasticsearch with OpenAI embeddings
- **Document Processing**: LlamaIndex with markdown and code parsers
- **Search**: Semantic similarity search with configurable ranking

### MCP Integration
- **GitHub Connector**: Docker-isolated GitHub API access
- **Real-time Data**: Live repository metadata and content
- **Authentication**: Personal access token with SSO support

### Response Synthesis
- **AI Integration**: OpenAI GPT-4 for response generation
- **Context Fusion**: Combines multiple data sources intelligently
- **Quality Control**: Confidence scoring and source attribution

## Configuration

### Advanced Settings

Modify `config.py` for advanced configuration:

```python
# Elasticsearch settings
ELASTICSEARCH_CONFIG = {
    "batch_size": 100,
    "max_retries": 3,
    "request_timeout": 30,
}

# Embedding settings
EMBEDDING_MODEL = "text-embedding-3-large"

# Query routing thresholds
ROUTING_CONFIDENCE_THRESHOLD = 0.7
```

### Custom Parsers

Add custom document parsers in `connectors/elasticsearch_connector.py`:

```python
# Add new file type support
SUPPORTED_EXTENSIONS = {
    '.py', '.js', '.ts', '.md', '.rst', '.txt',
    '.json', '.yaml', '.yml', '.toml'
}
```

## Troubleshooting

### Common Issues

1. **Authentication Errors**
   - Verify GitHub token has correct permissions
   - For SSO organizations, re-authorize the token
   - Check token expiration

2. **Elasticsearch Connection**
   - Verify cloud ID and credentials
   - Check network connectivity
   - Ensure index exists or has proper permissions

3. **OpenAI API Issues**
   - Verify API key validity
   - Check quota and billing status
   - Monitor rate limits

### Debug Mode

Enable debug logging:

```bash
python main.py --debug
```

### Performance Optimization

- Use smaller batch sizes for memory-constrained environments
- Adjust embedding model based on accuracy/speed requirements
- Configure Elasticsearch cluster for high-throughput scenarios

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement changes with tests
4. Submit a pull request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Code formatting
black kimchi/
isort kimchi/
```

## API Reference

### Core Classes

#### KimchiAssistant

Main assistant class for hybrid RAG + MCP operations.

```python
from core.assistant import KimchiAssistant

# Initialize and use
assistant = KimchiAssistant()
await assistant.initialize()

# Answer questions
response = await assistant.answer_question("What are recent commits?")
print(response["synthesized_answer"])

# Get system status  
status = assistant.get_status()
print(f"RAG available: {status['rag_available']}")
print(f"MCP available: {status['mcp_available']}")

# Cleanup when done
await assistant.cleanup()
```

#### KimchiCLI

Command-line interface for the assistant.

```python
from cli.interface import KimchiCLI

# Create CLI instance
cli = KimchiCLI()

# Run single query
await cli.run_single_query("How do I deploy this?")

# Run interactive session
await cli.run_interactive()
```

#### QueryRouter

AI-powered query routing between data sources.

```python
from core.query_router import QueryRouter, QueryType

router = QueryRouter()
decision = await router.route_query("Show me recent commits")

print(f"Route: {decision.query_type}")
print(f"Confidence: {decision.confidence}")
print(f"Reasoning: {decision.reasoning}")
```

### Configuration Classes

```python
from config import load_config

# Load application configuration
config = load_config()
print(f"GitHub owner: {config.github.owner}")
print(f"Elasticsearch index: {config.elasticsearch.index_name}")
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:
- Check the troubleshooting section
- Review the technical documentation
- Submit an issue on GitHub

---

**Note**: This system handles sensitive data including API keys and repository content. Ensure proper security practices in production deployments.
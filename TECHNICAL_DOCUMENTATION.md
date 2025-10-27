# Kimchi Technical Documentation

## Table of Contents
1. [System Architecture](#system-architecture)
2. [Code Structure and Components](#code-structure-and-components)
3. [LLM Integration and Prompt Engineering](#llm-integration-and-prompt-engineering)
4. [AI Engineering Best Practices](#ai-engineering-best-practices)
5. [Data Flow and Processing](#data-flow-and-processing)
6. [Configuration Management](#configuration-management)
7. [Error Handling and Resilience](#error-handling-and-resilience)
8. [Performance Optimization](#performance-optimization)
9. [Testing Strategy](#testing-strategy)
10. [Monitoring and Observability](#monitoring-and-observability)

## System Architecture

### Overview
Kimchi is a hybrid Retrieval-Augmented Generation (RAG) and Model Context Protocol (MCP) GitHub assistant that combines the strengths of both approaches to provide comprehensive GitHub repository assistance. The system uses intelligent routing to determine the optimal data source for each query.

### Architectural Components

```
┌─────────────────────────────────────────────────────────────┐
│                    User Interface (CLI)                     │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│              Enhanced GitHub Assistant                      │
│  ┌─────────────────┬─────────────────┬───────────────────┐  │
│  │  Query Router   │   RAG System    │   MCP Connector   │  │
│  │   (OpenAI)      │ (Elasticsearch) │   (GitHub API)    │  │
│  └─────────────────┴─────────────────┴───────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                External Services                            │
│  ┌──────────────┬──────────────┬─────────────────────────┐  │
│  │   OpenAI     │ Elasticsearch│      GitHub MCP         │  │
│  │    API       │   Cluster    │       Server            │  │
│  └──────────────┴──────────────┴─────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Design Principles

1. **Hybrid Intelligence**: Combines static knowledge (RAG) with live data (MCP)
2. **Intelligent Routing**: AI-powered decision making for optimal data source selection
3. **Graceful Degradation**: System continues to function when components are unavailable
4. **Modular Architecture**: Loosely coupled components for maintainability
5. **Context Awareness**: Maintains conversation context for coherent responses

### Core Architecture Patterns

#### 1. Strategy Pattern
The system uses the Strategy pattern for query routing, where different strategies (RAG, MCP, Hybrid) are selected based on query analysis.

#### 2. Adapter Pattern
Connectors (ElasticsearchConnector, MCPGitHubConnector) adapt external APIs to a common interface.

#### 3. Facade Pattern
The EnhancedGitHubAssistant acts as a facade, providing a simplified interface to the complex subsystems.

#### 4. Pipeline Pattern
The data ingestion process follows a pipeline pattern: Clone → Parse → Embed → Store.

## Code Structure and Components

### Core Modules

#### 1. Core Assistant (`core/assistant.py`)
**Purpose**: Main orchestrator that coordinates all system components.

**Key Responsibilities**:
- Initialize all connectors and services
- Route queries to appropriate data sources
- Synthesize responses from multiple sources
- Handle fallback scenarios

**Architecture Patterns**:
- Facade Pattern: Simplifies complex subsystem interactions
- Dependency Injection: Components are injected during initialization

```python
class KimchiAssistant:
    """AI-powered hybrid GitHub assistant with intelligent query routing."""
    
    async def initialize(self):
        """Initialize all components with proper error handling."""
        
    async def answer_question(self, question: str) -> str:
        """Main entry point for question answering."""
```

#### 2. Query Router (`core/query_router.py`)
**Purpose**: AI-powered intelligent routing system that determines optimal data sources.

**Key Features**:
- LLM-based query classification
- Confidence scoring for routing decisions
- Support for hybrid queries requiring multiple sources
- Reasoning explanations for routing decisions

**AI Engineering Approach**:
- Uses structured prompts with examples for consistent classification
- Implements confidence thresholds for reliable routing
- Provides fallback mechanisms for uncertain classifications

```python
class QueryRouter:
    """Intelligent router using OpenAI for query analysis."""
    
    async def route_query(self, query: str) -> RoutingDecision:
        """Analyze query and determine optimal routing strategy."""
```

#### 3. Elasticsearch Connector (`connectors/elasticsearch_connector.py`)
**Purpose**: Handles RAG operations including document processing, embedding generation, and vector search.

**Key Components**:
- **DocumentProcessor**: Handles multiple file types with specialized parsers
- **ElasticsearchConnector**: Manages ES operations and vector search
- **ParserConfig**: Configurable chunking and processing parameters

**AI Engineering Features**:
- Intelligent document chunking with overlap for context preservation
- Multiple embedding strategies for different content types
- Semantic search with hybrid scoring (keyword + vector)

#### 4. MCP GitHub Connector (`connectors/mcp_github_connector.py`)
**Purpose**: Interfaces with GitHub's official MCP server for real-time data access.

**Key Features**:
- Support for both local and remote MCP servers
- Comprehensive GitHub API coverage via MCP tools
- Robust connection management and retry logic
- Automatic tool discovery and capability mapping

#### 5. CLI Interface (`cli/interface.py`)
**Purpose**: Command-line interface providing both interactive and single-query modes.

**Key Features**:
- Interactive session management
- Single query processing
- Help and status commands
- Debug mode support

**Design Features**:
- Clean separation from core business logic
- User-friendly interface with colored output
- Comprehensive error handling and user guidance

#### 6. Utilities (`utils/`)
**Purpose**: Shared utilities and common functionality.

**Components**:
- **Logging** (`utils/logging.py`): Centralized logging configuration
- **Exceptions** (`utils/exceptions.py`): Custom exception hierarchy
- **Third-party Integration**: Configuration for external library logging

#### 7. Configuration Management (`config.py`)
**Purpose**: Centralized configuration with validation and environment variable management.

**Design Features**:
- Type-safe configuration using dataclasses
- Environment variable validation
- Default value management
- Configuration composition for complex setups

### Data Processing Pipeline (`data_pipeline.py`)

The data pipeline implements a robust ETL process:

1. **Extract**: Clone/update GitHub repositories
2. **Transform**: Parse documents with specialized processors
3. **Load**: Generate embeddings and store in Elasticsearch

**Key Engineering Practices**:
- Incremental processing to avoid reprocessing unchanged data
- Batch processing for efficient embedding generation
- Error recovery and resumption capabilities
- Progress tracking and monitoring

## LLM Integration and Prompt Engineering

### OpenAI Integration Strategy

#### 1. Query Routing Prompts
The system uses sophisticated prompt engineering for query classification:

```python
ROUTING_SYSTEM_PROMPT = """
You are an expert system for routing GitHub-related queries to appropriate data sources.

Available Sources:
- RAG: For documentation, tutorials, best practices (static knowledge)
- MCP: For live GitHub data (commits, issues, files, current state)
- HYBRID: For queries requiring both sources

Analyze the query and provide routing decisions with confidence scores.
"""
```

**Prompt Engineering Techniques**:
- **Few-shot Learning**: Provides examples for each query type
- **Structured Output**: Uses JSON schema for consistent responses
- **Confidence Scoring**: Enables fallback decision making
- **Reasoning Chain**: Requires explanation of routing decisions

#### 2. Response Synthesis
For hybrid queries, the system uses LLM for intelligent response synthesis:

```python
async def synthesize_response(self, rag_result: str, mcp_result: str, query: str) -> str:
    """Synthesize coherent response from multiple sources."""
```

**Synthesis Strategies**:
- **Context Weighting**: Prioritizes sources based on query type
- **Information Integration**: Combines complementary information
- **Conflict Resolution**: Handles contradictory information
- **Response Coherence**: Ensures natural, flowing responses

### Prompt Engineering Best Practices

#### 1. Structured Prompts
- Clear role definition for the AI system
- Explicit instructions with examples
- Consistent output format requirements
- Error handling instructions

#### 2. Context Management
- Conversation history preservation
- Relevant context injection
- Context window optimization
- Dynamic context pruning

#### 3. Temperature and Parameter Tuning
- Conservative temperature (0.1-0.3) for routing decisions
- Higher temperature (0.7) for creative response synthesis
- Max tokens optimization based on response type
- Stop sequences for structured outputs

## AI Engineering Best Practices

### 1. Robust Error Handling
The system implements comprehensive error handling at multiple levels:

```python
try:
    routing_decision = await self.query_router.route_query(question)
except Exception as e:
    # Fallback to default routing strategy
    routing_decision = self._get_fallback_routing(question)
```

**Error Recovery Strategies**:
- Graceful degradation when services are unavailable
- Automatic fallback to alternative data sources
- Retry mechanisms with exponential backoff
- User-friendly error messages with suggested actions

### 2. Configuration-Driven Behavior
All AI model parameters are externally configurable:

```python
@dataclass
class AIConfig:
    """AI model configuration parameters."""
    model: str = "gpt-4"
    temperature: float = 0.1
    max_tokens: int = 1000
    confidence_threshold: float = 0.7
```

### 3. Observability and Monitoring
Comprehensive logging and monitoring for AI operations:

```python
logger.info(f"Query routed to {routing_decision.query_type} with confidence {routing_decision.confidence}")
```

**Monitoring Metrics**:
- Query routing accuracy and confidence scores
- Response time for each data source
- Error rates and failure patterns
- User satisfaction indicators

### 4. Testing and Validation
Robust testing framework for AI components:

- Unit tests for individual components
- Integration tests for end-to-end workflows
- Performance tests for scalability validation
- A/B testing for prompt optimization

### 5. Prompt Version Control
Systematic management of prompt templates:

- Versioned prompt templates
- A/B testing for prompt variations
- Performance tracking for different prompt versions
- Rollback capabilities for prompt changes

## Data Flow and Processing

### Query Processing Flow

```
User Query → Query Router → Routing Decision → Data Source(s) → Response Synthesis → User
```

#### Detailed Flow:

1. **Query Reception**: User input received via CLI
2. **Query Analysis**: LLM analyzes query intent and requirements
3. **Routing Decision**: AI determines optimal data source(s)
4. **Data Retrieval**:
   - RAG: Semantic search in Elasticsearch
   - MCP: Real-time GitHub API calls
   - Hybrid: Parallel execution of both sources
5. **Response Synthesis**: LLM combines and formats results
6. **Response Delivery**: Formatted output to user

### Data Ingestion Pipeline

```
GitHub Repo → Document Parsing → Embedding Generation → Elasticsearch Storage
```

#### Pipeline Stages:

1. **Repository Cloning**: Git operations with authentication
2. **Document Discovery**: File system traversal with filtering
3. **Content Parsing**: Multiple parsers for different file types
4. **Chunking Strategy**: Intelligent text segmentation
5. **Embedding Generation**: OpenAI API calls with batching
6. **Vector Storage**: Elasticsearch indexing with metadata

### Embedding Strategy

The system uses a sophisticated embedding approach:

```python
class EmbeddingStrategy:
    """Handles different embedding approaches for different content types."""
    
    def get_embedding_config(self, content_type: str) -> EmbeddingConfig:
        """Returns optimized embedding configuration for content type."""
```

**Content-Specific Strategies**:
- **Code Files**: Syntax-aware chunking with function boundaries
- **Documentation**: Semantic chunking with section boundaries
- **Configuration**: Key-value pair preservation
- **Mixed Content**: Hybrid approach with content detection

## Configuration Management

### Hierarchical Configuration

The system uses a hierarchical configuration approach:

```
Environment Variables → Configuration Files → Default Values
```

#### Configuration Layers:

1. **Environment Variables**: Sensitive data and deployment-specific settings
2. **Configuration Files**: Structured settings with validation
3. **Default Values**: Sensible defaults for development

### Configuration Validation

```python
class ConfigManager:
    """Manages and validates application configuration."""
    
    def validate_config(self, config: AppConfig) -> None:
        """Validates configuration completeness and correctness."""
```

**Validation Features**:
- Required field checking
- Type validation
- Value range validation
- Dependency validation (e.g., API keys for enabled services)

### Dynamic Configuration

Support for runtime configuration updates:
- Hot-reloading of non-critical settings
- Graceful handling of configuration changes
- Configuration change notifications

## Error Handling and Resilience

### Multi-Level Error Handling

#### 1. Component Level
Each component implements specific error handling:

```python
class ElasticsearchConnector:
    async def search(self, query: str) -> List[Document]:
        try:
            return await self._perform_search(query)
        except ConnectionError:
            logger.warning("Elasticsearch connection failed, using cached results")
            return self._get_cached_results(query)
```

#### 2. System Level
The main assistant coordinates error recovery:

```python
async def answer_question(self, question: str) -> str:
    try:
        return await self._process_question(question)
    except Exception as e:
        return self._generate_fallback_response(question, e)
```

### Resilience Patterns

#### 1. Circuit Breaker
Prevents cascading failures by temporarily disabling failing services:

```python
class CircuitBreaker:
    """Implements circuit breaker pattern for external service calls."""
```

#### 2. Retry with Backoff
Intelligent retry mechanisms for transient failures:

```python
async def retry_with_backoff(func, max_retries=3, base_delay=1):
    """Exponential backoff retry mechanism."""
```

#### 3. Graceful Degradation
System continues functioning with reduced capabilities:

- RAG-only mode when MCP is unavailable
- MCP-only mode when Elasticsearch is down
- Cached responses when all external services fail

## Performance Optimization

### Caching Strategy

Multi-level caching for improved performance:

#### 1. Response Caching
Cache complete responses for identical queries:

```python
class ResponseCache:
    """Caches complete assistant responses."""
    
    async def get_cached_response(self, query: str) -> Optional[str]:
        """Retrieve cached response if available."""
```

#### 2. Embedding Caching
Cache embeddings to avoid recomputation:

```python
class EmbeddingCache:
    """Caches generated embeddings."""
```

#### 3. MCP Result Caching
Cache MCP API results with TTL:

```python
class MCPCache:
    """Caches MCP API results with expiration."""
```

### Batch Processing

Efficient batch processing for bulk operations:

- Embedding generation in batches
- Bulk Elasticsearch indexing
- Parallel MCP tool execution

### Connection Pooling

Optimized connection management:

- HTTP connection pooling for API calls
- Elasticsearch connection pooling
- Connection lifecycle management

## Testing Strategy

### Test Pyramid

#### 1. Unit Tests
Test individual components in isolation:

```python
class TestQueryRouter:
    """Unit tests for QueryRouter component."""
    
    async def test_route_live_data_query(self):
        """Test routing of live data queries."""
```

#### 2. Integration Tests
Test component interactions:

```python
class TestIntegration:
    """Integration tests for system components."""
    
    async def test_end_to_end_query_flow(self):
        """Test complete query processing flow."""
```

#### 3. System Tests
End-to-end testing of complete workflows:

```python
class TestSystem:
    """System-level tests."""
    
    async def test_complete_assistant_workflow(self):
        """Test complete assistant functionality."""
```

### AI-Specific Testing

#### 1. Prompt Testing
Validate prompt effectiveness:

```python
class TestPrompts:
    """Tests for prompt engineering."""
    
    def test_routing_prompt_accuracy(self):
        """Validate routing prompt classification accuracy."""
```

#### 2. Response Quality Testing
Measure response quality metrics:

- Relevance scoring
- Factual accuracy validation
- Response coherence measurement
- User satisfaction simulation

### Test Data Management

- Synthetic test data generation
- Real data anonymization
- Test data versioning
- Environment-specific test datasets

## Monitoring and Observability

### Logging Strategy

Structured logging throughout the system:

```python
import structlog

logger = structlog.get_logger()

logger.info("Query processed", 
           query_type=routing_decision.query_type,
           confidence=routing_decision.confidence,
           response_time=elapsed_time)
```

### Metrics Collection

Key performance indicators:

#### 1. System Metrics
- Response time percentiles
- Error rates by component
- Throughput measurements
- Resource utilization

#### 2. AI Metrics
- Routing accuracy
- Confidence score distributions
- Response quality scores
- User satisfaction ratings

#### 3. Business Metrics
- Query volume by type
- Feature usage patterns
- User engagement metrics
- Success rate measurements

### Health Checks

Comprehensive health monitoring:

```python
class HealthChecker:
    """Monitors system component health."""
    
    async def check_elasticsearch_health(self) -> HealthStatus:
        """Check Elasticsearch cluster health."""
    
    async def check_openai_api_health(self) -> HealthStatus:
        """Check OpenAI API availability."""
```

### Alerting

Proactive alerting for critical issues:

- Service availability alerts
- Performance degradation alerts
- Error rate threshold alerts
- Capacity planning alerts

## Deployment and Operations

### Containerization

Docker-based deployment with multi-stage builds:

```dockerfile
FROM python:3.11-slim as base
# Base image with common dependencies

FROM base as development
# Development environment with additional tools

FROM base as production
# Optimized production image
```

### Environment Management

Support for multiple deployment environments:

- Development: Local development with mocked services
- Staging: Production-like environment for testing
- Production: Optimized for performance and reliability

### Secrets Management

Secure handling of sensitive configuration:

- Environment variable injection
- Secret rotation support
- Encrypted configuration storage
- Audit logging for secret access

### Scaling Considerations

Architecture designed for horizontal scaling:

- Stateless design for easy replication
- External state storage (Elasticsearch, cache)
- Load balancing support
- Auto-scaling capabilities

## Future Enhancements

### Planned Improvements

1. **Multi-Repository Support**: Extend to handle multiple repositories simultaneously
2. **Advanced Caching**: Implement distributed caching with Redis
3. **Real-time Updates**: WebSocket support for live repository changes
4. **Enhanced AI Models**: Integration with newer LLM models and local models
5. **Custom Embeddings**: Fine-tuned embeddings for code understanding
6. **Graph-based RAG**: Knowledge graphs for better relationship understanding
7. **Collaborative Features**: Multi-user support with shared contexts
8. **Plugin Architecture**: Extensible plugin system for custom connectors

### Architectural Evolution

The system is designed to evolve while maintaining backward compatibility:

- Plugin-based architecture for easy extension
- Versioned APIs for component communication
- Migration tools for data format changes
- Feature flag system for gradual rollouts

## Conclusion

Kimchi represents a sophisticated implementation of modern AI engineering practices, combining the strengths of RAG and MCP systems with intelligent routing and robust error handling. The architecture emphasizes modularity, observability, and resilience while maintaining high performance and user experience quality.

The system demonstrates best practices in:
- AI system design and integration
- Prompt engineering and LLM optimization
- Error handling and graceful degradation
- Performance optimization and caching
- Comprehensive testing and monitoring
- Production-ready deployment practices

This technical foundation provides a solid base for future enhancements and scaling while maintaining system reliability and user satisfaction.

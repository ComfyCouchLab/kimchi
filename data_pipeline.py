"""
Kimchi Data Pipeline - RAG Data Ingestion

This module provides the data ingestion pipeline that:
1. Clones a GitHub repository using GitHubConnector
2. Processes documents and creates embeddings using ElasticsearchConnector
3. Stores the embeddings in Elasticsearch for vector search

This is separate from the main assistant and is used to populate the RAG knowledge base.
"""

import sys
import argparse
import os
from typing import Optional

from connectors import GitHubConnector, ElasticsearchConnector
from connectors.github_connector import GitHubConfig, GitHubConnectorError
from connectors.elasticsearch_connector import ElasticsearchConfig, ElasticsearchConnectorError
from config import load_config, ConfigurationError


class KimchiDataPipeline:
    """
    Data ingestion pipeline for RAG knowledge base.
    
    This class coordinates the workflow for populating Elasticsearch with GitHub repository data:
    1. Clone/update repository from GitHub
    2. Parse documents and create embeddings
    3. Store embeddings in Elasticsearch
    """
    
    def __init__(self, 
                 github_config: Optional[GitHubConfig] = None,
                 elasticsearch_config: Optional[ElasticsearchConfig] = None,
                 embedding_model: str = "text-embedding-3-large"):
        """
        Initialize the pipeline with connectors.
        
        Args:
            github_config: Configuration for GitHub operations.
            elasticsearch_config: Configuration for Elasticsearch operations.
            embedding_model: OpenAI embedding model to use.
        """
        self.github_connector = GitHubConnector(github_config)
        self.elasticsearch_connector = ElasticsearchConnector(elasticsearch_config, embedding_model)
    
    def run(self, 
            force_reclone: bool = False, 
            update_repo: bool = False,
            verbose: bool = True,
            show_progress: bool = True) -> None:
        """
        Run the complete data ingestion pipeline.
        
        Args:
            force_reclone: Force fresh clone of repository.
            update_repo: Try to update existing repository instead of clone.
            verbose: Enable verbose output during processing.
            show_progress: Show progress during ingestion.
        """
        try:
            # Step 1: Handle repository operations
            print("GitHub Repository Operations")
            print("-" * 30)
            if verbose:
                repo_info = self.github_connector.get_repository_info()
                print(f"Repository: {repo_info['owner']}/{repo_info['repo']}")
                print(f"Branch: {repo_info['branch']}")
                print(f"Local path: {repo_info['local_path']}")
                print(f"Exists locally: {repo_info['exists_locally']}")
            
            if update_repo:
                repo_path = self.github_connector.update_repository()
            else:
                # Use the connector's built-in cloning with proper path handling
                clone_url = self.github_connector.get_clone_url()
                config_info = self.github_connector.get_repository_info()
                local_path = config_info['local_path']
                
                if force_reclone and os.path.exists(local_path):
                    import shutil
                    shutil.rmtree(local_path)
                    print(f"Removed existing directory: {local_path}")
                
                success = self.github_connector.clone_repository(clone_url, local_path)
                if success:
                    repo_path = local_path
                else:
                    raise Exception("Failed to clone repository")
            
            print(f"Repository ready at: {repo_path}")
            
            # Step 2: Process documents and ingest into Elasticsearch
            print("\nDocument Processing and Elasticsearch Ingestion")
            print("-" * 50)
            if verbose:
                es_info = self.elasticsearch_connector.get_store_info()
                print(f"Elasticsearch index: {es_info['index_name']}")
                print(f"Embedding model: {es_info['embedding_model']}")
                print(f"Batch size: {es_info['batch_size']}")
            
            self.elasticsearch_connector.process_and_ingest_documents(
                repo_path=repo_path,
                show_progress=show_progress,
                verbose=verbose
            )
            
            print("\nData Ingestion Pipeline Completed Successfully")
            print("=" * 50)
            
        except (GitHubConnectorError, ElasticsearchConnectorError) as e:
            print(f"Pipeline failed: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"Unexpected error: {e}")
            sys.exit(1)
        finally:
            # Ensure Elasticsearch connection is closed
            self.elasticsearch_connector.close()
    
    def get_pipeline_info(self) -> dict:
        """
        Get information about the pipeline configuration.
        
        Returns:
            dict: Pipeline configuration information.
        """
        return {
            'github': self.github_connector.get_repository_info(),
            'elasticsearch': self.elasticsearch_connector.get_store_info()
        }


def main():
    """Main entry point for the data ingestion pipeline."""
    parser = argparse.ArgumentParser(
        description="Kimchi Data Ingestion Pipeline - Populate RAG knowledge base",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                              # Use .env configuration
  %(prog)s --force-reclone              # Force fresh repository clone
  %(prog)s --update-repo                # Update existing repository
  %(prog)s --owner microsoft --repo vscode  # Override repository
  %(prog)s --index custom-index         # Use different Elasticsearch index
        """
    )
    
    parser.add_argument(
        '--force-reclone',
        action='store_true',
        help='Force fresh clone of repository (removes existing copy)'
    )
    
    parser.add_argument(
        '--update-repo',
        action='store_true',
        help='Update existing repository instead of cloning'
    )
    
    parser.add_argument(
        '--owner',
        help='GitHub repository owner (overrides GITHUB_OWNER)'
    )
    
    parser.add_argument(
        '--repo',
        help='GitHub repository name (overrides GITHUB_REPO)'
    )
    
    parser.add_argument(
        '--branch',
        help='GitHub repository branch (overrides GITHUB_BRANCH)'
    )
    
    parser.add_argument(
        '--index',
        help='Elasticsearch index name (overrides ELASTIC_INDEX)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--no-progress',
        action='store_true',
        help='Disable progress indicators'
    )
    
    args = parser.parse_args()
    
    try:
        # Load configuration from environment
        config = load_config()
        
        # Override configuration with command line arguments
        if args.owner:
            config.github.owner = args.owner
        if args.repo:
            config.github.repo = args.repo
        if args.branch:
            config.github.branch = args.branch
        if args.index:
            config.elasticsearch.index_name = args.index
        if args.verbose:
            config.verbose = True
        if args.no_progress:
            config.show_progress = False
        
        # Create pipeline with loaded configuration
        pipeline = KimchiDataPipeline(
            github_config=config.github,
            elasticsearch_config=config.elasticsearch,
            embedding_model=config.embedding_model
        )
        
        # Print pipeline information
        print("Kimchi Data Ingestion Pipeline Starting")
        print("=" * 50)
        pipeline_info = pipeline.get_pipeline_info()
        print(f"GitHub Repository: {pipeline_info['github']['owner']}/{pipeline_info['github']['repo']}")
        print(f"Branch: {pipeline_info['github']['branch']}")
        print(f"Elasticsearch Index: {pipeline_info['elasticsearch']['index_name']}")
        print(f"Embedding Model: {pipeline_info['elasticsearch']['embedding_model']}")
        print()
        
        # Run the pipeline
        pipeline.run(
            force_reclone=args.force_reclone,
            update_repo=args.update_repo,
            verbose=config.verbose, 
            show_progress=config.show_progress
        )
        
    except ConfigurationError as e:
        print(f"Configuration error: {e}")
        print("Please check your environment variables and .env file.")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"Failed to initialize pipeline: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

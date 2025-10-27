#!/usr/bin/env python3
"""
Setup script for Kimchi - Hybrid RAG + MCP GitHub Assistant

This script helps users set up the environment and configuration for Kimchi.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path


def print_banner():
    """Print welcome banner."""
    print("=" * 60)
    print("🥬 Kimchi Setup - Hybrid GitHub Assistant")
    print("=" * 60)
    print("Setting up your intelligent GitHub assistant with:")
    print("• RAG (Retrieval Augmented Generation) for knowledge")
    print("• MCP (Model Context Protocol) for live GitHub data")
    print("• AI-powered query routing and response synthesis")
    print("=" * 60)


def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        sys.exit(1)
    print(f"✅ Python version: {sys.version.split()[0]}")


def check_docker():
    """Check if Docker is available."""
    try:
        result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Docker available: {result.stdout.strip()}")
            return True
        else:
            print("⚠️  Docker not found - MCP local server will not be available")
            return False
    except FileNotFoundError:
        print("⚠️  Docker not found - MCP local server will not be available")
        return False


def install_dependencies():
    """Install Python dependencies."""
    print("\n📦 Installing Python dependencies...")
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], 
                      check=True)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False


def create_env_file():
    """Create .env file with configuration template."""
    env_file = Path('.env')
    
    if env_file.exists():
        print("⚠️  .env file already exists, skipping creation")
        return
    
    print("\n📝 Creating .env configuration file...")
    
    env_template = """# Kimchi Configuration File
# Copy this file to .env and fill in your values

# Required: OpenAI API key for AI routing and response synthesis
OPENAI_API_KEY=your_openai_api_key_here

# Optional: GitHub token for enhanced access (recommended)
GITHUB_TOKEN=your_github_token_here
# Alternative name (either works)
GITHUB_PERSONAL_ACCESS_TOKEN=your_github_token_here

# Optional: Elasticsearch configuration for RAG
ELASTICSEARCH_URL=http://localhost:9200
ELASTICSEARCH_INDEX=kimchi_index

# Optional: GitHub repository for RAG knowledge base
GITHUB_OWNER=your_username
GITHUB_REPO=your_repository
GITHUB_BRANCH=main

# Optional: Configuration flags
VERBOSE=true
SHOW_PROGRESS=true
"""
    
    with open('.env', 'w') as f:
        f.write(env_template)
    
    print("✅ Created .env template file")
    print("📝 Please edit .env and add your API keys and configuration")


def check_configuration():
    """Check if required configuration is present."""
    from dotenv import load_dotenv
    load_dotenv()
    
    print("\n🔍 Checking configuration...")
    
    issues = []
    
    # Check OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        issues.append("OPENAI_API_KEY not set (required for AI functionality)")
    else:
        print("✅ OpenAI API key configured")
    
    # Check GitHub token (optional but recommended)
    github_token = os.getenv('GITHUB_TOKEN') or os.getenv('GITHUB_PERSONAL_ACCESS_TOKEN')
    if not github_token:
        issues.append("GITHUB_TOKEN not set (optional but recommended for enhanced access)")
    else:
        print("✅ GitHub token configured")
    
    # Check Elasticsearch (optional)
    if os.getenv('ELASTICSEARCH_URL'):
        print("✅ Elasticsearch configured (RAG will be available)")
    else:
        issues.append("ELASTICSEARCH_URL not set (RAG functionality will be limited)")
    
    if issues:
        print("\n⚠️  Configuration issues:")
        for issue in issues:
            print(f"   • {issue}")
        print("\nPlease update your .env file to resolve these issues.")
        return False
    
    print("\n✅ Configuration looks good!")
    return True


def run_test():
    """Run a quick test to verify setup."""
    print("\n🧪 Running quick test...")
    
    try:
        # Test imports
        sys.path.insert(0, str(Path.cwd()))
        
        from core.assistant import KimchiAssistant
        from core.query_router import QueryRouter
        
        print("✅ Core modules imported successfully")
        
        # Test basic initialization (without actual connections)
        assistant = KimchiAssistant()
        print("✅ Assistant initialized successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def print_usage_instructions():
    """Print usage instructions."""
    print("\n" + "=" * 60)
    print("🚀 Setup Complete! Usage Instructions:")
    print("=" * 60)
    
    print("\n1. Basic Usage:")
    print("   python main.py                    # Interactive mode")
    print("   python main.py 'Your question'   # Single query mode")
    
    print("\n2. Examples:")
    print("   python main.py 'What are recent commits?'")
    print("   python main.py 'How do I set up CI/CD?'")
    print("   python main.py 'Analyze recent changes and suggest improvements'")
    
    print("\n3. Data Ingestion (RAG setup):")
    print("   python data_pipeline.py          # Populate knowledge base")
    
    print("\n4. Run Tests:")
    print("   python -m pytest                 # Run all tests")
    print("   python tests/run_tests.py        # Alternative test runner")
    
    print("\n5. Configuration:")
    print("   • Edit .env file with your API keys")
    print("   • Start Elasticsearch for RAG functionality")
    print("   • Ensure Docker is running for local MCP server")
    
    print("\n📚 For more information, see README.md")
    print("=" * 60)


def main():
    """Main setup function."""
    print_banner()
    
    # Basic checks
    check_python_version()
    docker_available = check_docker()
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Setup failed during dependency installation")
        sys.exit(1)
    
    # Create configuration
    create_env_file()
    config_ok = check_configuration()
    
    # Test setup
    test_ok = run_test()
    
    # Show results
    print("\n" + "=" * 60)
    print("📊 Setup Summary:")
    print("=" * 60)
    print(f"Python: ✅")
    print(f"Dependencies: ✅")
    print(f"Docker: {'✅' if docker_available else '⚠️'}")
    print(f"Configuration: {'✅' if config_ok else '⚠️'}")
    print(f"Basic Test: {'✅' if test_ok else '❌'}")
    
    if config_ok and test_ok:
        print("\n🎉 Setup completed successfully!")
        print_usage_instructions()
    else:
        print("\n⚠️  Setup completed with issues.")
        print("Please review the configuration and resolve any issues before using Kimchi.")
        if not config_ok:
            print("💡 Make sure to edit the .env file with your API keys.")


if __name__ == "__main__":
    main()

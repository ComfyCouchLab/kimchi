#!/usr/bin/env python3
"""
Environment Check Tool

This script verifies that all required environment variables are properly configured
for the Elasticsearch observability tools.
"""

import os
from dotenv import load_dotenv

def check_environment():
    """Check all required environment variables."""
    
    print("🔍 ENVIRONMENT CONFIGURATION CHECK")
    print("="*60)
    
    # Load environment variables
    load_dotenv()
    
    # Required variables
    required_vars = {
        'OPENAI_API_KEY': 'OpenAI API access for embeddings',
        'ELASTIC_CLOUD_ID': 'Elasticsearch Cloud connection',
        'ELASTIC_USER': 'Elasticsearch username', 
        'ELASTIC_PASSWORD': 'Elasticsearch password',
        'ELASTIC_INDEX': 'Elasticsearch index name'
    }
    
    # Optional variables
    optional_vars = {
        'ELASTICSEARCH_DEBUG': 'Enable detailed search debugging',
        'GITHUB_TOKEN': 'GitHub access for MCP features',
        'GITHUB_OWNER': 'GitHub repository owner',
        'GITHUB_REPO': 'GitHub repository name'
    }
    
    print("\n✅ REQUIRED VARIABLES:")
    missing_required = []
    
    for var, description in required_vars.items():
        value = os.getenv(var)
        if value:
            # Show partial value for security
            display_value = value[:8] + "..." if len(value) > 8 else value
            print(f"  ✅ {var}: {display_value} ({description})")
        else:
            print(f"  ❌ {var}: MISSING ({description})")
            missing_required.append(var)
    
    print("\n🔧 OPTIONAL VARIABLES:")
    for var, description in optional_vars.items():
        value = os.getenv(var)
        if value:
            display_value = value[:8] + "..." if len(value) > 8 else value
            print(f"  ✅ {var}: {display_value} ({description})")
        else:
            print(f"  ⚪ {var}: Not set ({description})")
    
    print("\n" + "="*60)
    
    if missing_required:
        print(f"❌ CONFIGURATION INCOMPLETE")
        print(f"Missing required variables: {', '.join(missing_required)}")
        print(f"\nPlease add these to your .env file:")
        for var in missing_required:
            print(f"  {var}=your_value_here")
        return False
    else:
        print(f"✅ CONFIGURATION COMPLETE")
        print(f"All required environment variables are set!")
        return True

def test_openai_connection():
    """Test OpenAI API connection."""
    try:
        from llama_index.embeddings.openai import OpenAIEmbedding
        
        print("\n🧪 TESTING OPENAI CONNECTION...")
        embed_model = OpenAIEmbedding(model="text-embedding-3-large")
        test_embedding = embed_model.get_text_embedding("test")
        
        print(f"✅ OpenAI API working!")
        print(f"   Embedding dimensions: {len(test_embedding)}")
        print(f"   Test embedding range: [{min(test_embedding):.4f}, {max(test_embedding):.4f}]")
        return True
        
    except Exception as e:
        print(f"❌ OpenAI API error: {e}")
        return False

def test_elasticsearch_connection():
    """Test Elasticsearch connection."""
    try:
        from connectors.elasticsearch_connector import ElasticsearchConnector
        
        print("\n🧪 TESTING ELASTICSEARCH CONNECTION...")
        connector = ElasticsearchConnector()
        store = connector.connect()
        
        if store:
            print(f"✅ Elasticsearch connection working!")
            print(f"   Index: {connector.config.index_name}")
            print(f"   Cloud ID: {connector.config.cloud_id[:20]}...")
            connector.close()
            return True
        else:
            print(f"❌ Failed to connect to Elasticsearch")
            return False
            
    except Exception as e:
        print(f"❌ Elasticsearch connection error: {e}")
        return False

def main():
    """Main function to run all checks."""
    print("🚀 KIMCHI ELASTICSEARCH OBSERVABILITY - SYSTEM CHECK")
    print("="*80)
    
    # Check environment variables
    env_ok = check_environment()
    
    if not env_ok:
        print("\n🛑 Cannot proceed with connection tests due to missing configuration.")
        return
    
    # Test connections
    openai_ok = test_openai_connection()
    es_ok = test_elasticsearch_connection()
    
    # Summary
    print("\n" + "="*80)
    print("📋 SYSTEM CHECK SUMMARY:")
    print(f"  Environment Variables: {'✅ OK' if env_ok else '❌ FAIL'}")
    print(f"  OpenAI API:           {'✅ OK' if openai_ok else '❌ FAIL'}")
    print(f"  Elasticsearch:        {'✅ OK' if es_ok else '❌ FAIL'}")
    
    if env_ok and openai_ok and es_ok:
        print(f"\n🎉 ALL SYSTEMS GO!")
        print(f"Your observability tools are ready to use.")
        print(f"\nNext steps:")
        print(f"  python elasticsearch_debug.py --query 'your query'")
        print(f"  python elasticsearch_debug.py --demo")
        print(f"  python elasticsearch_debug.py --interactive")
    else:
        print(f"\n⚠️  SOME ISSUES FOUND")
        print(f"Please fix the failing checks before using the observability tools.")

if __name__ == "__main__":
    main()

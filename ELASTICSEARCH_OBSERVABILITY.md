# Elasticsearch Observability Guide 🔍

This guide explains how to use the new observability tools to understand exactly what's happening with your Elasticsearch queries and improve search accuracy.

## 🚀 Quick Start

### 1. Check Your Environment
First, verify that all required environment variables are properly configured:
```bash
python check_environment.py
```

This will check:
- ✅ OpenAI API key configuration
- ✅ Elasticsearch connection settings  
- ✅ All required environment variables
- 🧪 Test actual API connections

### 2. Enable Debug Mode
Add this to your `.env` file:
```bash
ELASTICSEARCH_DEBUG=true
```

### 3. Run Query Analysis
```bash
# Analyze a specific query
python elasticsearch_debug.py --query "What's EIS?"

# Compare two queries
python elasticsearch_debug.py --compare "What's EIS?" "Elastic Inference Service"

# Analyze your index structure
python elasticsearch_debug.py --analyze-index

# Run comprehensive demo
python elasticsearch_debug.py --demo

# Interactive mode
python elasticsearch_debug.py --interactive
```

## 🔬 What You Can Observe

### 1. Query Embedding Analysis
- **Embedding dimensions**: Shows the vector size (should be 3072 for text-embedding-3-large)
- **Embedding range**: Min/max values in the vector
- **L2 Norm**: Should be 1.0000 for normalized embeddings
- **First 10 values**: Sample of the embedding vector

### 2. Search Performance
- **Execution time**: How long the search takes
- **Results count**: Number of documents returned
- **Score distribution**: Best, worst, and average similarity scores

### 3. Result Quality
- **Content analysis**: Length distribution of returned documents
- **Metadata inspection**: What fields are available
- **Result preview**: Sample content from top matches

### 4. Query Comparison
- **Embedding similarity**: How similar two queries are in vector space
- **Result overlap**: Percentage of documents that appear in both result sets
- **Performance differences**: Execution time comparison

## 🚀 Improving Search Accuracy

### Common Issues and Solutions

#### 1. **Low Similarity Scores (< 0.5)**
```
📉 Low similarity scores - results may not be very relevant
```
**Solutions:**
- Try more specific keywords
- Use synonyms or alternative terms
- Check if your documents contain the expected content

#### 2. **Slow Query Execution (> 1000ms)**
```
⏱️  Slow query execution - consider optimizing index or reducing result count
```
**Solutions:**
- Reduce the number of results requested (`k` parameter)
- Consider index optimization
- Check network connectivity to Elasticsearch

#### 3. **No Results Found**
```
❌ No results found - try broader or different keywords
```
**Solutions:**
- Use broader terms
- Check spelling
- Try related concepts or synonyms

#### 4. **Few Results (< 3)**
```
⚠️  Few results found - consider expanding the query or checking index content
```
**Solutions:**
- Add more context to your query
- Use alternative phrasing
- Check what documents are actually in your index

## 📊 Understanding Scores

### Score Interpretation
- **1.0**: Perfect match (exact or very close semantic match)
- **0.8-0.9**: High relevance
- **0.5-0.8**: Moderate relevance
- **< 0.5**: Low relevance (may not be useful)

### Your EIS Query Analysis
Based on your "What's EIS?" query:
- ✅ **Perfect match found** (score: 1.0) - the main README.md
- ✅ **Good secondary matches** (scores: 0.43-0.46) - related content
- ⚠️ **Performance issue** - 1.8s execution time is slow

## 🎯 Query Optimization Tips

### 1. **Be Specific but Not Too Narrow**
```bash
# Too broad
"service"

# Good
"Elastic Inference Service"

# Too narrow (might miss results)
"eis-ray deployment configuration with GPU support"
```

### 2. **Use Domain-Specific Terms**
```bash
# Generic
"machine learning"

# Specific to your domain
"inference serving Ray framework"
```

### 3. **Try Multiple Phrasings**
```bash
# Compare these to see which works better
python elasticsearch_debug.py --compare "What is EIS?" "Elastic Inference Service explanation"
```

## 🔧 Advanced Analysis

### Export Analysis Results
The tool automatically exports detailed analysis to JSON files:
```bash
elasticsearch_query_analysis_YYYYMMDD_HHMMSS.json
```

### Custom Analysis
You can modify the `elasticsearch_debug.py` script to:
- Test specific embeddings
- Analyze query patterns
- Compare different embedding models
- Monitor performance over time

## 🐛 Troubleshooting

### Common Error Messages

#### "Vector store not initialized"
- Ensure your Elasticsearch credentials are correct
- Check network connectivity
- Verify the index exists

#### "No OpenAI API key"
- Set `OPENAI_API_KEY` in your `.env` file
- Check API key permissions

#### "Index not found"
- Verify the index name in your configuration
- Check if documents have been ingested

## 📈 Monitoring Search Quality

### Regular Health Checks
Run these commands periodically:

```bash
# Check index health
python elasticsearch_debug.py --analyze-index

# Test key queries
python elasticsearch_debug.py --query "your important query"

# Compare query variations
python elasticsearch_debug.py --demo
```

### Performance Baselines
- **Good execution time**: < 500ms
- **Acceptable execution time**: 500ms - 1000ms  
- **Slow execution time**: > 1000ms (needs optimization)

### Quality Metrics
- **Excellent**: Top result score > 0.8
- **Good**: Top result score 0.5 - 0.8
- **Poor**: Top result score < 0.5

## 🎯 Next Steps

1. **Enable debug mode** in your `.env` file
2. **Test your key queries** with the analysis tool
3. **Compare query variations** to find the best phrasing
4. **Monitor performance** regularly
5. **Optimize based on insights** from the analysis

## 📞 Getting Help

If you see unexpected results:
1. Check the embedding analysis for anomalies
2. Examine the sample documents to understand your data
3. Compare with known good queries
4. Look at the detailed logs for error messages

Happy searching! 🚀

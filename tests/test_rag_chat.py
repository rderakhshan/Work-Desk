"""
Test script to verify RAG LAB chat functionality.

This script:
1. Checks if the vector database has any data
2. Tests document retrieval
3. Verifies LLM connectivity (Ollama)
4. Tests end-to-end chat flow
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ai_workdesk.rag.vector_store import VectorStoreManager
from ai_workdesk.tools.llm.ollama_client import OllamaClient
from ai_workdesk.core.config import get_settings
from loguru import logger

def test_database_status():
    """Test 1: Check if database has data."""
    print("\n" + "="*60)
    print("TEST 1: Database Status")
    print("="*60)
    
    try:
        settings = get_settings()
        embedding_provider = "ollama" if getattr(settings, "ollama_embedding_model", None) else "huggingface"
        
        vector_store = VectorStoreManager(embedding_provider=embedding_provider)
        stats = vector_store.get_stats()
        
        print(f"✅ Vector Store Status: {stats.get('status')}")
        print(f"📊 Total Chunks: {stats.get('total_chunks', 0)}")
        print(f"🗄️  Database: {stats.get('vector_db')}")
        
        if stats.get('total_chunks', 0) == 0:
            print("\n⚠️  WARNING: Database is EMPTY! No documents to chat with.")
            print("   Please ingest some documents first in the Embedding LAB.")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error checking database: {e}")
        return False

def test_retrieval(vector_store):
    """Test 2: Test document retrieval."""
    print("\n" + "="*60)
    print("TEST 2: Document Retrieval")
    print("="*60)
    
    try:
        test_query = "test"
        print(f"🔍 Testing retrieval with query: '{test_query}'")
        
        docs = vector_store.similarity_search(
            query=test_query,
            k=3,
            score_threshold=0.0  # Accept any similarity
        )
        
        print(f"✅ Retrieved {len(docs)} documents")
        
        if docs:
            print("\n📄 Sample retrieved document:")
            print(f"   Content: {docs[0].page_content[:200]}...")
            print(f"   Metadata: {docs[0].metadata}")
            return True
        else:
            print("⚠️  No documents retrieved (this might be okay if database is empty)")
            return False
            
    except Exception as e:
        print(f"❌ Error during retrieval: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ollama_connection():
    """Test 3: Test Ollama LLM connectivity."""
    print("\n" + "="*60)
    print("TEST 3: Ollama LLM Connection")
    print("="*60)
    
    try:
        print("🔌 Connecting to Ollama...")
        client = OllamaClient(model="deepseek-r1:7b", temperature=0.7, max_tokens=100)
        
        test_prompt = "Say 'Hello, I am working!' in one sentence."
        print(f"💬 Testing with prompt: '{test_prompt}'")
        
        response = client.chat(test_prompt)
        print(f"✅ Ollama Response: {response[:200]}...")
        return True
        
    except Exception as e:
        print(f"❌ Error connecting to Ollama: {e}")
        print("   Make sure Ollama is running: 'ollama serve'")
        print("   And the model is available: 'ollama pull deepseek-r1:7b'")
        return False

def test_end_to_end_chat(vector_store):
    """Test 4: End-to-end RAG chat."""
    print("\n" + "="*60)
    print("TEST 4: End-to-End RAG Chat")
    print("="*60)
    
    try:
        # Step 1: Retrieve context
        query = "What is this about?"
        print(f"🔍 Query: '{query}'")
        
        docs = vector_store.similarity_search(query=query, k=3, score_threshold=0.0)
        
        if not docs:
            print("⚠️  No documents retrieved - skipping LLM call")
            return False
        
        print(f"✅ Retrieved {len(docs)} documents")
        
        # Step 2: Build context
        context = "\n\n".join([f"[Doc {i+1}]\n{doc.page_content}" for i, doc in enumerate(docs)])
        
        # Step 3: Create prompt
        system_prompt = "You are a helpful AI assistant. Answer based on the provided context."
        full_prompt = f"""{system_prompt}

Context from knowledge base:
{context}

User question: {query}

Answer the question based on the context above."""
        
        print(f"\n📝 Prompt length: {len(full_prompt)} characters")
        
        # Step 4: Call LLM
        print("🤖 Calling Ollama...")
        client = OllamaClient(model="deepseek-r1:7b", temperature=0.7, max_tokens=200)
        response = client.chat(full_prompt)
        
        print(f"\n✅ RAG Response:\n{response}\n")
        return True
        
    except Exception as e:
        print(f"❌ Error in end-to-end test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("\n🧪 RAG LAB Chat Functionality Test Suite")
    print("="*60)
    
    results = {}
    
    # Test 1: Database Status
    results['database'] = test_database_status()
    
    if not results['database']:
        print("\n⚠️  Cannot proceed with further tests - database is empty or inaccessible")
        return
    
    # Initialize vector store for remaining tests
    settings = get_settings()
    embedding_provider = "ollama" if getattr(settings, "ollama_embedding_model", None) else "huggingface"
    vector_store = VectorStoreManager(embedding_provider=embedding_provider)
    
    # Test 2: Retrieval
    results['retrieval'] = test_retrieval(vector_store)
    
    # Test 3: Ollama
    results['ollama'] = test_ollama_connection()
    
    # Test 4: End-to-End
    if results['retrieval'] and results['ollama']:
        results['e2e'] = test_end_to_end_chat(vector_store)
    else:
        results['e2e'] = False
        print("\n⚠️  Skipping end-to-end test due to previous failures")
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name.upper()}")
    
    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 All tests passed! RAG LAB chat is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")

if __name__ == "__main__":
    main()

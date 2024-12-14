# product_review_agent.py
import os
import logging
import json
from typing import Dict, List, Optional
from langchain_anthropic import ChatAnthropic
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd
from functools import lru_cache
import re
import shutil
import warnings
from dotenv import load_dotenv
from pathlib import Path
import nltk
from pymilvus import (
    connections, 
    utility,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType
)
import redis
import atexit
from datetime import datetime, timedelta
import time

# Configure logging based on .env
load_dotenv()
logging.basicConfig(level=os.getenv('LOG_LEVEL', 'INFO'))
logger = logging.getLogger(__name__)

# Environment setup
def setup_environment():
    """Initialize environment and dependencies"""
    try:
        nltk.download('punkt_tab')
        
        # Validate required environment variables
        required_vars = ['ANTHRO_KEY', 'OA_API']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
        # Set API keys
        os.environ['ANTHROPIC_API_KEY'] = os.getenv('ANTHRO_KEY')
        os.environ['OPENAI_API_KEY'] = os.getenv('OA_API')
        
        # SQLite3 configuration
        __import__('pysqlite3')
        import sys
        sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
        
    except Exception as e:
        logger.error(f"Environment setup failed: {str(e)}")
        raise


def get_service_connections():
    """Get Redis and Milvus connections from the global connection pool"""
    redis_client = None
    milvus_conn = None
    
    try:
        # Get Redis connection with retry logic
        redis_host = os.getenv('REDIS_HOST', 'localhost')
        redis_port = int(os.getenv('REDIS_PORT', 6379))
        redis_password = os.getenv('REDIS_PASSWORD', None)
        max_retries = 3
        retry_delay = 1  # seconds
        
        for attempt in range(max_retries):
            try:
                redis_client = redis.Redis(
                    host=redis_host,
                    port=redis_port,
                    password=redis_password if redis_password else None,
                    db=0,
                    decode_responses=True,
                    socket_timeout=5,
                    socket_connect_timeout=5,
                    retry_on_timeout=True,
                    health_check_interval=30
                )
                # Test the connection
                redis_client.ping()
                break
            except redis.ConnectionError as e:
                if attempt == max_retries - 1:
                    raise ConnectionError(f"Failed to connect to Redis after {max_retries} attempts: {str(e)}")
                time.sleep(retry_delay)

        # Get Milvus connection with connection pooling
        if not connections.has_connection("default"):
            milvus_host = os.getenv('MILVUS_HOST', 'localhost')
            milvus_port = int(os.getenv('MILVUS_PORT', 19530))
            
            connections.connect(
                alias="default",
                host=milvus_host,
                port=milvus_port,
                pool_size=10,  # Connection pool size
                timeout=300,    # Connection timeout in seconds
                retry_on_error=True,
                max_retries=3
            )
        
        milvus_conn = connections.get_connection("default")
        
        # Verify Milvus connection
        if not utility.is_connected():
            raise ConnectionError("Failed to establish Milvus connection")
            
        return redis_client, milvus_conn
        
    except Exception as e:
        # Clean up resources if an error occurs
        if redis_client:
            try:
                redis_client.close()
            except:
                pass
                
        if milvus_conn and connections.has_connection("default"):
            try:
                connections.disconnect("default")
            except:
                pass
                
        logger.error(f"Error getting service connections: {str(e)}")
        raise
        
    finally:
        
        def cleanup_connections():
            if redis_client:
                try:
                    redis_client.close()
                except:
                    pass
                    
            if connections.has_connection("default"):
                try:
                    connections.disconnect("default")
                except:
                    pass
                    
        atexit.register(cleanup_connections)


class ProductReviewAgent:
    def __init__(self, model_name="claude-3-5-sonnet-20240620"):
        self.llm = ChatAnthropic(
            model=model_name,
            max_tokens=300,
            temperature=0.7
        )
        self.embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        
        # Data paths from environment variables
        self.redis_data_path = Path(os.getenv('REDIS_DATA_PATH', './data/redis-setup'))
        self.milvus_data_path = Path(os.getenv('MILVUS_DATA_PATH', './data/milvus-setup'))

        # Pre-compile patterns once during initialization
        self.patterns = {
            'price': re.compile(r'price|cost|how\s+much', re.I),
            'review': re.compile(r'review|rating|experience', re.I),
            'spec': re.compile(r'spec|feature|detail|technical', re.I)
        }
        # Pre-define enhancement keywords
        self.enhancements = {
            'price': ' price cost',
            'review': ' review rating',
            'spec': ' specifications features'
        }

        # Load system prompt from external file for better maintainability
        self.system_prompt = self._load_system_prompt()
        self.collection_name = "product_reviews"
        self.dim = 1536  # Dimension for OpenAI embeddings
        self.initialize_vectorstore()


    def _load_system_prompt(self) -> str:
        """Load system prompt from file or Redis cache"""
        try:
            prompt_path = Path('config/system_prompt.txt')
            if prompt_path.exists():
                return prompt_path.read_text().strip()

        except (FileNotFoundError, IOError) as e:
            logger.error(f"Failed to load system prompt: {str(e)}")
            raise


    @lru_cache(maxsize=1000)
    def _enhance_query(self, query: str) -> str:
        """Enhance query with cached results and early returns"""

        if not query:
            return query

        for intent, pattern in self.patterns.items():
            if pattern.search(query):
                return query + self.enhancements[intent]
        return query  # Return original if no pattern matches


    def initialize_vectorstore(self):
        """Initialize vector store with product data"""
        try:
            # Define the collection schema
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=1000),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=65535)
            ]
        
            schema = CollectionSchema(
                fields=fields,
                description="Product reviews collection"
            )

            # Drop existing collection if it exists
            if utility.has_collection(self.collection_name):
                utility.drop_collection(self.collection_name)

            # Create new collection
            self.collection = Collection(
                name=self.collection_name,
                schema=schema,
                using='default'
            )
        
            # Load and process data
            file_path = Path('data/cleaned_dataset_full.csv')
            if not file_path.exists():
                raise FileNotFoundError(f"Dataset not found at {file_path}")

            df = pd.read_csv(file_path)

            df['combined'] = df.apply(
                lambda row: ' '.join(f"{col}: {val}" for col, val in row.items()), 
                axis=1
            )
            # Combine all rows into a single text with clear product boundaries

            doc_metadata = {
                "source_file": str(file_path),
                "total_products": len(df),
                "data_columns": list(df.columns)
            }

            # Create documents from combined text
            documents = [Document(page_content=text) for text in df['combined']]  

            # Use RecursiveCharacterTextSplitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000,
                chunk_overlap=200,
                length_function=len
            )
        
            doc_chunks = text_splitter.split_documents(documents)
            print("Number of chunks created:", len(doc_chunks))
        
            # Prepare data for insertion
            data = []
            
            for i, chunk in enumerate(doc_chunks):
                embedding = self.embeddings.embed_query(chunk.page_content)
                titles = re.findall(r"Title: (.+?)\n", chunk.page_content)
            
                chunk_metadata = {
                    **doc_metadata,
                    "chunk_id": i,
                    "total_chunks": len(doc_chunks),
                    "titles": titles,
                    "products_count": len(titles)
                }

                data.append({
                    "title": titles[0] if titles else "",
                    "embedding": embedding,
                    "text": chunk.page_content,
                    "metadata": json.dumps(chunk_metadata)
                })

      
            # Insert data first
            self.collection.insert(data)
        
            # Create index after data insertion
            index_params = {
                "metric_type": "L2",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024}
            }
            self.collection.create_index(
                field_name="embedding",
                index_params=index_params
            )
        
            # Load the collection into memory
            self.collection.load()
            
        except Exception as e:
            logger.error(f"Error initializing vectorstore: {str(e)}")
            raise


    def process_review_query(self, state: Dict, config: dict) -> Dict:
        """Process product review queries"""
        try:
            messages = state["messages"]
            query = messages[-1].content
            enhanced_query = self._enhance_query(query)
            thread_id = config["configurable"]["thread_id"]
            
            logger.info(f"Processing review query for thread {thread_id}")
            
            # Retrieve relevant documents
            # retriever = self.vectorstore.as_retriever(
            #     search_type="mmr", 
            #     search_kwargs={"k": 2, "fetch_k": 5}
            # )

            # 1. Simplify retrieval first
            retriever = self.vectorstore.as_retriever(
                search_type="similarity",  # Changed from MMR to simple similarity
                search_kwargs={"k": 2}     # Reduced number of results
            )

            results = retriever.invoke(enhanced_query)
            
            # if not results:
            #     return {"error": "No relevant information found"}

            # 2. Add basic context compression  
            context = "\n\n".join([doc.page_content for doc in results])
            compressed_context = context[:3000]  # Simple length-based compression

            # Format messages with system prompt
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=self._format_review_prompt(query, compressed_context))
            ]
            response = self.llm.invoke(messages)
            
            return {
                "review_response": response.content,
                "thread_id": thread_id
            }
            
        except Exception as e:
            logger.error(f"Error processing review query: {e}")
            return {"error": str(e)}


    def _format_review_prompt(self, query: str, context: str) -> str:
        """Format the prompt for review processing"""
        return f"""
        Context from product database:
        {context}
        
        User Query:
        {query}
        """

def setup_product_review_agent() -> ProductReviewAgent:
    """Setup and return the product review agent"""
    try:
        # Initialize the environment first
        setup_environment()
        
        # Create and return the agent
        return ProductReviewAgent()
        
    except Exception as e:
        logger.error(f"Error setting up product review agent: {str(e)}")
        raise

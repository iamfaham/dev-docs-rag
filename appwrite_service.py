import os
from dotenv import load_dotenv
from appwrite.client import Client
from appwrite.services.databases import Databases
from appwrite.services.storage import Storage
from appwrite.input_file import InputFile
import json
import logging
from typing import List, Dict, Any, Optional
import tempfile
import time

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AppwriteService:
    def __init__(self):
        """Initialize Appwrite client and services"""
        # Validate required environment variables
        self._validate_environment()

        self.client = Client()

        # Set up client with environment variables
        self.client.set_endpoint(
            os.getenv("APPWRITE_ENDPOINT", "https://cloud.appwrite.io/v1")
        )
        self.client.set_project(os.getenv("APPWRITE_PROJECT_ID"))
        self.client.set_key(os.getenv("APPWRITE_API_KEY"))

        # Initialize services
        self.databases = Databases(self.client)
        self.storage = Storage(self.client)

        # Database and collection IDs
        self.database_id = os.getenv("APPWRITE_DATABASE_ID", "react_docs_db")
        self.chunks_collection_id = os.getenv(
            "APPWRITE_COLLECTION_ID", "document_chunks"
        )
        self.completion_collection_id = "completion_status"
        self.bucket_id = os.getenv("APPWRITE_BUCKET_ID", "react_docs_bucket")

        # Initialize database and storage if they don't exist
        self._initialize_database()
        self._initialize_storage()

    def _validate_environment(self):
        """Validate that required environment variables are set"""
        required_vars = ["APPWRITE_PROJECT_ID", "APPWRITE_API_KEY"]

        missing_vars = []
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)

        if missing_vars:
            error_msg = (
                f"Missing required environment variables: {', '.join(missing_vars)}"
            )
            logger.error(error_msg)
            logger.error("Please set these variables in your .env file:")
            for var in missing_vars:
                logger.error(f"  {var}=your_value_here")
            raise ValueError(error_msg)

    def _initialize_database(self):
        """Initialize database and chunks collection if they don't exist"""
        try:
            # Check if database exists
            try:
                self.databases.get(database_id=self.database_id)
                logger.info(f"Database {self.database_id} already exists")
            except Exception:
                # Create database
                self.databases.create(
                    database_id=self.database_id, name="React Documentation Database"
                )
                logger.info(f"Created database {self.database_id}")

            # Initialize chunks collection
            self._initialize_chunks_collection()

            # Initialize completion status collection
            self._initialize_completion_collection()

        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            raise

    def _initialize_storage(self):
        """Check if storage bucket exists (don't create if it doesn't)"""
        try:
            # Check if bucket exists
            try:
                self.storage.get_bucket(bucket_id=self.bucket_id)
                logger.info(f"Storage bucket {self.bucket_id} exists and is accessible")
            except Exception as e:
                logger.error(
                    f"Storage bucket {self.bucket_id} not found or not accessible: {str(e)}"
                )
                logger.error(
                    "Please make sure the bucket exists and your API key has access to it"
                )
                raise

        except Exception as e:
            logger.error(f"Error checking storage bucket: {str(e)}")
            raise

    def _initialize_chunks_collection(self):
        """Initialize chunks collection"""
        try:
            # Check if chunks collection exists
            try:
                self.databases.get_collection(
                    database_id=self.database_id,
                    collection_id=self.chunks_collection_id,
                )
                logger.info(
                    f"Chunks collection {self.chunks_collection_id} already exists"
                )
            except Exception:
                # Create chunks collection
                self.databases.create_collection(
                    database_id=self.database_id,
                    collection_id=self.chunks_collection_id,
                    name="Document Chunks",
                )

                # Create attributes for the chunks collection
                self.databases.create_string_attribute(
                    database_id=self.database_id,
                    collection_id=self.chunks_collection_id,
                    key="content",
                    size=65536,  # 64KB for content
                    required=True,
                )

                self.databases.create_string_attribute(
                    database_id=self.database_id,
                    collection_id=self.chunks_collection_id,
                    key="title",
                    size=255,
                    required=True,
                )

                self.databases.create_string_attribute(
                    database_id=self.database_id,
                    collection_id=self.chunks_collection_id,
                    key="url",
                    size=500,
                    required=False,
                )

                self.databases.create_string_attribute(
                    database_id=self.database_id,
                    collection_id=self.chunks_collection_id,
                    key="chunk_id",
                    size=100,
                    required=True,
                )

                logger.info(
                    f"Created chunks collection {self.chunks_collection_id} with attributes"
                )

        except Exception as e:
            logger.error(f"Error initializing chunks collection: {str(e)}")
            raise

    def _initialize_completion_collection(self):
        """Initialize completion status collection"""
        try:
            # Check if completion collection exists
            try:
                self.databases.get_collection(
                    database_id=self.database_id,
                    collection_id=self.completion_collection_id,
                )
                logger.info(
                    f"Completion collection {self.completion_collection_id} already exists"
                )
            except Exception:
                # Create completion collection
                self.databases.create_collection(
                    database_id=self.database_id,
                    collection_id=self.completion_collection_id,
                    name="Completion Status",
                )

                # Create attributes for the completion collection
                self.databases.create_string_attribute(
                    database_id=self.database_id,
                    collection_id=self.completion_collection_id,
                    key="url",
                    size=500,
                    required=True,
                )

                self.databases.create_string_attribute(
                    database_id=self.database_id,
                    collection_id=self.completion_collection_id,
                    key="status",
                    size=50,
                    required=True,
                )

                self.databases.create_string_attribute(
                    database_id=self.database_id,
                    collection_id=self.completion_collection_id,
                    key="completed_at",
                    size=100,
                    required=True,
                )

                self.databases.create_integer_attribute(
                    database_id=self.database_id,
                    collection_id=self.completion_collection_id,
                    key="chunks_count",
                    required=True,
                )

                logger.info(
                    f"Created completion collection {self.completion_collection_id} with attributes"
                )

        except Exception as e:
            logger.error(f"Error initializing completion collection: {str(e)}")
            raise

    def get_docs_file_id(self, url: str) -> str:
        """Generate file ID based on the documentation URL"""
        url_lower = url.lower()

        # Map URLs to file IDs
        if "react.dev" in url_lower or "reactjs.org" in url_lower:
            return "react_docs_raw.json"
        elif "docs.python.org" in url_lower or "python.org" in url_lower:
            return "python_docs_raw.json"
        elif "golang.org" in url_lower or "go.dev" in url_lower:
            return "golang_docs_raw.json"
        elif "developer.mozilla.org" in url_lower or "mdn" in url_lower:
            return "mdn_docs_raw.json"
        elif "vuejs.org" in url_lower:
            return "vue_docs_raw.json"
        elif "nodejs.org" in url_lower:
            return "nodejs_docs_raw.json"
        elif "angular.io" in url_lower:
            return "angular_docs_raw.json"
        elif "svelte.dev" in url_lower:
            return "svelte_docs_raw.json"
        elif "nextjs.org" in url_lower:
            return "nextjs_docs_raw.json"
        elif "nuxt.com" in url_lower:
            return "nuxt_docs_raw.json"
        elif "djangoproject.com" in url_lower or "django" in url_lower:
            return "django_docs_raw.json"
        elif "fastapi.tiangolo.com" in url_lower or "fastapi" in url_lower:
            return "fastapi_docs_raw.json"
        elif "docs.docker.com" in url_lower or "docker.com" in url_lower:
            return "docker_docs_raw.json"
        elif "kubernetes.io" in url_lower:
            return "kubernetes_docs_raw.json"
        elif "docs.mongodb.com" in url_lower or "mongodb.com" in url_lower:
            return "mongodb_docs_raw.json"
        elif "postgresql.org" in url_lower or "postgresql" in url_lower:
            return "postgresql_docs_raw.json"
        else:
            # For unknown URLs, create a generic ID based on domain
            from urllib.parse import urlparse

            parsed = urlparse(url)
            domain = parsed.netloc.replace(".", "_").replace("www_", "")
            return f"{domain}_docs_raw.json"

    def docs_already_exist(self, url: str) -> bool:
        """Check if documentation for this URL already exists in storage"""
        try:
            file_id = self.get_docs_file_id(url)
            # Try to get the file from storage
            self.storage.get_file(bucket_id=self.bucket_id, file_id=file_id)
            logger.info(f"Documentation already exists for {url} (file: {file_id})")
            return True
        except Exception as e:
            logger.info(f"Documentation does not exist for {url}: {str(e)}")
            return False

    def save_raw_docs_to_storage(
        self, docs: List[Dict[str, Any]], url: str = None
    ) -> bool:
        """Save raw documents as JSON file to Appwrite storage bucket"""
        temp_file_path = None
        max_retries = 3
        retry_delay = 2  # seconds

        for attempt in range(max_retries):
            try:
                logger.info(
                    f"Saving {len(docs)} raw documents to Appwrite storage (attempt {attempt + 1}/{max_retries})"
                )

                # Generate file ID based on URL
                file_id = self.get_docs_file_id(url) if url else "unknown_docs_raw.json"
                logger.info(f"Using file ID: {file_id}")

                # Create JSON content
                json_content = json.dumps(docs, indent=2, ensure_ascii=False)

                # Create temporary file with a unique name
                temp_file_path = tempfile.mktemp(suffix=".json")

                # Write content to temporary file
                with open(temp_file_path, "w", encoding="utf-8") as temp_file:
                    temp_file.write(json_content)

                # Upload file to storage bucket
                input_file = InputFile.from_path(temp_file_path)

                # Try to delete existing file first, then create new one
                try:
                    # Try to delete existing file
                    self.storage.delete_file(bucket_id=self.bucket_id, file_id=file_id)
                    logger.info(f"Deleted existing file: {file_id}")
                except Exception as e:
                    # File doesn't exist or can't be deleted, that's okay
                    logger.info(
                        f"Could not delete existing file (may not exist): {str(e)}"
                    )

                # Upload to storage with retry logic
                result = self.storage.create_file(
                    bucket_id=self.bucket_id,
                    file_id=file_id,
                    file=input_file,
                )

                logger.info(
                    f"Successfully saved raw documents to storage: {result['$id']}"
                )
                return True

            except Exception as e:
                logger.error(
                    f"Error saving raw documents to storage (attempt {attempt + 1}/{max_retries}): {str(e)}"
                )

                # Clean up temporary file on error
                if temp_file_path and os.path.exists(temp_file_path):
                    try:
                        os.unlink(temp_file_path)
                        temp_file_path = None
                    except (OSError, PermissionError) as cleanup_error:
                        logger.warning(
                            f"Could not delete temporary file {temp_file_path}: {str(cleanup_error)}"
                        )

                # If this is the last attempt, return False
                if attempt == max_retries - 1:
                    logger.error(
                        f"Failed to save raw documents after {max_retries} attempts"
                    )
                    return False

                # Wait before retrying
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff

        return False

    def get_raw_docs_from_storage(self, url: str = None) -> List[Dict[str, Any]]:
        """Retrieve raw documents from Appwrite storage bucket"""
        max_retries = 3
        retry_delay = 2  # seconds

        for attempt in range(max_retries):
            try:
                logger.info(
                    f"Retrieving raw documents from Appwrite storage (attempt {attempt + 1}/{max_retries})"
                )

                # Generate file ID based on URL
                file_id = self.get_docs_file_id(url) if url else "react_docs_raw.json"
                logger.info(f"Looking for file: {file_id}")

                # Download file from storage
                result = self.storage.get_file_download(
                    bucket_id=self.bucket_id, file_id=file_id
                )

                logger.info(f"Download result type: {type(result)}")

                # Handle different possible return types
                docs = None

                # Case 1: Result is already a list of dicts (JSON content)
                if isinstance(result, list) and result and isinstance(result[0], dict):
                    docs = result
                    logger.info("Result is already a list of documents")

                # Case 2: Result is bytes
                elif isinstance(result, bytes):
                    json_content = result.decode("utf-8")
                    docs = json.loads(json_content)
                    logger.info("Result is bytes, decoded successfully")

                # Case 3: Result is a list of bytes
                elif (
                    isinstance(result, list) and result and isinstance(result[0], bytes)
                ):
                    json_bytes = b"".join(result)
                    json_content = json_bytes.decode("utf-8")
                    docs = json.loads(json_content)
                    logger.info("Result is list of bytes, joined and decoded")

                # Case 4: Result is a single dict
                elif isinstance(result, dict):
                    docs = [result]
                    logger.info("Result is a single document dict")

                # Case 5: Try to convert to string and parse
                else:
                    try:
                        json_str = str(result)
                        docs = json.loads(json_str)
                        logger.info("Result converted to string and parsed")
                    except Exception as e:
                        logger.error(f"Failed to parse result: {str(e)}")
                        raise ValueError(
                            f"Could not parse downloaded file content: {str(e)}"
                        )

                if docs is None:
                    raise ValueError("Could not parse the downloaded file content")

                logger.info(f"Retrieved {len(docs)} raw documents from storage")
                return docs

            except Exception as e:
                logger.error(
                    f"Error retrieving raw documents from storage (attempt {attempt + 1}/{max_retries}): {str(e)}"
                )

                # If this is the last attempt, return empty list
                if attempt == max_retries - 1:
                    logger.error(
                        f"Failed to retrieve raw documents after {max_retries} attempts"
                    )
                    return []

                # Wait before retrying
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff

        return []

    def get_chunks_file_id(self, url: str) -> str:
        """Generate chunks file ID based on the documentation URL"""
        url_lower = url.lower()

        # Map URLs to chunks file IDs
        if "react.dev" in url_lower or "reactjs.org" in url_lower:
            return "react_docs_chunks.json"
        elif "docs.python.org" in url_lower or "python.org" in url_lower:
            return "python_docs_chunks.json"
        elif "golang.org" in url_lower or "go.dev" in url_lower:
            return "golang_docs_chunks.json"
        elif "developer.mozilla.org" in url_lower or "mdn" in url_lower:
            return "mdn_docs_chunks.json"
        elif "vuejs.org" in url_lower:
            return "vue_docs_chunks.json"
        elif "nodejs.org" in url_lower:
            return "nodejs_docs_chunks.json"
        elif "angular.io" in url_lower:
            return "angular_docs_chunks.json"
        elif "svelte.dev" in url_lower:
            return "svelte_docs_chunks.json"
        elif "nextjs.org" in url_lower:
            return "nextjs_docs_chunks.json"
        elif "nuxt.com" in url_lower:
            return "nuxt_docs_chunks.json"
        elif "djangoproject.com" in url_lower or "django" in url_lower:
            return "django_docs_chunks.json"
        elif "fastapi.tiangolo.com" in url_lower or "fastapi" in url_lower:
            return "fastapi_docs_chunks.json"
        elif "docs.docker.com" in url_lower or "docker.com" in url_lower:
            return "docker_docs_chunks.json"
        elif "kubernetes.io" in url_lower:
            return "kubernetes_docs_chunks.json"
        elif "docs.mongodb.com" in url_lower or "mongodb.com" in url_lower:
            return "mongodb_docs_chunks.json"
        elif "postgresql.org" in url_lower or "postgresql" in url_lower:
            return "postgresql_docs_chunks.json"
        else:
            # For unknown URLs, create a generic ID based on domain
            from urllib.parse import urlparse

            parsed = urlparse(url)
            domain = parsed.netloc.replace(".", "_").replace("www_", "")
            return f"{domain}_docs_chunks.json"

    def chunks_already_exist(self, url: str) -> bool:
        """Check if chunks for this URL already exist in storage"""
        try:
            file_id = self.get_chunks_file_id(url)
            # Try to get the file from storage
            self.storage.get_file(bucket_id=self.bucket_id, file_id=file_id)
            logger.info(f"Chunks already exist for {url} (file: {file_id})")
            return True
        except Exception as e:
            logger.info(f"Chunks do not exist for {url}: {str(e)}")
            return False

    def save_chunks_to_storage(
        self, chunks: List[Dict[str, Any]], url: str = None
    ) -> bool:
        """Save document chunks as JSON file to Appwrite storage bucket (FAST)"""
        temp_file_path = None
        max_retries = 3
        retry_delay = 2  # seconds

        for attempt in range(max_retries):
            try:
                logger.info(
                    f"Saving {len(chunks)} chunks to Appwrite storage (attempt {attempt + 1}/{max_retries})"
                )

                # Generate file ID based on URL
                file_id = (
                    self.get_chunks_file_id(url) if url else "unknown_docs_chunks.json"
                )
                logger.info(f"Using chunks file ID: {file_id}")

                # Create JSON content
                json_content = json.dumps(chunks, indent=2, ensure_ascii=False)

                # Create temporary file with a unique name
                temp_file_path = tempfile.mktemp(suffix=".json")

                # Write content to temporary file
                with open(temp_file_path, "w", encoding="utf-8") as temp_file:
                    temp_file.write(json_content)

                # Upload file to storage bucket
                input_file = InputFile.from_path(temp_file_path)

                # Try to delete existing file first, then create new one
                try:
                    # Try to delete existing file
                    self.storage.delete_file(bucket_id=self.bucket_id, file_id=file_id)
                    logger.info(f"Deleted existing chunks file: {file_id}")
                except Exception as e:
                    # File doesn't exist or can't be deleted, that's okay
                    logger.info(
                        f"Could not delete existing chunks file (may not exist): {str(e)}"
                    )

                # Upload to storage with retry logic
                result = self.storage.create_file(
                    bucket_id=self.bucket_id,
                    file_id=file_id,
                    file=input_file,
                )

                logger.info(f"Successfully saved chunks to storage: {result['$id']}")
                return True

            except Exception as e:
                logger.error(
                    f"Error saving chunks to storage (attempt {attempt + 1}/{max_retries}): {str(e)}"
                )

                # Clean up temporary file on error
                if temp_file_path and os.path.exists(temp_file_path):
                    try:
                        os.unlink(temp_file_path)
                        temp_file_path = None
                    except (OSError, PermissionError) as cleanup_error:
                        logger.warning(
                            f"Could not delete temporary file {temp_file_path}: {str(cleanup_error)}"
                        )

                # If this is the last attempt, return False
                if attempt == max_retries - 1:
                    logger.error(f"Failed to save chunks after {max_retries} attempts")
                    return False

                # Wait before retrying
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff

        return False

    def get_chunks_from_storage(self, url: str = None) -> List[Dict[str, Any]]:
        """Retrieve document chunks from Appwrite storage bucket (FAST)"""
        max_retries = 3
        retry_delay = 2  # seconds

        for attempt in range(max_retries):
            try:
                logger.info(
                    f"Retrieving chunks from Appwrite storage (attempt {attempt + 1}/{max_retries})"
                )

                # Generate file ID based on URL
                file_id = (
                    self.get_chunks_file_id(url) if url else "react_docs_chunks.json"
                )
                logger.info(f"Looking for chunks file: {file_id}")

                # Download file from storage
                result = self.storage.get_file_download(
                    bucket_id=self.bucket_id, file_id=file_id
                )

                logger.info(f"Download result type: {type(result)}")

                # Handle different possible return types
                chunks = None

                # Case 1: Result is already a list of dicts (JSON content)
                if isinstance(result, list) and result and isinstance(result[0], dict):
                    chunks = result
                    logger.info("Result is already a list of chunks")

                # Case 2: Result is bytes
                elif isinstance(result, bytes):
                    json_content = result.decode("utf-8")
                    chunks = json.loads(json_content)
                    logger.info("Result is bytes, decoded successfully")

                # Case 3: Result is a list of bytes
                elif (
                    isinstance(result, list) and result and isinstance(result[0], bytes)
                ):
                    json_bytes = b"".join(result)
                    json_content = json_bytes.decode("utf-8")
                    chunks = json.loads(json_content)
                    logger.info("Result is list of bytes, joined and decoded")

                # Case 4: Result is a single dict
                elif isinstance(result, dict):
                    chunks = [result]
                    logger.info("Result is a single chunk dict")

                # Case 5: Try to convert to string and parse
                else:
                    try:
                        json_str = str(result)
                        chunks = json.loads(json_str)
                        logger.info("Result converted to string and parsed")
                    except Exception as e:
                        logger.error(f"Failed to parse result: {str(e)}")
                        raise ValueError(
                            f"Could not parse downloaded chunks file content: {str(e)}"
                        )

                if chunks is None:
                    raise ValueError(
                        "Could not parse the downloaded chunks file content"
                    )

                logger.info(f"Retrieved {len(chunks)} chunks from storage")
                return chunks

            except Exception as e:
                logger.error(
                    f"Error retrieving chunks from storage (attempt {attempt + 1}/{max_retries}): {str(e)}"
                )

                # If this is the last attempt, return empty list
                if attempt == max_retries - 1:
                    logger.error(
                        f"Failed to retrieve chunks after {max_retries} attempts"
                    )
                    return []

                # Wait before retrying
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff

        return []

    def save_chunks(self, chunks: List[Dict[str, Any]], url: str = None) -> bool:
        """Save document chunks - optimized version using storage bucket"""
        try:
            logger.info(f"Saving {len(chunks)} chunks using optimized method")

            # Use the fast storage method instead of database
            return self.save_chunks_to_storage(chunks, url)

        except Exception as e:
            logger.error(f"Error saving chunks: {str(e)}")
            return False

    def get_all_chunks(self, url: str = None) -> List[Dict[str, Any]]:
        """Retrieve all document chunks - optimized version using storage bucket"""
        try:
            logger.info("Retrieving all chunks using optimized method")

            # Use the fast storage method instead of database
            return self.get_chunks_from_storage(url)

        except Exception as e:
            logger.error(f"Error retrieving chunks: {str(e)}")
            return []

    def search_chunks(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for chunks containing specific text"""
        try:
            logger.info(f"Searching for chunks with query: {query}")

            # Search documents in the collection
            response = self.databases.list_documents(
                database_id=self.database_id,
                collection_id=self.chunks_collection_id,
                queries=[],
            )

            chunks = []
            for doc in response["documents"]:
                # Simple client-side search for now
                if (
                    query.lower() in doc["content"].lower()
                    or query.lower() in doc["title"].lower()
                ):
                    chunks.append(
                        {
                            "content": doc["content"],
                            "title": doc["title"],
                            "url": doc.get("url", ""),
                            "chunk_id": doc["chunk_id"],
                        }
                    )

            logger.info(f"Found {len(chunks)} matching chunks")
            return chunks[:limit]

        except Exception as e:
            logger.error(f"Error searching chunks in Appwrite: {str(e)}")
            return []

    def delete_raw_docs_from_storage(self) -> bool:
        """Delete raw documents file from storage bucket"""
        try:
            logger.info("Deleting raw documents from storage")

            # Delete file from storage
            self.storage.delete_file(
                bucket_id=self.bucket_id, file_id="react_docs_raw.json"
            )

            logger.info("Successfully deleted raw documents from storage")
            return True

        except Exception as e:
            logger.error(f"Error deleting raw documents from storage: {str(e)}")
            return False

    def delete_all_chunks(self) -> bool:
        """Delete all chunks from the database (use with caution)"""
        try:
            logger.info("Deleting all chunks from Appwrite")

            # Get all documents
            response = self.databases.list_documents(
                database_id=self.database_id,
                collection_id=self.chunks_collection_id,
            )

            # Delete each document
            for doc in response["documents"]:
                self.databases.delete_document(
                    database_id=self.database_id,
                    collection_id=self.chunks_collection_id,
                    document_id=doc["$id"],
                )

            logger.info("Successfully deleted all chunks")
            return True

        except Exception as e:
            logger.error(f"Error deleting chunks from Appwrite: {str(e)}")
            return False

    def get_raw_docs_count(self) -> int:
        """Get the total number of raw documents in storage"""
        try:
            # Check if raw docs file exists
            try:
                self.storage.get_file(
                    bucket_id=self.bucket_id, file_id="react_docs_raw.json"
                )
                # If file exists, get the count from the content
                docs = self.get_raw_docs_from_storage()
                return len(docs)
            except Exception:
                return 0
        except Exception as e:
            logger.error(f"Error getting raw docs count: {str(e)}")
            return 0

    def get_chunks_count(self) -> int:
        """Get the total number of chunks in the database"""
        try:
            response = self.databases.list_documents(
                database_id=self.database_id,
                collection_id=self.chunks_collection_id,
            )
            return response["total"]
        except Exception as e:
            logger.error(f"Error getting chunks count: {str(e)}")
            return 0

    def clear_all_data(self) -> bool:
        """Clear all data from both storage and database"""
        try:
            logger.info("Clearing all data from storage and database")
            success1 = self.delete_raw_docs_from_storage()
            success2 = self.delete_all_chunks()
            return success1 and success2
        except Exception as e:
            logger.error(f"Error clearing all data: {str(e)}")
            return False

    def list_storage_files(self) -> List[str]:
        """List all files in the storage bucket"""
        try:
            response = self.storage.list_files(bucket_id=self.bucket_id)
            files = [file["$id"] for file in response["files"]]
            logger.info(f"Found {len(files)} files in storage")
            return files
        except Exception as e:
            logger.error(f"Error listing storage files: {str(e)}")
            return []

    def save_completion_status(self, url: str, chunks_count: int) -> bool:
        """Save completion status for a documentation URL"""
        try:
            import datetime

            # Check if completion record already exists
            existing_record = self.get_completion_status(url)

            if existing_record:
                # Update existing record
                self.databases.update_document(
                    database_id=self.database_id,
                    collection_id=self.completion_collection_id,
                    document_id=existing_record["$id"],
                    data={
                        "url": url,
                        "status": "completed",
                        "completed_at": datetime.datetime.now().isoformat(),
                        "chunks_count": chunks_count,
                    },
                )
                logger.info(f"Updated completion status for {url}")
            else:
                # Create new record
                self.databases.create_document(
                    database_id=self.database_id,
                    collection_id=self.completion_collection_id,
                    document_id="unique()",
                    data={
                        "url": url,
                        "status": "completed",
                        "completed_at": datetime.datetime.now().isoformat(),
                        "chunks_count": chunks_count,
                    },
                )
                logger.info(f"Saved completion status for {url}")

            return True
        except Exception as e:
            logger.error(f"Error saving completion status: {str(e)}")
            return False

    def get_completion_status(self, url: str) -> Optional[Dict[str, Any]]:
        """Get completion status for a documentation URL"""
        try:
            from appwrite.query import Query

            response = self.databases.list_documents(
                database_id=self.database_id,
                collection_id=self.completion_collection_id,
                queries=[Query.equal("url", url)],
            )

            if response["documents"]:
                return response["documents"][0]
            return None
        except Exception as e:
            logger.error(f"Error getting completion status: {str(e)}")
            return None

    def is_fully_processed(self, url: str) -> bool:
        """Check if documentation is fully processed (has completion status)"""
        try:
            completion_status = self.get_completion_status(url)
            return (
                completion_status is not None
                and completion_status.get("status") == "completed"
            )
        except Exception as e:
            logger.error(f"Error checking if fully processed: {str(e)}")
            return False


# Global instance
appwrite_service = AppwriteService()

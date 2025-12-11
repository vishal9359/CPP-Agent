"""
RAG System for C++ code using ChromaDB + LangChain components
"""
import os
import shutil
from typing import List, Dict
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class RAGSystem:
    """RAG system for storing and retrieving C++ code"""

    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        collection_name: str = "cpp_code",
    ):
        """
        Initialize RAG system

        Args:
            persist_directory: Directory to persist ChromaDB
            embedding_model: Name of the embedding model
            collection_name: Name for the Chroma collection
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model

        # Initialize embedding model (open-source)
        self.embedder = HuggingFaceEmbeddings(model_name=embedding_model)

        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=200,
            length_function=len,
        )

        # Initialize vector store (loads existing collection if present)
        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedder,
            persist_directory=self.persist_directory,
        )

    def add_documents(self, parsed_files: List[Dict]):
        """
        Add parsed C++ files to the vector database

        Args:
            parsed_files: List of parsed file dictionaries from CPPParser
        """
        documents: List[Document] = []

        for file_data in parsed_files:
            file_path = file_data["file_path"]
            file_name = file_data["file_name"]

            for idx, segment in enumerate(file_data["segments"]):
                base_metadata = {
                    "file_path": file_path,
                    "file_name": file_name,
                    "type": segment["type"],
                    "name": segment["name"],
                    "start_line": segment["start_line"],
                    "end_line": segment["end_line"],
                    "segment_index": idx,
                    "metadata": segment.get("metadata", ""),
                }

                # Split large segments to keep retrieval precise
                for chunk_idx, chunk in enumerate(
                    self.text_splitter.split_text(segment["content"])
                ):
                    metadata = base_metadata.copy()
                    metadata["chunk_index"] = chunk_idx
                    documents.append(
                        Document(
                            page_content=chunk,
                            metadata=metadata,
                        )
                    )

        if documents:
            self.vectorstore.add_documents(documents)
            self.vectorstore.persist()
            print(f"Added {len(documents)} chunks across {len(parsed_files)} files")
        else:
            print("No documents to add to the database")

    def search(self, query: str, n_results: int = 5) -> List[Document]:
        """Search for relevant code segments"""
        return self.vectorstore.similarity_search(query, k=n_results)

    def as_retriever(self, k: int = 5):
        """Return a retriever compatible with LangChain runnables"""
        return self.vectorstore.as_retriever(search_kwargs={"k": k})

    def clear_database(self):
        """Clear all documents from the database"""
        if os.path.exists(self.persist_directory):
            shutil.rmtree(self.persist_directory, ignore_errors=True)
        # Recreate empty store
        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedder,
            persist_directory=self.persist_directory,
        )
        print("Database cleared successfully")

    def get_stats(self) -> Dict:
        """Get statistics about the database"""
        count = self.vectorstore._collection.count() if self.vectorstore else 0
        return {
            "total_documents": count,
            "collection_name": self.collection_name,
            "embedding_model": self.embedding_model_name,
        }


"""
Main RAG Agent for C++ code understanding (open-source stack)
"""
import os
from typing import List, Dict
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_community.chat_models import ChatOllama
from langchain_community.llms import LlamaCpp
from cpp_parser import CPPParser
from rag_system import RAGSystem

load_dotenv()


class CPPRAGAgent:
    """Main agent for understanding and answering questions about C++ projects"""

    def __init__(
        self,
        cpp_project_path: str,
        llm_provider: str = "ollama",
        llm_model: str = "llama3",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        persist_directory: str = "./chroma_db",
        temperature: float = 0.2,
        context_window: int = 4096,
    ):
        """
        Initialize the C++ RAG Agent

        Args:
            cpp_project_path: Path to the C++ project directory
            llm_provider: "ollama" or "llama_cpp"
            llm_model: LLM model name (Ollama name or local gguf path)
            embedding_model: Embedding model name for vector search
            persist_directory: Where to persist ChromaDB
            temperature: LLM temperature
            context_window: Context window for llama_cpp (ignored for Ollama)
        """
        self.cpp_project_path = cpp_project_path
        self.temperature = temperature

        # Initialize components
        self.parser = CPPParser()
        self.rag_system = RAGSystem(
            persist_directory=persist_directory, embedding_model=embedding_model
        )

        # Initialize LLM (open-source only)
        self.llm = self._init_llm(
            provider=llm_provider,
            model=llm_model,
            temperature=temperature,
            context_window=context_window,
        )

        # Build RAG chain
        self.retriever = self.rag_system.as_retriever(k=6)
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert C++ code assistant. Use the provided code "
                    "context to answer the question. Cite file names and line ranges "
                    "when relevant. If the context is insufficient, say so explicitly.",
                ),
                (
                    "human",
                    "Question: {question}\n\nContext:\n{context}\n\nAnswer succinctly.",
                ),
            ]
        )
        self.chain = (
            RunnableParallel(
                question=RunnablePassthrough(),
                context_documents=self.retriever,
            )
            | RunnableLambda(
                lambda x: {
                    "question": x["question"],
                    "context": self._build_context(x["context_documents"]),
                }
            )
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def _init_llm(
        self, provider: str, model: str, temperature: float, context_window: int
    ):
        """Initialize an open-source LLM backend."""
        provider = provider.lower()
        if provider == "ollama":
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            return ChatOllama(
                model=model,
                temperature=temperature,
                base_url=base_url,
            )
        if provider == "llama_cpp":
            if not os.path.exists(model):
                raise ValueError(
                    "llama_cpp selected but model path not found: "
                    f"{model}. Provide a local GGUF path."
                )
            return LlamaCpp(
                model_path=model,
                temperature=temperature,
                n_ctx=context_window,
            )
        raise ValueError("Unsupported llm_provider. Use 'ollama' or 'llama_cpp'.")

    def index_project(self, force_reindex: bool = False):
        """
        Index the C++ project

        Args:
            force_reindex: If True, clear existing index and reindex
        """
        if force_reindex:
            self.rag_system.clear_database()

        print(f"Parsing C++ project at: {self.cpp_project_path}")
        parsed_files = self.parser.parse_directory(self.cpp_project_path)

        if not parsed_files:
            print("No C++ files found in the project directory")
            return

        print(f"Found {len(parsed_files)} C++ files")
        print("Indexing code segments...")

        self.rag_system.add_documents(parsed_files)

        stats = self.rag_system.get_stats()
        print(f"Indexing complete! Total chunks: {stats['total_documents']}")

    def answer_question(self, question: str, max_context: int = 5) -> Dict:
        """
        Answer a question about the C++ project

        Args:
            question: User's question
            max_context: Maximum number of code segments to use as context

        Returns:
            Dictionary with answer and relevant code references
        """
        # Adjust retriever depth dynamically
        self.retriever.search_kwargs["k"] = max_context

        # Search for relevant code (for sources)
        search_results = self.rag_system.search(question, n_results=max_context)

        if not search_results:
            return {
                "answer": "I couldn't find relevant code in the project for this question.",
                "sources": [],
            }

        # Generate answer with LangChain runnable chain
        answer = self.chain.invoke(question)

        # Format sources
        sources = [
            {
                "file": doc.metadata.get("file_name"),
                "type": doc.metadata.get("type"),
                "name": doc.metadata.get("name"),
                "lines": f"{doc.metadata.get('start_line')}-{doc.metadata.get('end_line')}",
            }
            for doc in search_results
        ]

        return {
            "answer": answer,
            "sources": sources,
            "context_segments": len(search_results),
        }

    def _build_context(self, search_results) -> str:
        """Build context string from search results"""
        context_parts = []

        for i, doc in enumerate(search_results, 1):
            metadata = doc.metadata
            context_parts.append(
                f"[Context {i}] {metadata.get('file_name')} "
                f"{metadata.get('type')} '{metadata.get('name')}' "
                f"(lines {metadata.get('start_line')}-{metadata.get('end_line')})\n"
                f"{doc.page_content}\n"
            )

        return "\n".join(context_parts)

    def get_project_stats(self) -> Dict:
        """Get statistics about the indexed project"""
        return self.rag_system.get_stats()


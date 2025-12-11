# C++ RAG Agent (Open-Source Stack)

A Retrieval-Augmented Generation (RAG) agent that can understand C++ projects, answer questions, and generate flowcharts using only open-source models.

## Features

- 🔍 **Code Parsing**: Automatically parses C++ files (tree-sitter) to extract functions, classes, structs, and namespaces
- 📚 **Vector Database**: Uses ChromaDB with SentenceTransformers embeddings
- 🤖 **Open-Source LLM**: LangChain pipeline with Ollama or llama.cpp (no closed models)
- 🗺️ **Flowcharts**: LLM-guided Graphviz diagrams for any C++ project (parser fallback)
- 💬 **Interactive CLI**: Friendly REPL with `stats` and `diagram` commands

## Quickstart

```bash
pip install -r requirements.txt

# Index and chat (Ollama with llama3 by default)
python cli.py D:/git-project/poseidonos --llm-provider ollama --llm-model llama3

# Generate a flowchart only (LLM-driven)
python cli.py D:/git-project/poseidonos --diagram --diagram-llm --diagram-name poseidonos_flow
```

### Prerequisites
- Python 3.9+
- Graphviz binaries installed (for diagram rendering)
- **Ollama** running locally (`ollama serve`) **or** a local GGUF file for `llama_cpp`
- Sufficient disk space for Chroma persistence (`./chroma_db` by default)

## Usage (CLI)

- `python cli.py <path>` – index (if needed) and start Q&A
- `--llm-provider {ollama,llama_cpp}` – choose backend
- `--llm-model` – Ollama model name (e.g., `llama3`) or GGUF path for `llama_cpp`
- `--embedding-model` – SentenceTransformers model (default: `sentence-transformers/all-MiniLM-L6-v2`)
- `--reindex` – force reindexing even if a DB exists
- `--diagram` – generate flowchart and exit (use `--diagram-path` to target a subdir/file)

Interactive commands:
- `stats` – show indexed chunk count and embedding model
- `diagram [optional/path]` – generate a flowchart for the project or a specific module
- `exit` / `quit` – leave the REPL

## How It Works

1. **Parse**: tree-sitter extracts code structures from C++ sources
2. **Chunk & Embed**: segments are chunked and embedded with SentenceTransformers
3. **Store**: embeddings persisted in ChromaDB
4. **Retrieve & Generate**: LangChain runnable pipeline retrieves top-k chunks and queries an open-source LLM for grounded answers
5. **Diagram**: Graphviz visualizes files, includes, and definitions for a project or module path

## Project Structure

```
cpp_rag_agent/
├── agent.py            # Main agent (LangChain RAG + open-source LLM)
├── cli.py              # Command-line interface
├── cpp_parser.py       # C++ parser via tree-sitter
├── rag_system.py       # Chroma + embeddings management
├── diagram_generator.py# Graphviz flowchart builder
├── example_usage.py    # Usage sample (incl. poseidonos path)
├── requirements.txt    # Dependencies
└── chroma_db/          # Vector DB (created automatically)
```

## Target Project Example

To work with the provided Poseidonos repo:

```bash
python cli.py D:/git-project/poseidonos --llm-provider ollama --llm-model llama3
```

After indexing, ask questions or run `diagram` to visualize modules.

## Troubleshooting

- **No C++ files found**: verify the path and that extensions include `.cpp`, `.h`, `.hpp`, `.cc`, `.cxx`, `.hxx`
- **Graphviz errors**: ensure Graphviz system binaries are installed and on PATH
- **LLM errors**:
  - Ollama: ensure `ollama serve` is running and the model is pulled (`ollama pull llama3`)
  - llama_cpp: provide a valid GGUF path via `--llm-model /path/to/model.gguf`
- **Slow indexing**: large repos can take time; re-run with `--reindex` after code changes

## License

MIT License - feel free to use and modify as needed!


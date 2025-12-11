"""
Setup script for C++ RAG Agent
Helps with installation, especially tree-sitter-cpp
"""
import subprocess
import sys
import os

def install_requirements():
    """Install all requirements"""
    print("Installing Python packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    print("✓ Packages installed successfully")

def verify_installation():
    """Verify that all components are installed correctly"""
    print("\nVerifying installation...")
    
    try:
        import chromadb
        print("✓ ChromaDB installed")
    except ImportError:
        print("✗ ChromaDB not installed")
        return False
    
    try:
        import sentence_transformers
        print("✓ Sentence Transformers installed")
    except ImportError:
        print("✗ Sentence Transformers not installed")
        return False
    
    try:
        import tree_sitter
        import tree_sitter_cpp
        print("✓ Tree-sitter C++ installed")
    except ImportError:
        print("✗ Tree-sitter C++ not installed")
        print("  Note: You may need to rebuild tree-sitter if there are issues")
        return False

    try:
        import langchain
        import langchain_community  # noqa: F401
        print("✓ LangChain installed")
    except ImportError:
        print("✗ LangChain not installed")
        return False

    try:
        import graphviz  # noqa: F401
        print("✓ Graphviz python bindings installed")
    except ImportError:
        print("✗ Graphviz python bindings not installed")
        return False

    try:
        import ollama  # noqa: F401
        print("✓ Ollama client installed (ensure daemon is running)")
    except ImportError:
        print("⚠ Ollama client not installed — required if using --llm-provider ollama")

    try:
        import llama_cpp  # noqa: F401
        print("✓ llama-cpp-python installed (for local GGUF)")
    except ImportError:
        print("⚠ llama-cpp-python not installed — required if using --llm-provider llama_cpp")

    try:
        import rich
        print("✓ Rich installed")
    except ImportError:
        print("✗ Rich not installed")
        return False
    
    print("\n✓ All components verified!")
    return True

if __name__ == "__main__":
    print("C++ RAG Agent Setup")
    print("=" * 50)
    
    install_requirements()
    
    if verify_installation():
        print("\nSetup complete! You can now run:")
        print("  python cli.py /path/to/your/cpp/project")
    else:
        print("\nSetup completed with some warnings.")
        print("Please check the errors above and install missing packages manually.")


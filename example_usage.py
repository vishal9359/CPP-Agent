"""
Example usage of the C++ RAG Agent
"""
from agent import CPPRAGAgent
from diagram_generator import FlowchartGenerator

# Initialize the agent with your C++ project path
agent = CPPRAGAgent(
    cpp_project_path="D:/git-project/poseidonos",  # Change this to your project path
    llm_provider="ollama",  # or "llama_cpp"
    llm_model="llama3",  # Ollama model name or local GGUF path
)

# Index the project (only needed once, or when code changes)
print("Indexing project...")
agent.index_project(force_reindex=True)

# Ask questions
questions = [
    "What classes are defined in this project?",
    "How does the main function work?",
    "What is the purpose of the UserManager class?",
]

for question in questions:
    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print("=" * 60)

    result = agent.answer_question(question)

    print(f"\nAnswer:\n{result['answer']}")
    print("\nSources:")
    for source in result["sources"]:
        print(f"  - {source['file']}: {source['name']} (lines {source['lines']})")

# Generate a flowchart for the whole project (LLM-driven, fallback to parser)
flowchart = FlowchartGenerator(parser=agent.parser)
diagram_path = flowchart.generate_flowchart(
    target_path=agent.cpp_project_path,
    output_dir="diagrams",
    file_name="poseidonos_flow",
    image_format="png",
    use_llm=True,
    llm=agent.llm,
)
print(f"\nFlowchart saved to: {diagram_path}")


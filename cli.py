"""
CLI interface for the C++ RAG Agent
"""
import argparse
import os
import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.markdown import Markdown
from agent import CPPRAGAgent
from diagram_generator import FlowchartGenerator

console = Console()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="C++ RAG Agent (open-source stack)")
    parser.add_argument("project_path", nargs="?", default=".", help="Path to C++ project")
    parser.add_argument(
        "--llm-provider",
        choices=["ollama", "llama_cpp"],
        default="ollama",
        help="LLM backend to use (open-source only)",
    )
    parser.add_argument(
        "--llm-model",
        default="llama3",
        help="Model name for Ollama or GGUF path for llama_cpp",
    )
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformers embedding model",
    )
    parser.add_argument(
        "--persist-dir",
        default="./chroma_db",
        help="Directory to persist ChromaDB",
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Force reindex even if a database already exists",
    )
    parser.add_argument(
        "--diagram",
        action="store_true",
        help="Generate a flowchart for the project and exit",
    )
    parser.add_argument(
        "--diagram-llm",
        action="store_true",
        help="Use the LLM to generate the diagram (recommended for generic projects)",
    )
    parser.add_argument(
        "--diagram-path",
        default=None,
        help="Specific file or directory to diagram (defaults to project_path)",
    )
    parser.add_argument(
        "--diagram-format",
        default="png",
        help="graphviz output format (png/svg/pdf)",
    )
    parser.add_argument(
        "--diagram-name",
        default="cpp_flow",
        help="Base filename for the generated diagram",
    )
    return parser


def main():
    """Main CLI entry point"""
    args = build_arg_parser().parse_args()
    project_path = args.project_path

    console.print(
        Panel.fit(
            "[bold cyan]C++ RAG Agent[/bold cyan]\n"
            "Open-source RAG + flowcharts for C++ projects",
            border_style="cyan",
        )
    )

    if not os.path.exists(project_path):
        console.print(f"[red]Error: Path '{project_path}' does not exist[/red]")
        return

    # Initialize agent (needed for LLM and retrieval even in diagram-only mode)
    console.print(f"\n[cyan]Initializing agent for: {project_path}[/cyan]")

    try:
        agent = CPPRAGAgent(
            cpp_project_path=project_path,
            llm_provider=args.llm_provider,
            llm_model=args.llm_model,
            embedding_model=args.embedding_model,
            persist_directory=args.persist_dir,
        )
    except Exception as e:
        console.print(f"[red]Error initializing agent: {e}[/red]")
        return

    # Diagram-only mode (uses agent.llm when requested)
    if args.diagram:
        target = args.diagram_path or project_path
        console.print(f"[cyan]Generating flowchart for: {target}[/cyan]")
        try:
            generator = FlowchartGenerator(parser=agent.parser)
            output = generator.generate_flowchart(
                target_path=target,
                output_dir="diagrams",
                file_name=args.diagram_name,
                image_format=args.diagram_format,
                use_llm=args.diagram_llm,
                llm=agent.llm if args.diagram_llm else None,
            )
            console.print(f"[green]Diagram saved to: {output}[/green]")
        except Exception as e:
            console.print(f"[red]Error generating diagram: {e}[/red]")
        return

    # Check if already indexed
    stats = agent.get_project_stats()
    if stats["total_documents"] > 0 and not args.reindex:
        console.print(
            f"[green]Found existing index with {stats['total_documents']} chunks[/green]"
        )
        reindex = Confirm.ask("[cyan]Reindex the project?[/cyan]", default=False)
    else:
        reindex = True

    # Index project
    if reindex:
        console.print("\n[yellow]Indexing project... This may take a while.[/yellow]")
        try:
            agent.index_project(force_reindex=reindex)
        except Exception as e:
            console.print(f"[red]Error indexing project: {e}[/red]")
            return
    else:
        console.print("[green]Using existing index[/green]")

    flow_generator = FlowchartGenerator(parser=agent.parser)

    # Interactive Q&A loop
    console.print("\n[bold green]Ready! Ask questions about your C++ project.[/bold green]")
    console.print(
        "[dim]Type 'exit' or 'quit' to exit, 'stats' for statistics, "
        "'diagram <path>' to build a flowchart[/dim]\n"
    )

    while True:
        try:
            question = Prompt.ask("[bold cyan]Question[/bold cyan]")

            if question.lower() in ["exit", "quit", "q"]:
                console.print("[yellow]Goodbye![/yellow]")
                break

            if question.lower() == "stats":
                stats = agent.get_project_stats()
                console.print(
                    f"[cyan]Total indexed chunks: {stats['total_documents']} "
                    f"(model: {stats['embedding_model']})[/cyan]"
                )
                continue

            if question.lower().startswith("diagram"):
                _, *rest = question.split(maxsplit=1)
                target = rest[0] if rest else project_path
                console.print(f"[cyan]Generating flowchart for: {target}[/cyan]")
                try:
                    output = flow_generator.generate_flowchart(
                        target_path=target,
                        output_dir="diagrams",
                        file_name=args.diagram_name,
                        image_format=args.diagram_format,
                        use_llm=True,
                        llm=agent.llm,
                    )
                    console.print(f"[green]Diagram saved to: {output}[/green]")
                except Exception as e:
                    console.print(f"[red]Error generating diagram: {e}[/red]")
                continue

            if not question.strip():
                continue

            # Get answer
            console.print("\n[dim]Searching and generating answer...[/dim]")
            result = agent.answer_question(question)

            # Display answer
            console.print("\n[bold green]Answer:[/bold green]")
            console.print(Markdown(result["answer"]))

            # Display sources
            if result["sources"]:
                console.print(f"\n[bold cyan]Sources ({len(result['sources'])}):[/bold cyan]")
                for source in result["sources"]:
                    console.print(
                        f"  • {source['file']} - {source['type']} '{source['name']}' "
                        f"(lines {source['lines']})"
                    )

            console.print()  # Empty line for spacing

        except KeyboardInterrupt:
            console.print("\n[yellow]Goodbye![/yellow]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]\n")


if __name__ == "__main__":
    main()


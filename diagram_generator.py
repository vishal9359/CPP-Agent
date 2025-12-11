"""
Flowchart/structure diagram generation for C++ projects.
Primary path uses an LLM to produce a DOT graph from a summarized
view of the project, with a parser-assisted fallback.
"""
from pathlib import Path
import re
from typing import List, Dict, Optional
from graphviz import Digraph, Source
from cpp_parser import CPPParser


class FlowchartGenerator:
    """Generate structural flow charts for a whole project or a module path."""

    def __init__(self, parser: Optional[CPPParser] = None):
        self.parser = parser or CPPParser()

    def generate_flowchart(
        self,
        target_path: str,
        output_dir: str = "diagrams",
        file_name: str = "cpp_flow",
        image_format: str = "png",
        use_llm: bool = True,
        llm=None,
    ) -> str:
        """
        Create a flow chart for the given project or module path.

        Args:
            target_path: File or directory to analyze
            output_dir: Directory to write the diagram into
            file_name: Base filename (without extension)
            image_format: graphviz output format (png/svg/pdf)
            use_llm: If True, ask the LLM to produce DOT; otherwise use parser-only
            llm: LangChain-compatible chat/LLM to generate DOT (required when use_llm)

        Returns:
            Path to the generated diagram file
        """
        target = Path(target_path)
        if not target.exists():
            raise FileNotFoundError(f"Target path does not exist: {target_path}")

        # Parse the requested scope to gather structured context
        parsed_files: List[Dict] = []
        if target.is_file():
            parsed_files.append(self.parser.parse_file(str(target)))
        else:
            parsed_files.extend(self.parser.parse_directory(str(target)))

        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        output_path = output_dir_path / file_name

        # LLM-first generation for generic coverage
        if use_llm and llm is not None:
            dot_text = self._generate_dot_with_llm(parsed_files, llm)
            if dot_text:
                src = Source(dot_text, filename=str(output_path), format=image_format)
                return src.render(cleanup=True)
            # fall back if LLM fails

        # Fallback: parser-guided deterministic graph
        graph = self._build_parser_graph(parsed_files, image_format)
        rendered = graph.render(str(output_path), cleanup=True)
        return rendered

    def _build_parser_graph(self, parsed_files: List[Dict], image_format: str) -> Digraph:
        """Deterministic parser-driven diagram (fallback)."""
        graph = Digraph("cpp_flow", format=image_format)
        graph.attr(rankdir="LR", concentrate="true", fontsize="10")

        include_edges = []

        for file_data in parsed_files:
            file_node_id = self._sanitize_id(file_data["file_path"])
            graph.node(
                file_node_id,
                label=f"{file_data['file_name']}\n{file_data['file_path']}",
                shape="box",
                style="filled",
                fillcolor="#e0f7fa",
            )

            include_edges.extend(
                self._find_includes(file_data.get("content", ""), file_node_id)
            )

            for segment in file_data["segments"]:
                seg_node_id = self._sanitize_id(
                    f"{file_data['file_path']}::{segment['name']}::{segment['type']}"
                )
                graph.node(
                    seg_node_id,
                    label=f"{segment['type']}: {segment['name']}\n"
                    f"Lines {segment['start_line']}-{segment['end_line']}",
                    shape="ellipse" if segment["type"] == "function" else "note",
                    style="filled",
                    fillcolor="#f1f8e9",
                )
                graph.edge(file_node_id, seg_node_id, label="defines", fontsize="9")

        for src, dest, label in include_edges:
            graph.edge(src, dest, label=label, style="dashed", fontsize="8")
        return graph

    def _generate_dot_with_llm(self, parsed_files: List[Dict], llm) -> Optional[str]:
        """
        Ask the LLM to produce a Graphviz DOT diagram from summarized code context.
        Returns DOT string or None if generation fails.
        """
        context_summary = self._summarize_for_llm(parsed_files)
        prompt = (
            "You are a C++ project visualizer. Produce a Graphviz DOT graph that "
            "shows files and the symbols defined in them, plus include relationships. "
            "Use simple nodes: file nodes as boxes, symbol nodes as ellipses (functions) "
            "or notes (classes/structs). Connect file -> symbol with label 'defines'. "
            "For includes, draw file -> included file with dashed style. "
            "Ensure the DOT is self-contained and valid. Do not add explanations.\n\n"
            f"Context:\n{context_summary}\n\n"
            "Return only the DOT code starting with 'digraph' and nothing else."
        )
        try:
            response = llm.invoke(prompt)
            text = response if isinstance(response, str) else getattr(response, "content", "")
            text = text.strip()
            if "digraph" not in text:
                return None
            return text
        except Exception as e:
            print(f"LLM diagram generation failed, falling back to parser logic: {e}")
            return None

    def _summarize_for_llm(self, parsed_files: List[Dict]) -> str:
        """Create a concise textual summary for the LLM to ground the DOT output."""
        lines = []
        for file_data in parsed_files[:200]:  # keep it bounded
            lines.append(f"File: {file_data['file_name']} ({file_data['file_path']})")
            includes = ", ".join(self._extract_includes(file_data.get("content", ""))[:10])
            if includes:
                lines.append(f"  Includes: {includes}")
            for segment in file_data["segments"][:50]:
                lines.append(
                    f"  {segment['type']}: {segment['name']} "
                    f"lines {segment['start_line']}-{segment['end_line']}"
                )
        return "\n".join(lines[:800])  # safety bound

    def _find_includes(self, content: str, file_node_id: str):
        """Find #include relationships and return edge tuples."""
        edges = []
        for include in self._extract_includes(content):
            target_id = self._sanitize_id(include)
            edges.append((file_node_id, target_id, "#include"))
        return edges

    def _extract_includes(self, content: str):
        pattern = re.compile(r'^\s*#\s*include\s+"([^"]+)"', re.MULTILINE)
        return pattern.findall(content)

    def _sanitize_id(self, value: str) -> str:
        """Sanitize strings so they are safe graphviz node IDs."""
        return re.sub(r"[^a-zA-Z0-9_:/.-]", "_", value)

"""
C++ Code Parser for extracting code segments and metadata
"""
import os
from pathlib import Path
from typing import List, Dict, Optional
import tree_sitter_cpp as tscpp
from tree_sitter import Language, Parser


class CPPParser:
    """Parser for C++ code files using tree-sitter"""
    
    def __init__(self):
        """Initialize the C++ parser"""
        try:
            self.language = Language(tscpp.language())
            self.parser = Parser()
            self.parser.set_language(self.language)
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize tree-sitter C++ parser: {e}\n"
                "Make sure tree-sitter-cpp is properly installed. "
                "You may need to rebuild it: pip install --force-reinstall tree-sitter-cpp"
            ) from e
    
    def parse_file(self, file_path: str) -> Dict:
        """
        Parse a C++ file and extract code segments
        
        Args:
            file_path: Path to the C++ file
            
        Returns:
            Dictionary with file metadata and code segments
        """
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        tree = self.parser.parse(bytes(content, 'utf8'))
        root_node = tree.root_node
        
        segments = []
        segments.extend(self._extract_functions(root_node, content))
        segments.extend(self._extract_classes(root_node, content))
        segments.extend(self._extract_structs(root_node, content))
        segments.extend(self._extract_namespaces(root_node, content))
        
        # If no specific segments found, chunk by lines
        if not segments:
            segments = self._chunk_by_lines(content, file_path)
        
        return {
            'file_path': file_path,
            'file_name': os.path.basename(file_path),
            'content': content,
            'segments': segments,
            'total_lines': len(content.split('\n'))
        }
    
    def _extract_functions(self, node, content: str) -> List[Dict]:
        """Extract function definitions"""
        functions = []
        
        def traverse(n):
            if n.type == 'function_definition':
                func_text = content[n.start_byte:n.end_byte]
                func_name = self._get_function_name(n, content)
                functions.append({
                    'type': 'function',
                    'name': func_name,
                    'content': func_text,
                    'start_line': n.start_point[0] + 1,
                    'end_line': n.end_point[0] + 1,
                    'metadata': f"Function: {func_name} in {n.start_point[0] + 1}-{n.end_point[0] + 1}"
                })
            
            for child in n.children:
                traverse(child)
        
        traverse(node)
        return functions
    
    def _extract_classes(self, node, content: str) -> List[Dict]:
        """Extract class definitions"""
        classes = []
        
        def traverse(n):
            if n.type == 'class_specifier':
                class_text = content[n.start_byte:n.end_byte]
                class_name = self._get_class_name(n, content)
                classes.append({
                    'type': 'class',
                    'name': class_name,
                    'content': class_text,
                    'start_line': n.start_point[0] + 1,
                    'end_line': n.end_point[0] + 1,
                    'metadata': f"Class: {class_name} in {n.start_point[0] + 1}-{n.end_point[0] + 1}"
                })
            
            for child in n.children:
                traverse(child)
        
        traverse(node)
        return classes
    
    def _extract_structs(self, node, content: str) -> List[Dict]:
        """Extract struct definitions"""
        structs = []
        
        def traverse(n):
            if n.type == 'struct_specifier':
                struct_text = content[n.start_byte:n.end_byte]
                struct_name = self._get_struct_name(n, content)
                structs.append({
                    'type': 'struct',
                    'name': struct_name,
                    'content': struct_text,
                    'start_line': n.start_point[0] + 1,
                    'end_line': n.end_point[0] + 1,
                    'metadata': f"Struct: {struct_name} in {n.start_point[0] + 1}-{n.end_point[0] + 1}"
                })
            
            for child in n.children:
                traverse(child)
        
        traverse(node)
        return structs
    
    def _extract_namespaces(self, node, content: str) -> List[Dict]:
        """Extract namespace definitions"""
        namespaces = []
        
        def traverse(n):
            if n.type == 'namespace_definition':
                ns_text = content[n.start_byte:n.end_byte]
                ns_name = self._get_namespace_name(n, content)
                namespaces.append({
                    'type': 'namespace',
                    'name': ns_name,
                    'content': ns_text,
                    'start_line': n.start_point[0] + 1,
                    'end_line': n.end_point[0] + 1,
                    'metadata': f"Namespace: {ns_name} in {n.start_point[0] + 1}-{n.end_point[0] + 1}"
                })
            
            for child in n.children:
                traverse(child)
        
        traverse(node)
        return namespaces
    
    def _get_function_name(self, node, content: str) -> str:
        """Extract function name from node"""
        for child in node.children:
            if child.type == 'function_declarator':
                for subchild in child.children:
                    if subchild.type == 'identifier':
                        return content[subchild.start_byte:subchild.end_byte]
        return "anonymous"
    
    def _get_class_name(self, node, content: str) -> str:
        """Extract class name from node"""
        for child in node.children:
            if child.type == 'type_identifier':
                return content[child.start_byte:child.end_byte]
        return "anonymous"
    
    def _get_struct_name(self, node, content: str) -> str:
        """Extract struct name from node"""
        for child in node.children:
            if child.type == 'type_identifier':
                return content[child.start_byte:child.end_byte]
        return "anonymous"
    
    def _get_namespace_name(self, node, content: str) -> str:
        """Extract namespace name from node"""
        for child in node.children:
            if child.type == 'identifier':
                return content[child.start_byte:child.end_byte]
        return "global"
    
    def _chunk_by_lines(self, content: str, file_path: str, chunk_size: int = 50) -> List[Dict]:
        """Fallback: chunk content by lines"""
        lines = content.split('\n')
        chunks = []
        
        for i in range(0, len(lines), chunk_size):
            chunk_lines = lines[i:i + chunk_size]
            chunk_content = '\n'.join(chunk_lines)
            chunks.append({
                'type': 'code_block',
                'name': f"Lines {i+1}-{min(i+chunk_size, len(lines))}",
                'content': chunk_content,
                'start_line': i + 1,
                'end_line': min(i + chunk_size, len(lines)),
                'metadata': f"Code block: {i+1}-{min(i+chunk_size, len(lines))}"
            })
        
        return chunks
    
    def parse_directory(self, directory: str, extensions: List[str] = None) -> List[Dict]:
        """
        Parse all C++ files in a directory
        
        Args:
            directory: Path to directory
            extensions: List of file extensions to parse (default: .cpp, .h, .hpp, .cc, .cxx)
            
        Returns:
            List of parsed file dictionaries
        """
        if extensions is None:
            extensions = ['.cpp', '.h', '.hpp', '.cc', '.cxx', '.hxx']
        
        parsed_files = []
        directory_path = Path(directory)
        
        for ext in extensions:
            for file_path in directory_path.rglob(f'*{ext}'):
                try:
                    parsed = self.parse_file(str(file_path))
                    parsed_files.append(parsed)
                except Exception as e:
                    print(f"Error parsing {file_path}: {e}")
        
        return parsed_files


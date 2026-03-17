import os
import asyncio
import logging
from fastmcp import FastMCP

logger = logging.getLogger(__name__)
mcp = FastMCP("Research Assistant 🧪")

SAFE_REPO_PATH = "/Users/surfiniaburger/Desktop/mental-research-swarm/research_env"

@mcp.tool()
async def execute_command(command: str, cwd: str = SAFE_REPO_PATH, timeout: int = 300) -> str:
    logger.info(f"🐚 Executing: {command} in {cwd}")
    if not cwd.startswith(SAFE_REPO_PATH):
        return f"Error: CWD {cwd} is outside of safe research path"
    
    try:
        process = await asyncio.create_subprocess_shell(
            command,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
            return (stdout.decode() + stderr.decode()).strip()
        except asyncio.TimeoutError:
            process.kill()
            return "Error: Command timed out."
    except Exception as e:
        return f"Error executing command: {str(e)}"

@mcp.tool()
async def read_research_file(path: str) -> str:
    full_path = os.path.join(SAFE_REPO_PATH, path) if not os.path.isabs(path) else path
    if not full_path.startswith(SAFE_REPO_PATH):
        return "Error: Path outside of safe research scope."
    try:
        with open(full_path, "r") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"

@mcp.tool()
async def write_research_file(path: str, content: str, append: bool = False) -> str:
    full_path = os.path.join(SAFE_REPO_PATH, path) if not os.path.isabs(path) else path
    if not full_path.startswith(SAFE_REPO_PATH):
        return "Error: Path outside of safe research scope."
    mode = "a" if append else "w"
    try:
        with open(full_path, mode) as f:
            f.write(content)
        return f"Successfully {'appended to' if append else 'wrote to'} {path}"
    except Exception as e:
        return f"Error writing file: {str(e)}"

@mcp.tool()
async def patch_research_file(path: str, target_node: str, new_content: str) -> str:
    """
    Surgery-style update: Replaces a specific class or function in a file.
    target_node: Name of the class or function to replace.
    new_content: The full code of the new class/function.
    """
    import ast
    full_path = os.path.join(SAFE_REPO_PATH, path) if not os.path.isabs(path) else path
    if not full_path.startswith(SAFE_REPO_PATH):
        return "Error: Path outside of safe research scope."
    
    try:
        with open(full_path, "r") as f:
            source = f.read()
            
        tree = ast.parse(source)
        lines = source.splitlines()
        
        start_line = -1
        end_line = -1
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == target_node:
                start_line = node.lineno - 1
                end_line = node.end_lineno
                break
        
        if start_line == -1:
            return f"Error: Could not find target node '{target_node}' in {path}"
        
        # Replace the lines
        new_lines = lines[:start_line] + new_content.splitlines() + lines[end_line:]
        
        with open(full_path, "w") as f:
            f.write("\n".join(new_lines) + "\n")
            
        return f"Successfully patched {target_node} in {path}"
    except Exception as e:
        return f"Error patching file: {str(e)}"

if __name__ == "__main__":
    mcp.run()

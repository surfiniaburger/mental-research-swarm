import os
import json
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ResearchResult:
    val_bpb: float
    peak_vram_gb: float
    status: str
    description: str

class ResearchProtocolDriver:
    """
    Layer 3: Protocol Driver.
    Translates research concepts into MCP tool calls.
    """
    def __init__(self, mcp_client):
        self.mcp = mcp_client
        self.repo_path = "/Users/surfiniaburger/Desktop/mental-research-swarm/research_env"

    async def log_result(self, result: ResearchResult) -> None:
        line = f"baseline\t{result.val_bpb}\t{result.peak_vram_gb}\t{result.status}\t{result.description}\n"
        await self.mcp.call_tool("write_research_file", {
            "path": "results.tsv",
            "content": line,
            "append": True
        })

    async def ensure_setup(self) -> bool:
        from datetime import datetime
        # Use a more unique tag: MonthDay-HourMinute
        tag = datetime.now().strftime("%b%d-%H%M").lower()
        branch_name = f"autoresearch/{tag}"
        
        try:
            await self.mcp.call_tool("execute_command", {
                "command": f"git checkout -b {branch_name}",
                "cwd": self.repo_path
            })
        except:
            await self.mcp.call_tool("execute_command", {
                "command": f"git checkout {branch_name}",
                "cwd": self.repo_path
            })

        results_path = "results.tsv"
        res = await self.mcp.call_tool("read_research_file", {"path": results_path})
        if "Error" in res:
            header = "commit\tval_bpb\tmemory_gb\tstatus\tdescription\n"
            await self.mcp.call_tool("write_research_file", {
                "path": results_path,
                "content": header
            })
        return True

    async def get_history(self) -> List[ResearchResult]:
        """Reads results.tsv and returns a list of ResearchResult objects."""
        res = await self.mcp.call_tool("read_research_file", {"path": "results.tsv"})
        history = []
        if "Error" in res:
            return history
            
        lines = res.strip().split("\n")
        if len(lines) <= 1: # Header only
            return history
            
        for line in lines[1:]: # Skip header
            try:
                parts = line.split("\t")
                if len(parts) >= 4:
                    history.append(ResearchResult(
                        val_bpb=float(parts[1]),
                        peak_vram_gb=float(parts[2]),
                        status=parts[3],
                        description=parts[4] if len(parts) > 4 else ""
                    ))
            except Exception as e:
                logger.warning(f"Error parsing history line: {e}")
        return history

    async def run_experiment(self, description: str) -> ResearchResult:
        await self.mcp.call_tool("execute_command", {
            "command": f'git add train.py && git commit -m "autocommit: {description}"',
            "cwd": self.repo_path
        })

        await self.mcp.call_tool("execute_command", {
            "command": "uv run train.py > run.log 2>&1",
            "cwd": self.repo_path,
            "timeout": 1200
        })

        return await self._parse_metrics("run.log")

    async def _parse_metrics(self, log_filename: str) -> ResearchResult:
        log_content = await self.mcp.call_tool("read_research_file", {"path": log_filename})
        val_bpb = 0.0
        vram_mb = 0.0
        
        for line in log_content.splitlines():
            if line.startswith("val_bpb:"):
                val_bpb = float(line.split(":")[1].strip())
            if line.startswith("peak_vram_mb:"):
                vram_mb = float(line.split(":")[1].strip())
                
        return ResearchResult(
            val_bpb=val_bpb,
            peak_vram_gb=round(vram_mb / 1024.0, 1),
            status="keep" if val_bpb > 0 else "crash",
            description="Experiment run"
        )

    async def read_crash_log(self, max_lines: int = 30) -> str:
        """Read the last N lines of run.log for crash diagnosis feedback.
        
        Returns the tail of the log so the Brain/Hands can learn from failures
        instead of blindly repeating the same broken patterns.
        """
        try:
            content = await self.mcp.call_tool("read_research_file", {"path": "run.log"})
            if not content or not content.strip():
                return ""
            lines = content.strip().split("\n")
            tail = lines[-max_lines:]
            return "\n".join(tail)
        except Exception:
            return ""

class SkillWriterProtocolDriver:
    """
    Layer 3: Protocol Driver for Skill Updates.
    """
    def __init__(self, mcp_client):
        self.mcp = mcp_client
        self.repo_path = "/Users/surfiniaburger/Desktop/mental-research-swarm/research_env"

    async def get_latest_results(self) -> str:
        return await self.mcp.call_tool("read_research_file", {"path": "results.tsv"})

    async def get_latest_log(self) -> str:
        try:
            return await self.mcp.call_tool("read_research_file", {"path": "run.log"})
        except:
            return ""

    async def update_skill(self, new_instructions: str) -> bool:
        current_skill = await self.mcp.call_tool("read_research_file", {"path": "program.md"})
        section_header = "## Research Insights"
        
        if section_header in current_skill:
            base_skill = current_skill.split(section_header)[0].strip()
        else:
            base_skill = current_skill.strip()

        updated_skill = base_skill + "\n\n" + section_header + "\n\n" + new_instructions
        await self.mcp.call_tool("write_research_file", {
            "path": "program.md",
            "content": updated_skill.strip() + "\n"
        })
        return True

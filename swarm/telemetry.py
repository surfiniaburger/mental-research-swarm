import os
import subprocess
import time
import logging
import json
import re
from datetime import datetime
from typing import Optional, Dict, Any
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse

logger = logging.getLogger(__name__)

TELEMETRY_LOG = "telemetry.log"

def get_mac_hardware_stats() -> Dict[str, float]:
    """Retrieves basic hardware stats on macOS without external deps."""
    stats = {"cpu_util": 0.0, "mem_util_gb": 0.0, "vram_free_mb": 0.0}
    try:
        # CPU Util
        cpu_cmd = "top -l 1 | grep 'CPU usage' | awk '{print $3}' | tr -d '%'"
        cpu_out = subprocess.check_output(cpu_cmd, shell=True).decode().strip()
        stats["cpu_util"] = float(cpu_out) if cpu_out else 0.0

        # Memory Util (approximate via vm_stat)
        vm_out = subprocess.check_output("vm_stat", shell=True).decode()
        page_size = 4096
        for line in vm_out.splitlines():
            if "page size" in line:
                # "Mach Virtual Memory Statistics: (page size of 16384 bytes)"
                match = re.search(r'size of (\d+)', line)
                if match:
                    page_size = int(match.group(1))
            if "Pages active:" in line:
                match = re.search(r':\s+(\d+)\.', line)
                if match:
                    active = int(match.group(1))
                    stats["mem_util_gb"] = round((active * page_size) / (1024**3), 2)
        
        # VRAM (very approximate via system_profiler)
        vram_cmd = "system_profiler SPDisplaysDataType | grep VRAM | awk '{print $2}'"
        vram_out = subprocess.check_output(vram_cmd, shell=True).decode().strip()
        if vram_out:
             match = re.search(r'(\d+)', vram_out)
             if match:
                 stats["vram_free_mb"] = float(match.group(1))

    except Exception as e:
        logger.warning(f"Telemetry hardware sampling failed: {e}")
    
    return stats

def log_telemetry(event_type: str, metrics: Dict[str, Any]):
    """Appends structured telemetry to telemetry.log"""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "event_type": event_type,
        "metrics": metrics
    }
    with open(TELEMETRY_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")

def token_telemetry_callback(
    callback_context: CallbackContext, llm_response: LlmResponse
) -> Optional[LlmResponse]:
    """ADK after_model_callback to track tokens."""
    agent_name = callback_context.agent_name
    usage = getattr(llm_response, "usage", None)
    
    if usage:
        # Compatibility check for different ADK versions/model providers
        prompt_tokens = getattr(usage, "prompt_token_count", 0)
        completion_tokens = getattr(usage, "candidates_token_count", 0)
        
        # Hardware sampling
        hw = get_mac_hardware_stats()
        
        log_telemetry("token_usage", {
            "agent": agent_name,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            **hw
        })
    
    return None # Proceed as normal

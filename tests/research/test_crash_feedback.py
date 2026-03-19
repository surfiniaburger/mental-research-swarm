"""
Canon TDD Test Suite: Crash Feedback Loop

Test List (from implementation_plan.md):
6. [x] test_crash_feedback_reads_log — read_crash_log() returns last 30 lines
7. [x] test_crash_feedback_empty_log — Returns empty string for missing/empty log
8. [x] test_crash_context_formats_for_prompt — Crash context is formatted for injection
"""
import pytest
from swarm.drivers import ResearchProtocolDriver


class FakeMCP:
    """Layer 4: External System Stub (BDD four-layer model)."""
    def __init__(self, responses=None):
        self.responses = responses or {}
        self.calls = []
    
    async def call_tool(self, name, args=None):
        self.calls.append((name, args))
        return self.responses.get(name, "")


@pytest.mark.asyncio
async def test_crash_feedback_reads_log():
    """GIVEN a run.log with a traceback,
    WHEN read_crash_log is called,
    THEN it should return the last 30 lines as crash context."""
    crash_output = "\n".join([f"line {i}" for i in range(50)])
    mcp = FakeMCP(responses={"read_research_file": crash_output})
    driver = ResearchProtocolDriver(mcp)
    
    result = await driver.read_crash_log()
    
    lines = result.strip().split("\n")
    assert len(lines) == 30
    assert "line 49" in result  # Last line present


@pytest.mark.asyncio
async def test_crash_feedback_empty_log():
    """GIVEN no run.log or an empty one,
    WHEN read_crash_log is called,
    THEN it should return an empty string gracefully."""
    mcp = FakeMCP(responses={"read_research_file": ""})
    driver = ResearchProtocolDriver(mcp)
    
    result = await driver.read_crash_log()
    
    assert result == ""


@pytest.mark.asyncio
async def test_crash_context_formats_for_prompt():
    """GIVEN crash log content with a Python traceback,
    WHEN read_crash_log is called,
    THEN the result should contain the traceback for prompt injection."""
    traceback_text = (
        "Traceback (most recent call last):\n"
        '  File "train.py", line 42, in <module>\n'
        "    model = Transformer()\n"
        "TypeError: __init__() got unexpected keyword argument 'scheduler'\n"
    )
    mcp = FakeMCP(responses={"read_research_file": traceback_text})
    driver = ResearchProtocolDriver(mcp)
    
    result = await driver.read_crash_log()
    
    assert "TypeError" in result
    assert "train.py" in result

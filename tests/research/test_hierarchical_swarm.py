import pytest
import asyncio
from typing import AsyncGenerator
from google.adk.agents import BaseAgent, LlmAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.adk.sessions import InMemorySessionService
from swarm.agents import ManagerAgent, ResearchAgent, SkillWriterAgent, CriticAgent
from swarm.drivers import ResearchProtocolDriver, SkillWriterProtocolDriver

class MockAgent(LlmAgent):
    """Mock LLM Agent that simulates output."""
    output_content: str = ""
    model_config = {"arbitrary_types_allowed": True, "extra": "allow"}

    def __init__(self, name, model, output_content, output_key):
        super().__init__(name=name, model=model, output_key=output_key, output_content=output_content)

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        ctx.session.state[self.output_key] = self.output_content
        # Yield a dummy event
        yield Event(author=self.name)

class MockDriver:
    async def update_skill(self, content): pass
    async def call_tool(self, name, args): return "mock_val"

@pytest.mark.asyncio
async def test_manager_agent_orchestration():
    # Setup mocks
    brain = MockAgent("Brain", "qwen3.5", "Propose: higher LR", "brain_out")
    brain.driver = MockDriver()
    
    hands = MockAgent("Hands", "qwen3.5-coder", "def new_train(): pass", "hands_out")
    # For Hands, we need to mock the ResearchAgent's specific validation
    # ResearchAgent logic sets validated_code in ctx.session.state
    
    critic = MockAgent("Critic", "qwen3.5", "APPROVE: looks good", "critic_out")
    
    manager = ManagerAgent("Manager", brain, hands, critic)
    
    # Setup session
    session_service = InMemorySessionService()
    ctx = InvocationContext(
        session=await session_service.create_session(
            app_name="app", 
            user_id="user", 
            session_id="session", 
            state={
                "results_tsv": "val_bpb\n1.5",
                "program_md": "# strategy"
            }
        ),
        session_service=session_service,
        invocation_id="test_inv",
        agent=manager
    )
    
    # Execute Manager
    events = []
    async for event in manager.run_async(ctx):
        events.append(event)
        
    # Assertions
    # 1. Check if Contextual Packets were prepared
    assert "results_packet" in ctx.session.state
    assert "strategy_packet" in ctx.session.state
    
    # 2. Check if sub-agents were called (via their output in state)
    assert ctx.session.state["brain_out"] == "Propose: higher LR"
    assert ctx.session.state["hands_out"] == "def new_train(): pass"
    assert ctx.session.state["critic_out"] == "APPROVE: looks good"

@pytest.mark.asyncio
async def test_manager_agent_rejection_logic():
    brain = MockAgent("Brain", "qwen3.5", "...", "brain_out")
    brain.driver = MockDriver()
    hands = MockAgent("Hands", "qwen3.5-coder", "...", "hands_out")
    # Simulate a rejection from Critic
    critic = MockAgent("Critic", "qwen3.5", "REJECT: potential crash", "critic_out")
    
    manager = ManagerAgent("Manager", brain, hands, critic)
    
    session_service = InMemorySessionService()
    ctx = InvocationContext(
        session=await session_service.create_session(
            app_name="app", 
            user_id="user", 
            session_id="session", 
            state={
                "validated_code": "some_code_that_will_be_invalidated"
            }
        ),
        session_service=session_service,
        invocation_id="test_inv_reject",
        agent=manager
    )
    
    async for _ in manager.run_async(ctx): pass
    
    # Assert that validated_code was cleared on rejection
    assert ctx.session.state["validated_code"] is None

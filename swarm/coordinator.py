import logging
import asyncio
from datetime import datetime
from typing import AsyncGenerator
from google.adk.agents import BaseAgent, LlmAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from .agents import ResearchAgent, SkillWriterAgent

logger = logging.getLogger("swarm.coordinator")


class SwarmCoordinator(BaseAgent):
    """
    Orchestrator that drives the FULL research lifecycle:
      1. TheBrain analyzes results → proposes strategy
      2. TheHands generates code → validates via AST
      3. Driver writes code → runs experiment → logs results
      4. Repeat

    Unlike a pure LoopAgent (which just chains LLM calls),
    this coordinator interleaves LLM inference WITH real
    experiment execution via the protocol drivers.
    """
    research_agent: ResearchAgent
    skill_writer: SkillWriterAgent
    max_iterations: int = 100
    model_config = {"arbitrary_types_allowed": True}

    def __init__(
        self,
        name: str,
        research_agent: ResearchAgent,
        skill_writer: SkillWriterAgent,
        max_iterations: int = 100
    ):
        super().__init__(
            name=name,
            sub_agents=[skill_writer, research_agent],
            research_agent=research_agent,
            skill_writer=skill_writer,
            max_iterations=max_iterations
        )

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        # ── Phase 1: Environment Setup ───────────────────────────
        logger.info("═" * 60)
        logger.info("  PHASE 1: ENVIRONMENT SETUP")
        logger.info("═" * 60)
        
        success = await self.research_agent.driver.ensure_setup()
        if not success:
            logger.error("❌ Environment initialization FAILED.")
            return
        logger.info("✅ Environment initialized.")

        # Load initial files into session state
        ctx.session.state["program_md"] = await self.research_agent.driver.mcp.call_tool(
            "read_research_file", {"path": "program.md"}
        )
        ctx.session.state["train_py"] = await self.research_agent.driver.mcp.call_tool(
            "read_research_file", {"path": "train.py"}
        )

        # ── Phase 2: Baseline ────────────────────────────────────
        logger.info("═" * 60)
        logger.info("  PHASE 2: BASELINE EXPERIMENT")
        logger.info("═" * 60)
        
        baseline = await self.research_agent.driver.run_experiment("baseline-reset")
        await self.research_agent.driver.log_result(baseline)
        ctx.session.state["latest_bpb"] = baseline.val_bpb
        logger.info(f"📊 Baseline: val_bpb={baseline.val_bpb} | status={baseline.status}")

        # Reload train.py and results after baseline
        ctx.session.state["train_py"] = await self.research_agent.driver.mcp.call_tool(
            "read_research_file", {"path": "train.py"}
        )
        ctx.session.state["results_tsv"] = await self.research_agent.driver.mcp.call_tool(
            "read_research_file", {"path": "results.tsv"}
        )

        # ── Phase 3: Autonomous Loop ─────────────────────────────
        logger.info("═" * 60)
        logger.info(f"  PHASE 3: AUTONOMOUS LOOP (max {self.max_iterations} iterations)")
        logger.info("═" * 60)

        for iteration in range(1, self.max_iterations + 1):
            ctx.session.state["iteration"] = iteration
            
            logger.info("─" * 60)
            logger.info(f"  🔄 ITERATION {iteration}/{self.max_iterations}")
            logger.info("─" * 60)

            # ── Step A: TheBrain analyzes and proposes ────────────
            logger.info(f"  🧠 [{self.skill_writer.name}] Analyzing results & proposing strategy...")
            
            async for event in self.skill_writer.run_async(ctx):
                yield event
                
            brain_output = ctx.session.state.get(self.skill_writer.output_key, "")
            if brain_output:
                logger.info(f"  📝 TheBrain proposal: {brain_output[:150].replace(chr(10), ' ')}...")
                # Update program.md with the new insights
                await self.skill_writer.driver.update_skill(brain_output)
                # Reload program.md into state
                ctx.session.state["program_md"] = await self.research_agent.driver.mcp.call_tool(
                    "read_research_file", {"path": "program.md"}
                )
                logger.info("  ✅ program.md updated with new insights.")
            else:
                logger.warning("  ⚠️ TheBrain returned empty output, skipping skill update.")

            # ── Step B: TheHands generates code ───────────────────
            logger.info(f"  🔧 [{self.research_agent.name}] Generating modified train.py...")
            
            async for event in self.research_agent.run_async(ctx):
                yield event
            
            validated_code = ctx.session.state.get("validated_code")
            if validated_code:
                logger.info(f"  ✅ Code validated ({len(validated_code)} chars). Writing to disk...")
                await self.research_agent.driver.mcp.call_tool(
                    "write_research_file",
                    {"path": "train.py", "content": validated_code}
                )
                ctx.session.state["train_py"] = validated_code
            else:
                logger.warning("  ⚠️ Code validation failed. Skipping experiment, retrying next iteration.")
                continue

            # ── Step C: Run the experiment ─────────────────────────
            logger.info(f"  🚀 Running experiment (iteration {iteration})...")
            result = await self.research_agent.driver.run_experiment(f"iteration-{iteration}")
            await self.research_agent.driver.log_result(result)
            
            prev_bpb = ctx.session.state.get("latest_bpb", 999)
            ctx.session.state["latest_bpb"] = result.val_bpb
            
            # Reload results.tsv for the next brain analysis
            ctx.session.state["results_tsv"] = await self.research_agent.driver.mcp.call_tool(
                "read_research_file", {"path": "results.tsv"}
            )

            # ── Step D: Log outcome ───────────────────────────────
            delta = prev_bpb - result.val_bpb
            if result.status == "crash":
                logger.info(f"  💥 CRASH — reverting to previous code.")
                # Revert train.py by re-reading the git version
                await self.research_agent.driver.mcp.call_tool(
                    "execute_command",
                    {"command": "git checkout train.py", "cwd": self.research_agent.driver.repo_path}
                )
                ctx.session.state["train_py"] = await self.research_agent.driver.mcp.call_tool(
                    "read_research_file", {"path": "train.py"}
                )
            elif delta > 0:
                logger.info(f"  📈 IMPROVED! val_bpb={result.val_bpb:.4f} (Δ={delta:+.4f})")
            else:
                logger.info(f"  📉 Regressed. val_bpb={result.val_bpb:.4f} (Δ={delta:+.4f})")

            logger.info(f"  📊 Current best: {min(prev_bpb, result.val_bpb):.4f}")

        logger.info("═" * 60)
        logger.info(f"  🏁 SWARM CONCLUDED after {self.max_iterations} iterations")
        logger.info("═" * 60)

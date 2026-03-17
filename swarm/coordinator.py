import logging
import asyncio
import os
from datetime import datetime
from typing import AsyncGenerator
from google.adk.agents import BaseAgent, LlmAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from .agents import ResearchAgent, SkillWriterAgent, CriticAgent, ManagerAgent

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
    manager_agent: ManagerAgent
    max_iterations: int = 100
    model_config = {"arbitrary_types_allowed": True}

    def __init__(
        self,
        name: str,
        manager_agent: ManagerAgent,
        max_iterations: int = 100
    ):
        super().__init__(
            name=name,
            sub_agents=[manager_agent],
            manager_agent=manager_agent,
            max_iterations=max_iterations
        )

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        # ── Phase 1: Environment Setup ───────────────────────────
        logger.info("═" * 60)
        logger.info("  PHASE 1: ENVIRONMENT SETUP")
        logger.info("═" * 60)
        
        success = await self.manager_agent.hands.driver.ensure_setup()
        if not success:
            logger.error("❌ Environment initialization FAILED.")
            return
        logger.info("✅ Environment initialized.")

        # Ensure docs directory exists
        repo_path = self.manager_agent.hands.driver.repo_path
        docs_dir = os.path.join(repo_path, "docs")
        os.makedirs(docs_dir, exist_ok=True)
        
        # Inject context for ManagerAgent
        ctx.session.state["research_dir"] = repo_path

        # Load initial files into session state
        ctx.session.state["program_md"] = await self.manager_agent.hands.driver.mcp.call_tool(
            "read_research_file", {"path": "program.md"}
        )
        ctx.session.state["train_py"] = await self.manager_agent.hands.driver.mcp.call_tool(
            "read_research_file", {"path": "train.py"}
        )

        # ── Phase 2: Baseline ────────────────────────────────────
        logger.info("═" * 60)
        logger.info("  PHASE 2: BASELINE EXPERIMENT")
        logger.info("═" * 60)
        
        baseline = await self.manager_agent.hands.driver.run_experiment("baseline-reset")
        await self.manager_agent.hands.driver.log_result(baseline)
        ctx.session.state["latest_bpb"] = baseline.val_bpb
        logger.info(f"📊 Baseline: val_bpb={baseline.val_bpb} | status={baseline.status}")

        # Reload train.py and results after baseline
        ctx.session.state["train_py"] = await self.manager_agent.hands.driver.mcp.call_tool(
            "read_research_file", {"path": "train.py"}
        )
        ctx.session.state["results_tsv"] = await self.manager_agent.hands.driver.mcp.call_tool(
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

            try:
                # ── Steps A, B, C: Hierarchical Management ────────────
                logger.info(f"  🏢 [{self.manager_agent.name}] Delegating iteration tasks...")
                
                max_retries = 3
                current_retry = 0
                while current_retry < max_retries:
                    try:
                        async for event in self.manager_agent.run_async(ctx):
                            yield event
                        break # Success
                    except Exception as e:
                        if "Timeout" in str(e) or "APIConnectionError" in str(e):
                            current_retry += 1
                            wait_time = 2 ** current_retry
                            logger.warning(f"  ⚠️ Timeout detected ({e}). Retry {current_retry}/{max_retries} in {wait_time}s...")
                            import asyncio
                            await asyncio.sleep(wait_time)
                        else:
                            raise e
                
                # Check for results after management orchestration
                validated_code = ctx.session.state.get("validated_code")
                target_node = ctx.session.state.get("target_node")

                if validated_code:
                    if target_node:
                        logger.info(f"  ✅ Snippet approved. Patching '{target_node}' in train.py...")
                        await self.manager_agent.hands.driver.mcp.call_tool(
                            "patch_research_file",
                            {
                                "path": "train.py",
                                "target_node": target_node,
                                "new_content": validated_code
                            }
                        )
                    else:
                        logger.info(f"  ✅ Code approved. Writing full train.py...")
                        await self.manager_agent.hands.driver.mcp.call_tool(
                            "write_research_file",
                            {"path": "train.py", "content": validated_code}
                        )
                    
                    # Update local state with the new full file content
                    ctx.session.state["train_py"] = await self.manager_agent.hands.driver.mcp.call_tool(
                        "read_research_file", {"path": "train.py"}
                    )
                else:
                    logger.warning("  ⚠️ Code rejected or failed validation. Skipping experiment.")
                    continue

                # ── Step D: Run the experiment ─────────────────────────
                logger.info(f"  🚀 Running experiment (iteration {iteration})...")
                result = await self.manager_agent.hands.driver.run_experiment(f"iteration-{iteration}")
                await self.manager_agent.hands.driver.log_result(result)
                
                prev_bpb = ctx.session.state.get("latest_bpb", 999)
                ctx.session.state["latest_bpb"] = result.val_bpb
                
                # Reload results.tsv for the next brain analysis
                ctx.session.state["results_tsv"] = await self.manager_agent.hands.driver.mcp.call_tool(
                    "read_research_file", {"path": "results.tsv"}
                )

                # ── Step E: Log outcome ───────────────────────────────
                delta = prev_bpb - result.val_bpb
                if result.status == "crash":
                    logger.info(f"  💥 CRASH — reverting to previous code.")
                    # Revert train.py by re-reading the git version
                    await self.manager_agent.hands.driver.mcp.call_tool(
                        "execute_command",
                        {"command": "git checkout train.py", "cwd": self.manager_agent.hands.driver.repo_path}
                    )
                    ctx.session.state["train_py"] = await self.manager_agent.hands.driver.mcp.call_tool(
                        "read_research_file", {"path": "train.py"}
                    )
                elif delta > 0:
                    logger.info(f"  📈 IMPROVED! val_bpb={result.val_bpb:.4f} (Δ={delta:+.4f})")
                else:
                    logger.info(f"  📉 Regressed. val_bpb={result.val_bpb:.4f} (Δ={delta:+.4f})")

                logger.info(f"  📊 Current best: {min(prev_bpb, result.val_bpb):.4f}")

            except Exception as e:
                logger.error(f"  ❌ Iteration {iteration} FAILED: {e}")
                logger.info("  🔄 Attempting to recover for next iteration...")
                # Ensure we have fresh state for next time
                try:
                    ctx.session.state["train_py"] = await self.manager_agent.hands.driver.mcp.call_tool(
                        "read_research_file", {"path": "train.py"}
                    )
                except:
                    pass
                continue

        logger.info("═" * 60)
        logger.info(f"  🏁 SWARM CONCLUDED after {self.max_iterations} iterations")
        logger.info("═" * 60)

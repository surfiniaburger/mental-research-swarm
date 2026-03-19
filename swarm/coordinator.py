import logging
import asyncio
import os
from datetime import datetime
from typing import AsyncGenerator, Any
from google.adk.agents import BaseAgent, LlmAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
import taipy as tp
from .taipy_orchestrator import configure_tao, init_taipy_orchestrator
from .agents import ResearchAgent, SkillWriterAgent, CriticAgent, ManagerAgent
from .telemetry import get_mac_hardware_stats, log_telemetry

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
    session_state_path: str = ""
    start_iteration: int = 1
    global_best_bpb: float = 999.0
    scenario_cfg: Any = None
    taipy_core: Any = None
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
        self.session_state_path = os.path.join(self.manager_agent.hands.driver.repo_path, "docs", "session_state.json")
        self._load_session_state()
        
        # Initialize Taipy Core
        self.scenario_cfg = configure_tao()
        self.taipy_core = init_taipy_orchestrator()

    def _load_session_state(self):
        """Loads persistent session state from disk."""
        if os.path.exists(self.session_state_path):
            try:
                import json
                with open(self.session_state_path, "r") as f:
                    state = json.load(f)
                    self.start_iteration = state.get("iteration", 1)
                    self.global_best_bpb = state.get("global_best_bpb", 999.0)
                    logger.info(f"📂 Loaded session state: iteration={self.start_iteration}, best_bpb={self.global_best_bpb}")
            except Exception as e:
                logger.error(f"⚠️ Failed to load session state: {e}")
                self.start_iteration = 1
                self.global_best_bpb = 999.0
        else:
            self.start_iteration = 1
            self.global_best_bpb = 999.0

    def _save_session_state(self, iteration: int, global_best_bpb: float):
        """Saves session state to disk."""
        try:
            import json
            state = {
                "iteration": iteration,
                "global_best_bpb": global_best_bpb,
                "timestamp": datetime.now().isoformat()
            }
            with open(self.session_state_path, "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"⚠️ Failed to save session state: {e}")

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
        if self.global_best_bpb < 900:
            logger.info("═" * 60)
            logger.info(f"  PHASE 2: RESUMING SESSION (Best BPB: {self.global_best_bpb:.4f})")
            logger.info("═" * 60)
            ctx.session.state["global_best_bpb"] = self.global_best_bpb
        else:
            logger.info("═" * 60)
            logger.info("  PHASE 2: BASELINE EXPERIMENT")
            logger.info("═" * 60)
            
            baseline = await self.manager_agent.hands.driver.run_experiment("baseline-reset")
            await self.manager_agent.hands.driver.log_result(baseline)
            ctx.session.state["latest_bpb"] = baseline.val_bpb
            ctx.session.state["global_best_bpb"] = baseline.val_bpb
            self.global_best_bpb = baseline.val_bpb
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

        for iteration in range(self.start_iteration, self.max_iterations + 1):
            ctx.session.state["iteration"] = iteration
            
            # --- Taipy Scenario Create ---
            scenario = tp.create_scenario(self.scenario_cfg, name=f"Iteration-{iteration}")
            scenario.iteration.write(iteration)
            if "program_md" in ctx.session.state:
                scenario.input_program.write(ctx.session.state["program_md"])
            if "train_py" in ctx.session.state:
                scenario.input_train_py.write(ctx.session.state["train_py"])
            
            logger.info("─" * 60)
            logger.info(f"  🔄 ITERATION {iteration}/{self.max_iterations}")
            logger.info("─" * 60)

            # Sample Hardware Telemetry at start of iteration
            hw_stats = get_mac_hardware_stats()
            log_telemetry("heartbeat", {"iteration": iteration, **hw_stats})

            try:
                # ── Steps A, B, C: Hierarchical Management ────────────
                logger.info(f"  🏢 [{self.manager_agent.name}] Delegating iteration tasks...")
                
                # Initialize crash_feedback for this iteration (cleared if no crash last time)
                if "crash_feedback" not in ctx.session.state:
                    ctx.session.state["crash_feedback"] = ""
                
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
                
                # --- Taipy Scenario Log Results ---
                scenario.research_result.write(result)
                
                prev_bpb = ctx.session.state.get("latest_bpb", 999)
                ctx.session.state["latest_bpb"] = result.val_bpb
                
                # Reload results.tsv for the next brain analysis
                ctx.session.state["results_tsv"] = await self.manager_agent.hands.driver.mcp.call_tool(
                    "read_research_file", {"path": "results.tsv"}
                )

                # ── Step E: Log outcome ───────────────────────────────
                delta = prev_bpb - result.val_bpb
                global_best = ctx.session.state.get("global_best_bpb", 999.0)
                
                if result.status == "crash":
                    logger.info(f"  💥 CRASH — reverting to previous code.")
                    
                    # 📋 CRASH FEEDBACK: Read run.log so Brain learns from the failure
                    crash_log = await self.manager_agent.hands.driver.read_crash_log()
                    if crash_log:
                        ctx.session.state["crash_feedback"] = (
                            f"### ⚠️ LAST EXPERIMENT CRASHED (Iteration {iteration})\n"
                            f"The following error was captured from run.log:\n```\n{crash_log}\n```\n"
                            f"Analyze this error and ensure your next proposal does NOT repeat this failure pattern."
                        )
                        logger.info(f"  📋 Crash log captured ({len(crash_log)} chars) for next iteration.")
                    else:
                        ctx.session.state["crash_feedback"] = (
                            f"### ⚠️ LAST EXPERIMENT CRASHED (Iteration {iteration})\n"
                            f"The run.log was empty — the script failed before producing any output.\n"
                            f"This likely means a syntax error, missing import, or incompatible class structure.\n"
                            f"ENSURE your next proposal compiles and runs correctly."
                        )
                        logger.info(f"  📋 Empty crash log — likely syntax/import error.")
                    
                    # Revert train.py by re-reading the git version
                    await self.manager_agent.hands.driver.mcp.call_tool(
                        "execute_command",
                        {"command": "git checkout train.py", "cwd": self.manager_agent.hands.driver.repo_path}
                    )
                    ctx.session.state["train_py"] = await self.manager_agent.hands.driver.mcp.call_tool(
                        "read_research_file", {"path": "train.py"}
                    )
                else:
                    # Clear crash feedback on success — no need to carry it forward
                    ctx.session.state["crash_feedback"] = ""
                    if result.val_bpb < global_best:
                        ctx.session.state["global_best_bpb"] = result.val_bpb
                        global_best = result.val_bpb
                        logger.info(f"  📈 NEW GLOBAL BEST! val_bpb={result.val_bpb:.4f} (Δ={delta:+.4f})")
                    elif delta > 0:
                        logger.info(f"  📈 IMPROVED! val_bpb={result.val_bpb:.4f} (Δ={delta:+.4f})")
                    else:
                        logger.info(f"  📉 Regressed. val_bpb={result.val_bpb:.4f} (Δ={delta:+.4f})")
 
                logger.info(f"  📊 Session Best: {global_best:.4f}")
                self._save_session_state(iteration, global_best)

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

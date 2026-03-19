import logging
import ast
import os
from typing import AsyncGenerator, Any, Optional, List
from google.adk.agents import LlmAgent, BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from .prompt import PromptEnvelope, build_prompt_messages
from .drivers import ResearchProtocolDriver, ResearchResult, SkillWriterProtocolDriver
from .telemetry import token_telemetry_callback

logger = logging.getLogger(__name__)

class AntiRedundancyFilter:
    """🛡️ Identifies research stagnation by monitoring performance deltas."""
    def __init__(self, threshold: float = 0.005, window: int = 3):
        self.threshold = threshold
        self.window = window

    def check(self, history: List[ResearchResult]) -> bool:
        """Returns True if the last 'window' results show no improvement above 'threshold'."""
        if len(history) < self.window:
            return False
        
        # Look at the last 'window' results
        recent = history[-self.window:]
        bpbs = [r.val_bpb for r in recent if r.val_bpb > 0]
        
        if len(bpbs) < self.window:
            return False
            
        # Calculate max improvement spread in this window
        # We assume lower BPB is better. 
        # So we look at max(bpbs) - min(bpbs)
        max_delta = max(bpbs) - min(bpbs)
        return max_delta < self.threshold


from dataclasses import dataclass

@dataclass
class DiversifierResult:
    """Communicates the outcome of a diversifier check."""
    allowed: bool
    category: str
    reason: str = ""


class StrategyDiversifier:
    """🎯 Prevents the Brain from proposing the same category of change repeatedly.
    
    Tracks strategy categories across iterations. If the same category
    appears more than `max_streak` times consecutively, the strategy is rejected
    to force exploration of different research axes.
    """
    
    # Category keywords — each list defines a research axis
    CATEGORY_KEYWORDS = {
        "optimizer": ["optimizer", "adam", "adamw", "sgd", "learning_rate", "lr_scheduler",
                      "cosineanneal", "reducelronplateau", "configure_optimizers", "weight_decay",
                      "momentum", "warmup", "eta_min", "t_mult"],
        "attention": ["attention", "self_attention", "multi_head", "flash_attention",
                      "kv_cache", "rotary", "rope", "alibi", "causal_mask"],
        "architecture": ["transformer", "layer_norm", "feedforward", "mlp", "embedding",
                         "positional", "residual", "dropout", "hidden_dim", "num_layers",
                         "num_heads", "expert", "moe", "gating"],
        "loss": ["loss", "cross_entropy", "label_smoothing", "distillation",
                 "auxiliary_loss", "regularization", "l2_norm", "entropy"],
        "data": ["batch_size", "sequence_length", "tokenizer", "curriculum",
                 "data_augmentation", "sampling", "shuffle"],
    }
    
    def __init__(self, max_streak: int = 2):
        self.max_streak = max_streak
        self.history: List[str] = []
    
    def classify(self, strategy_text: str) -> str:
        """Classify a strategy into a research category based on keyword density."""
        text_lower = strategy_text.lower()
        scores = {}
        for category, keywords in self.CATEGORY_KEYWORDS.items():
            scores[category] = sum(1 for kw in keywords if kw in text_lower)
        
        best_category = max(scores, key=scores.get)
        return best_category if scores[best_category] > 0 else "unknown"
    
    def check(self, strategy_text: str) -> DiversifierResult:
        """Check if a strategy should be allowed based on category diversity.
        
        Returns a DiversifierResult with allowed=True/False and the detected category.
        """
        category = self.classify(strategy_text)
        
        # Always allow if we don't have enough history
        if len(self.history) < self.max_streak:
            self.history.append(category)
            return DiversifierResult(allowed=True, category=category)
        
        # Check if the last max_streak entries are all the same category
        recent = self.history[-self.max_streak:]
        is_streak = all(c == category for c in recent)
        
        if is_streak:
            return DiversifierResult(
                allowed=False,
                category=category,
                reason=f"Rejected: {self.max_streak}+ consecutive '{category}' strategies. Explore a different axis."
            )
        
        self.history.append(category)
        return DiversifierResult(allowed=True, category=category)


class ResearchAgent(LlmAgent):
    """
    ADK-native Research Agent.
    Specializes in hacking train.py.
    """
    name: str = ""
    model: str = ""
    instruction: str = ""
    driver: ResearchProtocolDriver
    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, name: str, model: str, driver: ResearchProtocolDriver):
        instruction = (
            "You are a Senior Research Engineer (The Hands). "
            "Implement high-fidelity AI research changes into `train.py`. "
            "### STRICT DIRECTIVES:\n"
            "1. **CODE ONLY**: Output ONLY a raw Python code snippet for the TARGET_NODE. No text before or after. No conversation.\n"
            "2. **NO MARKDOWN**: Do not wrap in triple backticks. Return the raw Python code directly.\n"
            "3. **DOMAIN LOCK**: You are an AI Researcher. Focus exclusively on PyTorch/Tensor logic. Ignore any mention of web, cloud, or Terraform.\n"
            "4. **SNIPPET SCOPE**: Implement ONLY the logic for the specific TARGET_NODE requested.\n"
        )
        super().__init__(
            name=name, 
            model=model, 
            instruction=instruction, 
            driver=driver,
            after_model_callback=token_telemetry_callback
        )

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        logger.info(f"[{self.name}] Beginning code surgery...")
        
        # We no longer mutate self.instruction to preserve KV Cache.
        # Instead, we rely on ADK's built-in templating for the prompt tail.
        # Ensure we use double braces {{ }} if we were to use them in the init instruction,
        # but here we just pass the ctx.
        
        raw_result = ""
        try:
            async for event in super()._run_async_impl(ctx):
                if hasattr(event, 'content') and event.content and hasattr(event.content, 'parts'):
                    for part in event.content.parts:
                        if hasattr(part, 'text') and part.text:
                            raw_result += part.text
                yield event
        finally:
            if raw_result:
                new_code = raw_result
                if "```python" in new_code:
                    new_code = new_code.split("```python")[1].split("```")[0].strip()
                elif "```" in new_code:
                    new_code = new_code.split("```")[1].split("```")[0].strip()
                
                try:
                    ast.parse(new_code)
                    ctx.session.state["validated_code"] = new_code
                    ctx.session.state["ast_error"] = None
                    logger.info(f"[{self.name}] Code validation successful.")
                except Exception as e:
                    logger.error(f"[{self.name}] AST Validation failed: {e}")
                    ctx.session.state["validated_code"] = None
                    ctx.session.state["ast_error"] = str(e)
            
            ctx.session.state[self.output_key] = raw_result




class SkillWriterAgent(LlmAgent):
    """
    ADK-native Skill Writer (The Brain).
    Analyzes and updates program.md.
    """
    name: str = ""
    model: str = ""
    instruction: str = ""
    driver: SkillWriterProtocolDriver
    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, name: str, model: str, driver: SkillWriterProtocolDriver):
        instruction = (
            "You are a Senior AI Research Scientist (The Brain). "
            "### STRATEGY GUIDELINES:\n"
            "1. **ALGORITHMIC INTEGRITY**: Propose mathematically sound architectural changes. NEVER suggest a 'crash' or 'stub' for testing.\n"
            "2. **CONTINUITY**: Build upon SUCCESSFUL history in the archives. Ignore hallucinated failures in the active TSV if they lack technical depth.\n"
            "3. **TOTAL FREEDOM**: You are not limited to MoE. Explore Attention, Optimizers, or Curricula.\n"
            "4. **AVOID LOCAL MINIMA**: If results plateau, pivot to a new Era.\n"
            "5. **DIVERSIFICATION MANDATE**: Do NOT propose optimizer-only changes for more than 2 consecutive iterations. "
            "Alternate between research axes: architecture, attention, loss functions, data processing, and optimizers.\n"
            "6. **CRASH AWARENESS**: If crash feedback is provided below, analyze the error and ensure your next proposal avoids the same failure pattern.\n\n"
            "RESULTS:\n{results_packet}\n\n"
            "CHRONICLE:\n{research_chronicle}\n\n"
            "FULL_STRATEGY:\n{strategy_packet}\n\n"
            "CRASH_FEEDBACK:\n{crash_feedback}\n\n"
            "Return target node as: TARGET_NODE: [NodeName]"
        )
        super().__init__(
            name=name, 
            model=model, 
            instruction=instruction, 
            driver=driver,
            after_model_callback=token_telemetry_callback
        )

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        logger.info(f"[{self.name}] Analyzing results via shared documents...")
        
        raw_result = ""
        async for event in super()._run_async_impl(ctx):
            if hasattr(event, 'content') and event.content and hasattr(event.content, 'parts'):
                for part in event.content.parts:
                    if hasattr(part, 'text') and part.text:
                        raw_result += part.text
            yield event
            
        ctx.session.state[self.output_key] = raw_result


class CriticAgent(LlmAgent):
    """
    ADK-native Code Critic.
    Reviews train.py for logical errors and alignment with strategy.
    """
    name: str = ""
    model: str = ""
    instruction: str = ""
    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, name: str, model: str):
        instruction = (
            "You are a Senior Code Reviewer (The Critic). "
            "Review the proposed changes against the strategy and PREVIOUS successful patches.\n\n"
            "STRATEGY:\n{strategy_packet}\n\n"
            "OLD_CODE:\n{target_snippet}\n\n"
            "NEW_CODE:\n{validated_code}\n\n"
            "ANTI-REDUNDANCY FILTER: If the proposed code is structurally 90% identical to previous "
            "successful versions but doesn't offer a clear theoretical breakthrough, REJECT it. "
            "Start your response with 'APPROVE' or 'REJECT'."
        )
        super().__init__(
            name=name, 
            model=model, 
            instruction=instruction,
            after_model_callback=token_telemetry_callback
        )

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        logger.info(f"[{self.name}] Reviewing surgical patch...")
        
        raw_result = ""
        async for event in super()._run_async_impl(ctx):
            if hasattr(event, 'content') and event.content and hasattr(event.content, 'parts'):
                for part in event.content.parts:
                    if hasattr(part, 'text') and part.text:
                        raw_result += part.text
            yield event
            
        ctx.session.state[self.output_key] = raw_result


class ManagerAgent(BaseAgent):
    """
    Mid-Level Manager.
    Orchestrates the Brain, Hands, and Critic.
    Implements 'Contextual Packets' for efficient communication.
    """
    brain: Any = None
    hands: Any = None
    critic: Any = None
    redundancy_filter: Any = None
    strategy_diversifier: Any = None
    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, name: str, brain: BaseAgent, hands: BaseAgent, critic: BaseAgent):
        super().__init__(
            name=name,
            sub_agents=[brain, hands, critic]
        )
        self.brain = brain
        self.hands = hands
        self.critic = critic
        self.redundancy_filter = AntiRedundancyFilter(threshold=0.005, window=3)
        self.strategy_diversifier = StrategyDiversifier(max_streak=2)
        self._fibonacci_cache = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]

    def _is_fibonacci(self, n: int) -> bool:
        return n in self._fibonacci_cache

    def _prepare_contextual_packets(self, ctx: InvocationContext):
        """Summarizes full files into contextual packets and writes to docs/."""
        research_dir = ctx.session.state.get("research_dir", "research_env")
        docs_dir = os.path.join(research_dir, "docs")
        os.makedirs(docs_dir, exist_ok=True)

        # 0. Load Mission Anchor (Immutable Context)
        mission_path = os.path.join(docs_dir, "MISSION.md")
        mission_content = ""
        if os.path.exists(mission_path):
            with open(mission_path, "r") as f:
                mission_content = f.read()
        
        # 1. Summarize Results (last 5 entries)
        results = ctx.session.state.get("results_tsv", "")
        if results:
            lines = results.strip().split("\n")
            header = lines[0]
            last_entries = lines[-5:]
            results_packet = f"{header}\n" + "\n".join(last_entries)
            
            # Prepend mission to results packet
            if mission_content:
                results_packet = f"### PRIMARY MISSION ###\n{mission_content}\n\n### RECENT RESULTS ###\n{results_packet}"
            
            ctx.session.state["results_packet"] = results_packet
            with open(os.path.join(docs_dir, "results_summary.md"), "w") as f:
                f.write(f"# Results Summary\n\n{results_packet}")
        
        # 2. Summarize Strategy
        program = ctx.session.state.get("program_md", "")
        if program:
            strategy_packet = program[-1000:]
            ctx.session.state["strategy_packet"] = strategy_packet
            with open(os.path.join(docs_dir, "current_strategy.md"), "w") as f:
                f.write(f"# Current Strategy\n\n{strategy_packet}")

        # 3. Load Last Critique (Cross-Iteration Memory)
        critique_path = os.path.join(docs_dir, "last_critique.md")
        if os.path.exists(critique_path):
            with open(critique_path, "r") as f:
                last_critique = f.read()
                ctx.session.state["critic_feedback"] = last_critique

        # 4. Handle Research Chronicle (Strategic Memory)
        chronicle_path = os.path.join(docs_dir, "research_chronicle.md")
        archive_dir = os.path.join(docs_dir, "archive")
        os.makedirs(archive_dir, exist_ok=True)

        if not os.path.exists(chronicle_path):
            with open(chronicle_path, "w") as f:
                f.write("# Research Chronicle\n\n## Era 1: Initial Baseline\nSetting up the environment and establishing dense performance metrics.")
        
        # [ANNEALING] Check for stagnation and prune context harder
        results = ctx.session.state.get("results_tsv", "")
        is_stagnant = False
        if results:
            lines = [l for l in results.strip().split("\n") if l.strip()]
            if len(lines) > 4:
                # Calculate delta over last 3 iterations
                try:
                    vals = [float(l.split()[1]) for l in lines[-3:] if len(l.split()) > 1]
                    if len(vals) == 3 and abs(vals[0] - vals[2]) < 0.005:
                        is_stagnant = True
                except (ValueError, IndexError):
                    pass

        with open(chronicle_path, "r") as f:
            content = f.read()
            # Recursive Compression: If chronicle > 3000 chars OR stagnant, prune old eras
            if len(content) > 3000 or is_stagnant:
                logger.info(f"[{self.name}] Stagnation or Size Limit detected. Performing Context Annealing...")
                lines = content.split("\n")
                # Keep Header + Last 10 lines of context (Harder pruning during stagnation)
                keep_lines = 10 if is_stagnant else 20
                content = lines[0] + f"\n\n... (Stagnation Annealing Pruned Earlier Context) ...\n\n" + "\n".join(lines[-keep_lines:])
                
                # If stagnant, also WIPE the last critique to avoid negative roleplay loops
                if is_stagnant and os.path.exists(os.path.join(docs_dir, "last_critique.md")):
                    with open(os.path.join(docs_dir, "last_critique.md"), "w") as fc:
                        fc.write("# Strategic Annealing\nStagnation detected. Previous critiques purged to allow fresh architectural perspective.")
            
            ctx.session.state["research_chronicle"] = content

        # 5. Dynamic Skill Loading (DEPRECATED)
        pass

        # Dynamic Skill Loading (DEPRECATED: Swarm now leads itself)
        pass

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        # Reset iteration state to prevent stale code from previous iterations being applied
        ctx.session.state["validated_code"] = None
        ctx.session.state["ast_error"] = None
        ctx.session.state["critic_feedback"] = None
        ctx.session.state["target_node"] = None
        
        logger.info(f"[{self.name}] Orchestrating research loop with Contextual Packets...")
        
        self._prepare_contextual_packets(ctx)
        
        # 1. Ask Brain for strategy (Uses results_packet + chronicle)
        iteration = ctx.session.state.get("iteration", 1)
        logger.info(f"[{self.name}] Step 1: Brain (Strategy Update - Iteration {iteration})")
        
        # 🛡️ ANTI-REDUNDANCY FILTER
        history = await self.hands.driver.get_history()
        if self.redundancy_filter.check(history):
            logger.warning(f"[{self.name}] Stagnation detected. Triggering Anti-Redundancy rejection.")
            yield Event(
                name="CriticFeedback",
                data={
                    "feedback": (
                        "### Evaluation Result: REJECT (Failed Anti-Redundancy Filter)\n\n"
                        "**Reasoning:**\n"
                        "1. **Redundancy Analysis**: Performance delta has dropped below 0.005 across the last 3 iterations.\n"
                        "2. **Stagnation**: Research has plateaued. Re-submitting structural variations of existing logic is redundant.\n"
                        "3. **Feedback**: Please provide specific deviations or a new theoretical adjustment to break the plateau."
                    )
                }
            )
            # Prune context even harder to force a pivot
            self._prepare_contextual_packets(ctx)
            if "research_chronicle" in ctx.session.state:
                lines = ctx.session.state["research_chronicle"].split("\n")
                ctx.session.state["research_chronicle"] = lines[0] + "\n\n!!! CRITICAL STAGNATION !!!\nPruning history to force a radical pivot.\n" + "\n".join(lines[-5:])
            return

        async for event in self.brain.run_async(ctx):
            yield event
            
        brain_out = ctx.session.state.get(self.brain.output_key, "")
        if brain_out:
            # Domain Guard: Check for severe hallucinations (e.g. AWS/Terraform/FastAPI)
            hallucination_keywords = ["aws", "terraform", "fastapi", "ec2", "s3", "docker-compose", "hcl", "instance_type"]
            found_hallucinations = [kw for kw in hallucination_keywords if kw in brain_out.lower()]
            if found_hallucinations:
                logger.error(f"[{self.name}] DOMAIN CORRUPTION DETECTED: Brain is discussing {found_hallucinations}. ABORTING UPDATE.")
                # Self-Healing: Invalidate the bad strategy and inject a corrective mission
                ctx.session.state[self.brain.output_key] = "[REJECTED: Domain Corruption]"
                ctx.session.state["strategy_packet"] = "ERROR: Previous strategy rejected for domain shift. STICK TO AI RESEARCH ONLY."
                return # Skip this iteration's surgery and chronicle update

            # 🎯 STRATEGY DIVERSIFIER: Check if Brain is stuck on one axis
            div_result = self.strategy_diversifier.check(brain_out)
            if not div_result.allowed:
                logger.warning(f"[{self.name}] Strategy Diversifier REJECTED: {div_result.reason}")
                ctx.session.state["strategy_packet"] += (
                    f"\n\n⚠️ DIVERSIFICATION REQUIRED: Your last {self.strategy_diversifier.max_streak} strategies "
                    f"were all '{div_result.category}'. You MUST explore a different research axis "
                    f"(e.g., attention, architecture, loss, data). Optimizer changes will be auto-rejected."
                )
                # Don't return — let the Brain re-run with the diversification feedback
                # but skip the skill update to avoid writing stale strategy
            else:
                logger.info(f"[{self.name}] Strategy category: '{div_result.category}' (allowed)")
                await self.brain.driver.update_skill(brain_out)
            
            # Fibonacci Strategic Checkpoint (Era Shift)
            iteration = ctx.session.state.get("iteration", 1)
            if self._is_fibonacci(iteration):
                logger.info(f"[{self.name}] Fibonacci Milestone reached (Iter {iteration}). Archiving Era...")
                research_dir = ctx.session.state.get("research_dir", "research_env")
                docs_dir = os.path.join(research_dir, "docs")
                archive_dir = os.path.join(docs_dir, "archive")
                chronicle_path = os.path.join(docs_dir, "research_chronicle.md")
                
                # Archive full era logs
                era_file = os.path.join(archive_dir, f"era_{iteration}.md")
                with open(era_file, "w") as f:
                    f.write(f"# Strategic Archive - Iteration {iteration}\n\n{brain_out}")
                
                # Append concise milestone to the active Chronicle
                with open(chronicle_path, "a") as f:
                    f.write(f"\n\n## Era {iteration}: Fibonacci Pivot\n{brain_out[:500]}...")
                
                # Inspiration Shuffle: Inject Distal memory if available
                # Logic: If iteration > 13, inject Era 2 or 3 asDistal memory to Spark new thoughts.
                if iteration >= 13:
                    ctx.session.state["results_packet"] += "\n\n[INSPIRATION SHUFFLE]: Re-evaluating successful patterns from Era 2 to challenge current local minima."

            ctx.session.state["strategy_summary"] = brain_out[:200] + "..."
            
            # Domain Guard: Check for severe hallucinations (e.g. AWS/Terraform/FastAPI)
            hallucination_keywords = ["aws", "terraform", "fastapi", "ec2", "s3", "lambda", "docker-compose"]
            found_hallucinations = [kw for kw in hallucination_keywords if kw in brain_out.lower()]
            if found_hallucinations:
                logger.error(f"[{self.name}] DOMAIN CORRUPTION DETECTED: Brain is discussing {found_hallucinations}. Triggering Self-Healing...")
                # Force a strategic pivot and clear the bad strategy
                ctx.session.state["results_packet"] += "\n\nCRITICAL ERROR: YOUR PREVIOUS STRATEGY WAS REJECTED FOR DOMAIN CORRUPTION. FOCUS ON AI RESEARCH ONLY."
                ctx.session.state["strategy_packet"] = "Domain Corruption Detected. Reverting to base AI research strategy."
                return # Skip this iteration's surgery

            # Extract Target Node for snippet-based editing
            # Proactive Chat Filter: Strip roleplay markers and emojis before processing
            import re
            filtered_out = re.sub(r'[🚨🟢🫡🧪🏢🏢🏗️🔬📊📊🧬🌀🃏💓📘🛡️]', '', brain_out)
            filtered_out = re.sub(r'System Status Update.*|Operator Update.*|Command Center.*', '', filtered_out, flags=re.IGNORECASE)
            
            if "TARGET_NODE:" in filtered_out:
                target = filtered_out.split("TARGET_NODE:")[1].split("\n")[0].strip()
                # Strip markdown bold markers ** that LLMs often use
                target = target.strip("*").strip()
                ctx.session.state["target_node"] = target
                logger.info(f"[{self.name}] Surgery Target identified: {target}")
            else:
                ctx.session.state["target_node"] = "Transformer" # Default
            
            # Extract reference snippet from train.py
            train_py = ctx.session.state.get("train_py", "")
            if train_py and ctx.session.state.get("target_node"):
                node_name = ctx.session.state["target_node"]
                # Basic snippet extraction: find node and try to get a reasonable chunk
                # In a real app we'd use AST here too, but for prompt we just need the text
                try:
                    import ast
                    tree = ast.parse(train_py)
                    lines = train_py.splitlines()
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == node_name:
                            snippet = "\n".join(lines[node.lineno-1:node.end_lineno])
                            ctx.session.state["target_snippet"] = snippet
                            break
                    else:
                        ctx.session.state["target_snippet"] = "Node not found in source."
                except (SyntaxError, IndexError) as e:
                    logger.error(f"[{self.name}] Error extracting snippet for surgery: {e}")
                    ctx.session.state["target_snippet"] = "Error extracting snippet."
            
            # Skill discovered removed.
            pass
            
        # 2. Ask Hands for code implementation
        max_correction_attempts = 2
        for attempt in range(max_correction_attempts):
            logger.info(f"[{self.name}] Step 2: Hands (Implementation - Attempt {attempt+1})")
            
            # If this is a correction, or we have residual feedback from last iteration
            ast_err = ctx.session.state.get("ast_error")
            critic_feedback = ctx.session.state.get("critic_feedback")
            
            if ast_err:
                ctx.session.state["correction_prompt"] = f"SYNTAX ERROR: {ast_err}\nPlease fix the syntax and return the full code."
            elif critic_feedback:
                ctx.session.state["correction_prompt"] = f"PREVIOUS FEEDBACK: {critic_feedback}\nPlease ensure your NEW implementation addresses these concerns."
            else:
                ctx.session.state["correction_prompt"] = ""

            async for event in self.hands.run_async(ctx):
                yield event
            
            # AST Validation Check
            validated_code = ctx.session.state.get("validated_code")
            if not validated_code:
                logger.warning(f"[{self.name}] Hands failed AST validation. Attempting correction...")
                continue
            
            # Write proposed patch to disk for persistent audit trail/critic review
            research_dir = ctx.session.state.get("research_dir", "research_env")
            with open(os.path.join(research_dir, "docs", "proposed_patch.py"), "w") as f:
                f.write(validated_code)
                
            # 3. Ask Critic for review
            logger.info(f"[{self.name}] Step 3: Critic (Review)")
            async for event in self.critic.run_async(ctx):
                yield event
                
            critic_out = ctx.session.state.get(self.critic.output_key, "")
            if "REJECT" in critic_out:
                logger.warning(f"[{self.name}] Code REJECTED by Critic. Attempting correction...")
                ctx.session.state["critic_feedback"] = critic_out
                # Shared Memory: Critique
                docs_dir = os.path.join(research_dir, "docs") # Ensure docs_dir is defined here
                critique_path = os.path.join(docs_dir, "last_critique.md")
                with open(critique_path, "w") as f:
                    f.write(critic_out)
                ctx.session.state["validated_code"] = None
                ctx.session.state["ast_error"] = None # Clear to prioritize critic
                continue
            else:
                logger.info(f"[{self.name}] Code APPROVED by Critic.")
                
                # Fibonacci Strategic Checkpoint (Meta-Strategy)
                # iteration = ctx.session.state.get("iteration", 0) # Fallback
                # Since iteration count isn't directly in ctx.session.state in current run_async flow, 
                # we'll assume the caller (coordinator) manages the meta-state or we use a simpler flag.
                # However, for this implementation, we will assume 'iteration' is available or updated.
                
                break

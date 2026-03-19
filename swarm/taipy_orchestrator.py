from taipy import Config
import os
from swarm.drivers import ResearchResult

def mock_research_task(p, t, i):
    return ResearchResult(val_bpb=0.0, peak_vram_gb=0.0, status="placeholder", description="Task executed")

def configure_tao():
    """
    Configures the Taipy Core (Scenario Management).
    TAO = Taipy AI Orchestrator.
    """
    # 1. Data Nodes (State)
    input_program_dn = Config.configure_data_node(id="input_program")
    input_train_py_dn = Config.configure_data_node(id="input_train_py")
    iteration_dn = Config.configure_data_node(id="iteration")
    
    # The output of a research task
    research_result_dn = Config.configure_data_node(id="research_result")

    # 2. Task (The actual execution)
    # This task will wrap the ResearchAgent's execution path
    research_task = Config.configure_task(
        id="run_research_experiment",
        function=mock_research_task,
        input=[input_program_dn, input_train_py_dn, iteration_dn],
        output=research_result_dn
    )

    # 3. Scenario (The experiment)
    scenario_cfg = Config.configure_scenario(
        id="research_scenario",
        task_configs=[research_task]
    )
    
    return scenario_cfg

def init_taipy_orchestrator():
    """Initializes the Core orchestrator."""
    from taipy import Core
    Config.configure_job_executions(mode="standalone")
    core = Core()
    core.run()
    return core

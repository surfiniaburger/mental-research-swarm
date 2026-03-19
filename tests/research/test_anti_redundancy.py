import pytest
from swarm.agents import ManagerAgent
from swarm.drivers import ResearchResult

@pytest.mark.asyncio
async def test_anti_redundancy_filter_triggered():
    """BDD: Manager rejects strategy if performance stagnates."""
    # Setup - mock manager with history
    history = [
        ResearchResult(val_bpb=1.45, status="keep", description="iter 1", peak_vram_gb=0.0),
        ResearchResult(val_bpb=1.448, status="keep", description="iter 2", peak_vram_gb=0.0), # delta 0.002
        ResearchResult(val_bpb=1.447, status="keep", description="iter 3", peak_vram_gb=0.0), # delta 0.001
    ]
    
    # The filter should trigger here if we try to submit another tiny improvement
    from swarm.agents import AntiRedundancyFilter
    filter_engine = AntiRedundancyFilter(threshold=0.005, window=3)
    
    is_redundant = filter_engine.check(history)
    assert is_redundant is True

@pytest.mark.asyncio
async def test_anti_redundancy_filter_not_triggered():
    """BDD: Manager allows strategy if improvement is significant."""
    history = [
        ResearchResult(val_bpb=1.45, status="keep", description="iter 1", peak_vram_gb=0.0),
        ResearchResult(val_bpb=1.44, status="keep", description="iter 2", peak_vram_gb=0.0), # delta 0.01 (Pass!)
        ResearchResult(val_bpb=1.43, status="keep", description="iter 3", peak_vram_gb=0.0), # delta 0.01 (Pass!)
    ]
    
    from swarm.agents import AntiRedundancyFilter
    filter_engine = AntiRedundancyFilter(threshold=0.005, window=3)
    
    is_redundant = filter_engine.check(history)
    assert is_redundant is False

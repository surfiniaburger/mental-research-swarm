"""
Canon TDD Test Suite: StrategyDiversifier

Test List (from implementation_plan.md):
1. [x] test_diversifier_allows_first_category — First proposal always passes
2. [x] test_diversifier_allows_different_categories — Alternating categories pass
3. [x] test_diversifier_rejects_same_category_streak — 3+ same categories rejected
4. [x] test_diversifier_classifies_optimizer_changes — Optimizer keywords correctly categorized
5. [x] test_diversifier_classifies_architecture_changes — Architecture keywords correctly categorized
"""
import pytest
from swarm.agents import StrategyDiversifier


def test_diversifier_allows_first_category():
    """GIVEN a fresh diversifier, WHEN the first strategy is submitted,
    THEN it should always be allowed regardless of category."""
    diversifier = StrategyDiversifier(max_streak=2)
    
    result = diversifier.check("Modify configure_optimizers to use AdamW with cosine annealing")
    
    assert result.allowed is True
    assert result.category == "optimizer"


def test_diversifier_allows_different_categories():
    """GIVEN two optimizer strategies, WHEN an architecture strategy is submitted,
    THEN it should be allowed because the category changed."""
    diversifier = StrategyDiversifier(max_streak=2)
    
    diversifier.check("Use AdamW optimizer with weight_decay=0.01")
    diversifier.check("Switch to SGD with momentum and lr_scheduler")
    result = diversifier.check("Add a second transformer layer with residual connections")
    
    assert result.allowed is True
    assert result.category == "architecture"


def test_diversifier_rejects_same_category_streak():
    """GIVEN two consecutive optimizer strategies,
    WHEN a third optimizer strategy is submitted,
    THEN it should be REJECTED to force exploration."""
    diversifier = StrategyDiversifier(max_streak=2)
    
    diversifier.check("Use AdamW optimizer with weight_decay=0.01")
    diversifier.check("Switch to SGD with momentum and lr_scheduler")
    result = diversifier.check("Try cosineanneal with eta_min=0.001 and warmup")
    
    assert result.allowed is False
    assert result.category == "optimizer"
    assert "Explore a different axis" in result.reason


def test_diversifier_classifies_optimizer_changes():
    """GIVEN various optimizer-related texts,
    WHEN classified, THEN they should all map to 'optimizer'."""
    diversifier = StrategyDiversifier()
    
    assert diversifier.classify("configure_optimizers with AdamW") == "optimizer"
    assert diversifier.classify("learning_rate warmup with cosineanneal") == "optimizer"
    assert diversifier.classify("ReduceLROnPlateau with SGD and momentum") == "optimizer"


def test_diversifier_classifies_architecture_changes():
    """GIVEN various architecture-related texts,
    WHEN classified, THEN they should all map to 'architecture'."""
    diversifier = StrategyDiversifier()
    
    assert diversifier.classify("Add more transformer layers with layer_norm") == "architecture"
    assert diversifier.classify("Increase num_heads and hidden_dim in the MLP") == "architecture"
    assert diversifier.classify("Implement mixture of experts with gating network") == "architecture"

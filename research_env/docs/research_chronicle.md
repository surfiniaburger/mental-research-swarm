# Research Chronicle

## Era 1: Initial Baseline
Setting up the environment and establishing dense performance metrics.

## Era 2: Strategic Shift
I am evaluating the current state of `MoE_FFNEfficient_V2`. While the logic is correct and handles routing (Top-1) accurately, there are critical performance bottlenecks identified in the "efficiency_note" rationale from the previous acceptance:

1.  **Python Loop Bottleneck:** The explicit Python loop (`for i in range(num_experts)`) to stack outputs prevents vectorization across the expert dimension for large `num_experts`.
2.  **Memory Footprint:** Constructing a temporary `[num_batches_flat, 
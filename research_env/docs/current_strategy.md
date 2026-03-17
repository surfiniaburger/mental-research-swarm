# Current Strategy

: [B, S, K]

# Layer forward pass
output = layer(inputs, indices) # Efficient gather occurs internally or manually
```

### Notes on Flattening vs `[B, S, D]`

The text mentions "Input shape ... `[batch, seq_len, hidden_size]`" (3D) but also usage examples with flattened shapes.
- **Implementation Choice**: This code prioritizes the 3D shape `[B, S, D]` as it is standard for Transformers.
- **Efficiency Note**: If performance profiling shows the `gather` operation is bottlenecked, experts can be stored in a flattened weight tensor (e.g., `[E*D_expert, D_in]`) and gathered using indices reshaped to `[B*S, E*D_expert]` before gathering. This avoids loop overhead but requires careful reshape handling.

This consolidated class structure allows for flexible expert routing while respecting the efficiency constraints of modern MoE architectures.

Let me know if you need to integrate this with a specific `MoEConfig` class or add auxiliary loss computation logic explicitly in the forward pass.

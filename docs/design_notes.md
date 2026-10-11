# Design Notes

- **Determinism everywhere.** Every "random" value is `stable_*` -- a pure
  function of a key -- so benchmarks and tests reproduce across machines.
- **Lazy expert weights.** The MoE materializes an expert's weights only when
  routed, so `n_experts=24576` costs nothing until a token actually touches it.
- **Monotone training signal.** The mock trainer only ever pushes `skill` up,
  avoiding the "reward -1 / +1 collapses skill to zero" trap.
- **Small dims, real economics.** Reference math uses dim=32; the memory/tier/
  quant math uses the true 125B/24,576 numbers, so the *conclusions* scale.

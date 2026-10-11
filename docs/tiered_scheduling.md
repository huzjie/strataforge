# Tiered Expert Scheduling

The core idea: a 24,576-expert MoE never needs all experts resident at once,
because each token routes to only ~10. So place the **hot** experts where
access is fast (GPU) and compute the **cold tail** where memory is cheap (CPU
+ RAM).

## Placement policy

1. Every routed expert id bumps a frequency counter.
2. If the expert is already in the GPU tier (LRU cache), serve it there.
3. Otherwise, a deterministic promotion decision (biased by `skill` in the
   mock) decides whether it enters the hot tier, evicting the least-recently-
   used resident when full.

## Why LRU works here

Expert popularity in a frontier MoE is extremely skewed: a small fraction of
experts handle most tokens (shared grammar, common idioms, frequent tokens).
LRU keeps exactly those hot, which is why ~4,000 of 24,576 experts (16%) can
cover most routing events.

## Measurable outcome

The `tier_hitrate` benchmark reports the fraction of routed experts served
from the GPU tier. In the trainable mock this climbs 0.75 -> 1.00 as `skill`
rises, which is the whole economic argument: better placement = less CPU
fallback = higher throughput at the same memory budget.

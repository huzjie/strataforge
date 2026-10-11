# Prompt: Tiered Scheduling Explainer

Explain to a systems engineer how hot/cold expert tiering lets a 125B MoE run
on a 12GB GPU. Cover: frequency-skewed routing, LRU promotion/eviction, and why
~16% resident experts cover most tokens. Give the placement pseudocode.

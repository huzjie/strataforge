# Speculative Decoding (MTP)

Generation is normally sequential: one token per forward pass. Speculative
decoding breaks that with a **guess-then-verify** loop:

1. A tiny MTP draft head guesses `draft_len` next tokens (cheap).
2. The main model verifies the guesses in **one** forward pass.
3. Correct guesses are accepted; the pass stops at the first mismatch.
4. Output stays identical to greedy decoding -- the main model decides every
   token -- but each pass now emits more than one token.

## Speedup math

With acceptance rate `alpha` and draft length `gamma`:

    tokens_per_pass = 1 + alpha * (gamma - 1)
    speedup         = (1 + alpha * (gamma - 1)) / (1 + c * gamma)

where `c` is the per-guess draft cost (tiny). At `alpha = 0.85`, `gamma = 2`,
`c = 0.05`, speedup ~= 1.68x -- matching Strata's reported 1.6-1.8x.

## Why quality is unaffected

The verifier is the *authoritative* model. Draft tokens are only accepted when
they match the ground-truth continuation; on mismatch the authoritative token
is emitted. So speculative decoding changes *how fast*, never *what it says*.

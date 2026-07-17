# pivot_2d — ρ(F0,Fhist) × peakedness(F0) conditioning

- Δ = moment − j0-only target rank; negative = moment helped. n=1645 val turns.

| | ρ low (disagree) | ρ mid | ρ high (redundant) |
|---|---|---|---|
| **F0 flat (anaphora?)** | Δ-4.7 · hurt 10% · help 58% · n=218 | Δ-3.3 · hurt 15% · help 45% · n=185 | Δ-2.8 · hurt 15% · help 47% · n=145 |
| **F0 mid** | Δ-4.1 · hurt 15% · help 54% · n=170 | Δ-4.0 · hurt 11% · help 53% · n=177 | Δ-2.0 · hurt 11% · help 36% · n=202 |
| **F0 peaked (pivot?)** | Δ-3.9 · hurt 10% · help 47% · n=160 | Δ-2.4 · hurt 14% · help 35% · n=187 | Δ-1.7 · hurt 8% · help 31% · n=201 |

- named MOMENT-HURT cases: ad249ee4 → ρ-tercile 2, peak-tercile 1 (Δ+16); 9ec0b4e8 → ρ-tercile 3, peak-tercile 1 (Δ+11); 124cf35a → ρ-tercile 2, peak-tercile 2 (Δ+18); c2244e8e → ρ-tercile 2, peak-tercile 2 (Δ+1)
- validation rule: PIVOT cell (peaked × ρ-low) must show hurt >> help relative to ANAPHORA cell (flat × ρ-low); otherwise the 2D gate is dead too.

"""eval.encoder_eval — Multi-version encoder quality eval infrastructure.

Composes existing eval/longmem/ pipeline pieces (replay, fresh_brain,
answerer, judge) with version-pinning (set_interaction_active) and
multi-dimensional encoding-quality probes that measure WHAT the encoder
wrote, not just whether the answer is right.

See README.md for architecture, runner.py for CLI entry.
"""

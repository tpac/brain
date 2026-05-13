"""Agent introspection — interview the agent about its prompt and actions.

Family of probes that audit prompt+behavior pairs by interviewing a fresh
stateless agent (Sonnet by default) on:

  aspect_probe        — pre-run, prompt only. "How does fresh agent READ
                        this prompt across N lenses?" (existing —
                        eval/encoder_prompt_probe.py)
  compliance_probe    — post-run, prompt + actions. "Why did the agent
                        comply / skip on each named rule?"
  coherence_probe     — pre-run, prompt only. "Where does the prompt
                        contradict itself? Which rule wins on overlap?"
  coverage_probe      — pre-run, prompt + conversation. "Given THIS
                        conversation, what would you encode and why?
                        What would you skip?"
  edge_case_probe     — pre-run, prompt + scenario. "How would you
                        handle scenario X (a corner case)?"
  counterfactual_probe — post-run, prompt + actions + proposed change.
                        "You skipped Y in this output. If the prompt
                        said Z, would you have included it?"

Each probe is a separate file in this package, sharing helpers from
_common.py. All produce structured JSON + a markdown report.
"""

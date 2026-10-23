"""Batchers for Ouro-1.4B decode. Each module exposes the same entry point:

    run(prompts, max_new_tokens, max_slots, exit_threshold, dtype) -> RunResult

- reference: every sequence alone (B=1), HF cache. The ground truth.
- baseline:  depth-synchronous decode steps; exited rows wait for the slowest.
- cdb:       Continuous Depth Batching; an exited row's slot is reused next block call.
"""

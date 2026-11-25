# Week 6b notebook: is KV sharing across loops free when trained in? (append-only)

Started 2026-11-24 by the session taking over Week 6b. Plan, review and the approved rule: CALIBRATION.md,
"Week 6b plan" (rule v2, APPROVED by Sid 2026-11-23, committed 0db9195 before any data). Background: HANDOFF_W6B.md.

# Frame
- Question: does an 81M looped LM (6 layers x 4 loops) trained from the start with one shared KV cache
  (loops 2-4 reuse loop 1's K/V) reach the same held-out loss as one cache per loop, within the noise margin?
- Metric: final held-out loss on the first 400 x 32 non-overlapping 1024-token windows of val.bin (13.1M tokens),
  last loop only. Lower is better. Two-seed variants use the seed mean.
- Margin, pass/fail and flags: exactly as the approved rule (m = max(0.01, 2 x seed gap); inconclusive if m > 0.03;
  match if L <= L(perloop) + m; beats mqa3 if L < L(mqa3) - m; not transferable if L(noloop) - L(perloop) <= m).
- Guardrails: finite loss throughout; loss curves without spikes or collapse; the GPU is an H100 80GB HBM3
  (env.txt, and the W6B_GPU check in every w6b task); runs see the same data order.
- Budget: Sid approves each launch with a cost estimate (H100 ~$4/h).
- What this cannot show: anything about 1.4B models or 20B+ token budgets directly; downstream task accuracy.

# State before any GPU spend (2026-11-24)
- CPU tests: `python3 -m pytest -q tests/test_train_looped.py`: 11 passed.
- Data prep already done (run w6b_prep-l40s-20261121-153746, exit 0): train.bin 4.2 GB = 2.1B tokens,
  val.bin 40 MB = 20M tokens (19,310 documents, max id 50256, decodes to clean text), FineWeb-Edu
  revision 87f09149ef4734204d70ed1d046ddc9ca3f2b8f9, val from shard 013 (train from 000-012). Not re-run.
- Added the H100 guard to the w6b tasks (ee9a78c).

# Predictions (written before the smoke run)
- Paper estimate: ~1.37 GFLOP per training token for the 4-loop variants (6 x 208.6M matmul params + attention),
  ~0.52 GFLOP for noloop. At 30-45 % of the H100's 989 TFLOP/s bf16: 1.7-2.6 h per 2B-token looped run.
- Steady step time (524k tokens) for perloop: 1.6-2.4 s (= 7.2e14 FLOP / 300-450 TFLOP/s). sharefirst within 5 %
  of perloop. mqa3 a little faster. sharefirst_w64 1.3-1.8x perloop (masked attention over 2T keys, no flash
  kernel). noloop ~0.4x perloop.
- If perloop is slower than 3 s/step, utilization is < 25 % and the plan's cost estimate is wrong; look at why
  before the probe.

# Experiments
## E1 - smoke test (all six variants, 10 steps each)
Command: `./run_modal.sh --task w6b_smoke --gpu H100`, run w6b_smoke-h100-20261124-121303, code 7ba8a36's parent
(ee9a78c; the probe-offset change does not touch the smoke path). GPU: NVIDIA H100 80GB HBM3, torch 2.5.1+cu124.
Logs: results/modal/w6b_smoke-h100-20261124-121303/ (checkpoints not kept).
Result (steady = steps 2-10, n=9 per variant; step 1 includes torch.compile):

| variant | step 1 (s) | steady s/step (min-max) | tokens/s | vs perloop | 2B-token run (h) | loss @10 |
|---|---|---|---|---|---|---|
| perloop | 51.2 | 2.678 (2.669-2.690) | 196k | 1.00 | 2.84 | 9.532 |
| sharefirst | 30.1 | 2.550 (2.545-2.559) | 206k | 0.95 | 2.70 | 9.538 |
| sharefirst_w64 | 41.3 | 4.662 (4.654-4.676) | 112k | 1.74 | 4.94 | 9.536 |
| pairs | 30.6 | 2.575 (2.568-2.579) | 204k | 0.96 | 2.73 | 9.533 |
| mqa3 | 43.8 | 2.601 (2.592-2.608) | 202k | 0.97 | 2.76 | 9.547 |
| noloop | 13.3 | 1.362 (1.360-1.365) | 385k | 0.51 | 1.44 | 9.651 |

Guardrails: finite losses, no NaN/warning/error lines in log.txt, right GPU, params as designed (mqa3 75.8M, rest 81.1M).
Verdict vs prediction: perloop 2.68 s/step is above the predicted 1.6-2.4 s (about 27 % of bf16 peak, not 30-45 %)
but under the 3 s stop line. sharefirst -5 %, mqa3 -3 %: as predicted. w64 1.74x: top of the 1.3-1.8x range.
noloop 0.51x vs ~0.4x predicted: the embedding/LM head (38.6M params, every variant pays it) is a larger share
than I allowed for. Main-run cost estimate rises from ~2 h/run to 2.7-2.8 h (4.9 h for w64).

## E2 - lr probe (launched 2026-11-24 12:31, before any of its data was read)
Command: `./run_modal.sh --task w6b_lrprobe --gpu H100` (detached), run w6b_lrprobe-h100-20261124-123103,
code 7ba8a36 (tag "-dirty" = untracked notebook and smoke logs only). GPU: NVIDIA H100 80GB HBM3 (env.txt).
Change: lr in {5e-4, 1e-3, 2e-3} for perloop and sharefirst, 200M tokens each (381 steps, warmup 40), scored on
val windows 12801+ (100 x 32 windows), disjoint from the main runs' final eval (pre-registered, 7ba8a36).
Decision (approved rule): main-run lr = perloop's lowest final probe loss; sharefirst's best lr is reported.
Expected cost: 6 x 381 steps at 2.55-2.68 s + compiles = ~1.8 h, ~$7.
Prediction: perloop's best lr is 1e-3 or 2e-3 (5e-4 is too low for 381 steps at this size); the best two lrs are
within 0.05 nats; sharefirst prefers the same lr as perloop. At equal lr, sharefirst is within 0.03 nats of perloop
at 200M tokens. If sharefirst's best lr differs, that is the first sign the shared cache trains differently.
Result (final loss on val windows 12801+, 100 x 32 x 1024 = 3.3M tokens; n=1 seed per cell; 0.28-0.31 h per run,
task 12:31-14:19 = 1.8 h, ~$7):

| lr | perloop | sharefirst |
|---|---|---|
| 5e-4 | **4.6995** | **4.6097** |
| 1e-3 | 7.1306 | 5.8263 |
| 2e-3 | 7.6682 | 5.3197 |

Curves (val at steps 100/200/300/381): perloop 5e-4 6.04/5.16/4.82/4.72; perloop 1e-3 7.41/7.27/7.16/7.14;
perloop 2e-3 7.68/7.67/7.67/7.66; sharefirst 5e-4 5.96/5.07/4.73/4.63; sharefirst 1e-3 7.40/6.72/6.04/5.84;
sharefirst 2e-3 6.92/6.07/5.53/5.34. No NaN, no loss spikes; grad norms 0.08-2.7.
Guardrails: H100 80GB HBM3 (env.txt), all 8 commands exit 0, evals disjoint from the final-eval tokens.
Decision (approved rule, applied mechanically): perloop's lowest final probe loss is at lr 5e-4, so every main run uses
lr 5e-4 (W6B_LR in modal_app.py). sharefirst's best is also 5e-4: no disagreement to report.
Verdict vs prediction: REFUTED on three counts. (1) The best lr is 5e-4, not 1e-3/2e-3. (2) The lrs are not within
0.05 nats: at 1e-3 and 2e-3 perloop stalls near 7.1-7.7 nats, about the loss of predicting token frequencies without
context, and does not leave it in 381 steps. (3) At 5e-4 sharefirst is 0.09 nats BELOW perloop, not within 0.03.
Why / what I learned:
- High lr + this probe's 40-step warmup puts the 24-layer-deep looped model on a plateau it cannot leave. The main
  runs warm up for 200 steps, so 1e-3 might have trained there; the probe cannot say. 5e-4 is the safe side either way.
- The shared cache trains more stably: sharefirst escapes the plateau at 1e-3 and 2e-3, perloop does not. One seed,
  but the gap is large (1.3-2.3 nats).
- 5e-4 is the lowest lr tested, so the best lr could be lower; for 2B tokens I expect it is not much lower. Caveat.
- The 0.09-nat sharefirst lead is one seed at 10 % of the budget; it is not evidence for the main question.
Follow-up: watch the main runs' early evals for the same plateau (perloop at 5e-4 was at 6.04 by step 100 here).

## E3 - second lr probe point, 2.5e-4 (pre-registered a698ff0; launched before any of its data)
Command: `./run_modal.sh --task w6b_lrprobe2 --gpu H100`; same settings and held-out scoring as E2.
Decision: main-run lr = perloop's lowest final probe loss over {2.5e-4, 5e-4, 1e-3, 2e-3}.
Expected cost: 2 x ~0.3 h = ~0.65 h, ~$2.6.
Prediction: perloop at 2.5e-4 ends worse than at 5e-4 (4.70), around 4.85-5.0: with only 381 steps the lower lr
learns more slowly, and E2's 5e-4 curve showed no sign of instability. So the main lr stays 5e-4. Same for sharefirst.
If 2.5e-4 wins, the probe's short horizon is favouring low lr less than I think, and 2.5e-4 is again an edge.
Result (run w6b_lrprobe2-h100-20261124-145136, H100 80GB HBM3, all 4 commands exit 0, 0.31 + 0.29 h, ~$2.5):
perloop 2.5e-4: final 4.9408 (val 5.99/5.30/5.04/4.96 at steps 100/200/300/381);
sharefirst 2.5e-4: final 4.9298 (5.98/5.28/5.02/4.95). No spikes, grad norms 0.46-1.01.
Full grid, final probe loss: perloop 4.9408 / 4.6995 / 7.1306 / 7.6682 and sharefirst 4.9298 / 4.6097 / 5.8263 /
5.3197 at lr 2.5e-4 / 5e-4 / 1e-3 / 2e-3.
Decision (rule): perloop's best is 5e-4, now an interior point of the grid; main lr stays 5e-4. sharefirst agrees.
Verdict vs prediction: SUPPORTED (2.5e-4 worse than 5e-4, 4.94 predicted 4.85-5.0, same for sharefirst).
Note: at 2.5e-4 the two variants are 0.011 apart, at 5e-4 0.090. With one seed, the 5e-4 lead is not a finding.

## E4 - main runs (launched 2026-11-24 15:22-15:24, lr 5e-4, no further approval needed per Sid)
Runs (one Modal app each, all H100 80GB HBM3 per logs; meta lines confirm lr 5e-4, 3814 steps, warmup 200):
w6b_perloop-h100-20261124-152246, w6b_perloop_s1-h100-20261124-152259, w6b_sharefirst-h100-20261124-152312,
w6b_sharefirst_s1-h100-20261124-152325, w6b_sharefirst_w64-h100-20261124-152338, w6b_mqa3-h100-20261124-152351,
w6b_noloop-h100-20261124-152404. Expected ~21 GPU-h, ~$85. Spending checks run by `watch_w6b.py -w 120 --guard`.
Predictions (before any main-run data): seed gap d < 0.01 nats, so m = 0.01-0.02 and rule 0 does not trigger;
noloop worse than perloop by > 0.1 nats (looping helps); sharefirst within m of perloop (rule 1 pass), as in MoR
(+0.006-0.016 nats) and Monea; sharefirst_w64 at or below sharefirst; mqa3 within ~0.03 of perloop, so rule 2
(sharefirst beats mqa3 by > m) is uncertain, ~40 %.
Ops note (17:20): the guard (`watch_w6b.py -w 120 --guard`, on the laptop) did not refresh from 16:43 to 17:20: the
Mac slept and time.sleep does not count sleep. Training on Modal was unaffected. A one-off check at 17:20 over every
run's full eval history found no spending-check violation. Fix: `caffeinate -dims -w <guard pid>` keeps the Mac awake
while the guard runs.
noloop finished: final 3.4420 (1.48 h, H100 80GB HBM3), evals 4.96 (250) -> 3.37 (3814), smooth.

## E4 interruption (found 18:25)
All six unfinished main runs stopped together at ~17:14: last log lines 17:05-17:12, no error in the logs, no
containers left, Modal call status FAILURE with no result. Cause: the Modal workspace hit its spend limit
("Workspace ... has exceeded its spend limit" on relaunch at ~18:30). Nothing has billed since.
State kept on the volume (/vol/w6b/<variant>/ckpt.pt, complete 972,940,918-byte files, mqa3 909,239,926):
perloop and perloop_s1 last ckpt 17:05-17:07 (step 2250), sharefirst and sharefirst_s1 17:11-17:13 (step 2500),
mqa3 17:05 (step 2250), sharefirst_w64 17:02 (step ~1250-1500). noloop had finished (16:54).
I stopped the six dead apps (so a late Modal retry can't race a resumed run on the same checkpoint dir).
Resume plan: relaunch the same six tasks; train_looped.py resumes from ckpt.pt and draws batches from (seed, step),
so a resumed run sees the same data as an uninterrupted one. log.jsonl will hold duplicate records for the steps
redone after the last checkpoint; analysis keeps the last record per step. Remaining cost ~8.5 GPU-h, ~$35.

# Landing recovery — 2026-11-11

## Status

**First trial completed; all four candidates rejected for lost rotation.** Keep
`export-1500/model_1500.pt`; no deployed policy has been replaced. Actual GPU completion
and paired comparisons are recorded in `out/frontflip-landing-repair/state.json` and
`out/frontflip-landing-repair/evaluation/comparison.json`.

This workspace started before the frontflip task existed. Its task implementation and
qualified reference were recovered from the existing `feat/frontflip` checkout at
`C:/Users/clayt/Documents/Jumper Crab`, including the November 10–11 working-tree recipe.
The other checkout and its uncommitted source work were not modified. The current task
has since advanced to the separately named [rotation-guard trial](ROTATION_GUARD.md).
Reproduce this first trial's reward numbers with its archived source package, not a
later working-tree recipe.

Baseline SHA-256:
`cea9a567a1bc577f031a94ba526e77882a3169ed3f5bd08b20d699072b32e1de`.
This is the October 25 Colab `model_1500`, trained with `warp:cuda:0`, 256 environments,
not an identically numbered checkpoint from a different run.

## Why the November 10 trial is not a starting checkpoint

The corrected November 11 review rejected every fine-tuned milestone. Contact costs
were zeroed on terminating steps: a 99 N strike cost 102.4 on that control step, while a
101 N strike paid only the flat 50 termination penalty. Later milestones learned harder
non-pad strikes instead of softer six-limb landings.

The old probe also reset all environments when one ended, mixed episode windows, and
called an integral of independent step maxima an impulse. A reported force reduction
from that probe was not evidence of an improved landing.

## This separately named trial

- Charge contact and foot-load costs on terminating steps, without changing collision limits.
- Reduce the squared non-pad impact weight from -5 to -0.5 so the existing landing is
  not more expensive than abandoning the attempt.
- Replace the absolute 30 N per-foot knee with a bounded, scale-invariant sharing cost
  using simultaneous upward support reactions during 1.0–1.6 s, after at least 270° turn.
- Give continuous landing credit proportional to the square of the supported-foot
  fraction. Keep the strict orientation and rotation factors and all rotation tracking terms.
- Sum both claw-pad force vectors before computing a limb's peak. Integrate vertical
  force at the 400 Hz physics timestep for impulse; never integrate 200 Hz peak samples.

The sharing cost is at most `10 * (5/6) * 0.6 = 5` per episode. A one-off synthetic
control verified equal-load cost 0, single-limb cost 5/6, scale invariance, both-pad
summation, and identical impact pricing for terminated/nonterminated environments.
At the new weight, the 99 N step costs 10.24 and the 101 N terminating step costs 60.89.

## CPU evidence before submitting GPU training

Windows AMD CPU, `native:cpu`, 32 environments, seed 1234, 700 control steps, training
observation noise **on**, with play-mode startup friction/COM/encoder-bias draws:

| Quantity | Baseline measurement |
|---|---:|
| Complete 0.5 s first-episode landing windows | 32/32 |
| Physical failures | 0/32 |
| Forward rotation at first touchdown, median / minimum | 343.2° / 289.1° |
| Non-pad impact, median / p90 | 75.3 / 92.8 N |
| Limb peak medians, LF RF LM RM LR RR | 33.7, 35.3, 36.4, 39.3, 20.3, 21.0 N |
| Simultaneous vector-sum support peak, median | 106.9 N |
| Vertical-impulse share coefficient of variation, median | 0.396 |
| Discounted continuation versus immediate -50 termination, median / minimum | +49.8 / +29.8 |

The rotation and non-pad force reproduce the historical 32-start baseline. The total
support peak and impulse use corrected definitions and must not be compared directly
with the old sum-of-independent-maxima results. The discounted comparison uses the
actual reward sequence and gamma 0.99; it prices the existing landing, not an improved
policy or every possible crash trajectory.

A second baseline control batch used the final qualification definition below. Nominal
physics qualified 32/32 at both seeds (1234 and 2718). Randomized physics qualified
15/32 at seed 1234, with no physical failures, but only 8/32 at seed 2718, with 9 physical
failures. Randomized seed 2718 had an 87.1 N median non-pad peak. These are existing
baseline weaknesses, not results from the new GPU policy; one seed is not adequate.

The stock measured baseline replay also completed: `native:cpu`, one environment,
700 samples through 3.495 s, training observation noise on, peak absolute actuator
torque 1.7464 N·m. Its CSV and plot are under
`out/frontflip-landing-repair/kaggle-inputs/measure/`.

Two CPU policy updates completed. A second smoke test of the actual Kaggle driver
verified actor, critic, optimizer, iteration 1500, environment step counter 36024, and
learning rate 1e-5 were restored exactly. No CPU-smoke checkpoint is an accepted policy.

## GPU experiment and acceptance

Private Kaggle kernel:
`claytoncrumbly/jumper-frontflip-landing-recovery-trial-20261111`.
The Kaggle account is distinct from the requested GitHub identity, `Claytonjc-git`;
no GitHub push is needed or performed.

Two training seeds, 1234 and 2718; 150 updates per seed; 256 environments; requested
Tesla T4; checkpoints at 0, 50, and 150 updates. The launcher must verify actual CUDA
physics, imported source paths, all 540 source-file hashes, and exact checkpoint
restoration before training. Actual GPU details belong to its downloaded
`gpu-verification.json`, not the accelerator request.

The first submission was rejected because its kernel slug collided with the input
dataset slug. A distinct `-trial-` kernel slug resolved that conflict; version 1 was
accepted and completed. Downloaded verification confirms an actual Tesla T4 and a CUDA
physics step. Both seeds completed 150 updates; both milestone-zero actor/critic/optimizer
states match the source checkpoint exactly.

Paired CPU screening uses 8 starts per checkpoint in nominal and randomized physics,
with observation noise on. Candidates preserving successful flips proceed to 32-start
checks under both modes and both evaluation seeds. Acceptance requires:

1. No physical failures and no truncated landing windows.
2. At least 90% qualified attempts, with qualification no worse than the paired baseline.
3. At least 340° and no more than 380° accumulated turn before/at first touchdown,
   upright touchdown, six supports reached within 100 ms, and a final upright six-foot
   stance with turn still within 340–380°. Six supports within 100 ms is explicitly
   distinct from six simultaneous contacts on the very first touchdown sample.
4. No higher median or p90 peak on any limb, and a lower worst-limb median.
5. Lower median non-pad impact without a worse p90.

Only complete, nonfailed landing windows contribute to landing-force summaries;
failed attempts remain explicit failures, never zero-force successes. This is simulation
qualification, not hardware validation. The 50 N reference limb-force qualification
limit is not the same as the unchanged 100 N training termination threshold.

## First-trial result: lower forces, insufficient rotation

Eight paired CPU starts per mode, observation noise on, evaluation seed 1234. All four
candidates failed nominal rotation/landing qualification (0/8 against baseline 8/8), so
none advanced to larger holdouts or replaced the policy.

| Training seed / updates | Nominal non-pad median | Nominal turn median | Randomized failures |
|---|---:|---:|---:|
| 1234 / 50 | 52.6 N | 334.9° | 2/8 |
| 1234 / 150 | 53.3 N | 286.9° | 7/8 |
| 2718 / 50 | 60.6 N | 333.0° | 3/8 |
| 2718 / 150 | 55.3 N | 332.4° | 3/8 |

The 1234/50 candidate also completed a stock 700-step measured replay at
`out/frontflip-landing-repair/gpu-output/seed-1234/measure/`. Its softer impact is **not**
a successful frontflip improvement. The old landing reward evaluates current turn,
allowing rotation regained on the ground to earn later credit. A separately named
follow-up must constrain turn at the *first* touchdown, not just after recovery.

## Validation limits

`ruff check .` passes. The focused registry/parity/replay/controls suite passes all 137
checks with Windows UTF-8 enabled. The full suite reports 1198 passed, 17 failed,
27 skipped, and one deselected self-SIGTERM test. All 17 failures reproduce in the
untouched historical checkout: Windows ABI/symlink/socket behavior, newline-sensitive
hashes, Rust version detection, and existing translation hashes. On Windows the
self-SIGTERM test terminates pytest outright and cannot yield an ordinary test result.

Python/simulation setup gates A–D pass. Rust/controller gates E–F fail because the
controller toolchain is absent. The operator explicitly approved **training-only**
validation with those gates unverified; no controller build, deployment, or unrelated
Windows fixes are included.

## Local artifacts

`out/frontflip-landing-repair/` contains the source/input manifest, launchers, CPU probes,
raw per-attempt traces, test logs, and private Kaggle state. The one-off probes were
written outside the repository and copied there for reproducibility; no task-specific
regression tests were added to `tests/`.

After training finishes, download its output and run the prepared comparison:

```powershell
kaggle kernels output claytoncrumbly/jumper-frontflip-landing-recovery-trial-20261111 -p out/frontflip-landing-repair/gpu-output
.venv\Scripts\python.exe out/frontflip-landing-repair/evaluate_candidates.py --download out/frontflip-landing-repair/gpu-output --baseline out/frontflip-landing-repair/kaggle-inputs/model_1500.pt --out out/frontflip-landing-repair/evaluation
```

A stock `scripts/play.py --headless --measure` replay must accompany any candidate report.
Do not infer improvement from training reward or from a kernel that merely completed.

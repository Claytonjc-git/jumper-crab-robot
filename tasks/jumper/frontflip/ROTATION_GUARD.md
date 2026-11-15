# First-touchdown rotation guard — 2026-11-11

**Status: training and paired validation completed; no replacement policy accepted.**
The [first landing-recovery trial](LANDING_RECOVERY.md) reduced some impacts but lost
rotation on every candidate. Its source package, checkpoints, comparisons and rejection
remain intact under `out/frontflip-landing-repair/`.

## What changed after that measured failure

`FrontflipCommand` now latches accumulated forward rotation at the **first touchdown
after real flight**. The task explicitly chooses 15 ms continuously airborne and at
least 90° accumulated turn to distinguish flight from startup contact jitter.
Repeated reward/termination/command readers cannot advance that state twice per step;
partial resets clear only the reset environments.

The landing reward requires that latched turn to be within **340–380°**. Rotating farther
on the ground cannot recover that eligibility. A training-only termination rejects
first touchdown outside the same range, paying the existing flat failure cost in
addition to unmasked contact costs. The previous trial's impact/share weights,
reference, robot mechanics, collision limits, observations and rotation-tracking terms
are unchanged.

**Replay deliberately omits that new termination.** Evaluation records the full failed
landing rather than making a premature reset appear to reduce force. The independent
probe still applies strict first-touchdown/final-stance qualification and reports all
physical failures.

## CPU controls before GPU submission

Windows AMD CPU, `native:cpu`, training observation noise on, seed 1234:

| Control | Measured result |
|---|---|
| Source policy, nominal, guard enabled | 8/8 qualified; 0 guard terminations |
| Rejected first-trial seed 1234/+50, nominal, guard enabled | 8/8 rejected at first under-rotated touchdown |
| Source policy, randomized, guard enabled | 15/32 qualified; 15 first-touchdown rotation terminations |
| Synthetic under-rotated touchdown, later recovered to 360° | Landing credit remains 0 |
| Synthetic properly rotated touchdown, later at 360° | Landing credit remains 1 |

The synthetic controls also verify startup is not touchdown and repeated readers are
idempotent. The rollout probe verifies partial resets clear latched rotation only for
the reset environments. Two CPU policy updates through the actual trial driver passed
exact actor/critic/optimizer/iteration/environment-counter/learning-rate restoration.
The focused registry/parity/replay/controls suite passes all 137 tests; lint passes.

## Bounded GPU run

Private kernel: `claytoncrumbly/jumper-frontflip-rotation-guard-20261111`.
Private dataset: `claytoncrumbly/jumper-frontflip-rotation-guard-inputs-20261111`.
GitHub identity remains `Claytonjc-git`; no push or deployment is performed.

The unchanged `cea9a567…` October 25 Colab source checkpoint is used for **both** seeds
1234 and 2718, not a rejected first-trial checkpoint. Each seed trains 50 updates at
256 environments; checkpoints are saved at 0, 10, 25, and 50 updates. The source archive
contains 540 individually hashed files. Downloaded verification confirms an actual
Tesla T4 and a successful CUDA physics step. Both milestone-zero actor/critic/optimizer
states match the source exactly. Submission through training completion took 8.2 minutes,
including remote setup. Training mean reward changed -28.10 → -20.27 (seed 1234) and
-29.51 → -18.43 (seed 2718); these are the full 50-update runs, not evidence for accepting
the earlier checkpoint.

All six candidates first receive paired eight-start nominal and randomized CPU
screens. Candidates must preserve successful-flip counts without physical failures.
At most one candidate per training seed is selected by the lowest summed median
non-pad force across the two screen modes; that shortlist is frozen **before** larger
holdouts. Up to two finalists receive the same four 32-start holdout batches as the
baseline: two evaluation seeds × nominal/randomized physics, noise on.

The [first trial's acceptance criteria](LANDING_RECOVERY.md#gpu-experiment-and-acceptance)
remain unchanged: at least 90% proper rotation/landing qualification, no physical
failures or incomplete windows, no higher limb median/p90, lower worst-limb median,
and lower non-pad median without worse p90. Checkpoint hashes **and task-recipe hashes**
are checked before reusing any local measurement.

## Measured finalist: nominal improvement, not an accepted replacement

The frozen shortlist contained only `seed-1234-after-10`. The other five checkpoints
lost nominal qualification, introduced a randomized physical failure, or lost a
successful-flip count in the paired eight-start screen.

Experimental checkpoint:
`out/frontflip-rotation-guard/gpu-output/seed-1234/model_after_10.pt`.
It was trained with `warp:cuda:0`, 256 environments, 10 additional PPO updates on a
Tesla T4. Its stored iteration is 1509 and environment counter 36264; the source had
iteration 1500 and counter 36024. Filenames record completed updates, not the runner's
zero-based iteration label.

Four fresh CPU holdouts, 32 starts each, observation noise on, with the training-only
rotation termination **disabled**:

| Evaluation seed / physics | Qualified, source → candidate | Candidate physical failures | Non-pad median, source → candidate | Candidate touchdown turn median |
|---|---:|---:|---:|---:|
| 1234 / nominal | 32 → 32 | 0 | 69.0 → 63.0 N | 348.8° |
| 1234 / randomized | 15 → 16 | 0 | 75.3 → 63.3 N | 342.5° |
| 2718 / nominal | 32 → 32 | 0 | 68.3 → 63.2 N | 349.0° |
| 2718 / randomized | 8 → 11 | 7 | 87.1 → 73.4 N | 348.1° |

Nominal non-pad impact fell 8.7% and 7.4%, with all 64 nominal starts still qualifying.
However, this shifted load onto some support pads. At nominal seed 1234, LF median
rose 42.6 → 45.0 N and RM rose 41.0 → 44.5 N; the worst-limb median did not decrease.
At randomized seed 1234, LM rose 36.4 → 42.9 N and RM rose 39.3 → 42.6 N. Several limb
p90 values also rose. Pad summaries use only complete, nonfailed windows; the seven
failed randomized attempts are separately retained as failures, never soft landings.

Randomized qualification is still only 27/64, with seven physical failures, against
23/64 and nine failures for the source. Those small count differences are exploratory,
not a demonstrated robust success rate. The candidate fails the 90% qualification,
no-failure and per-limb load gates. **No checkpoint was exported, deployed, or substituted
for the original.** Reducing housing strikes is not the same as reducing limb landing loads.

The finalist also completed stock `scripts/play.py --headless --measure`: `native:cpu`,
one environment, training observation noise on, 700 samples through 3.495 s, peak
absolute actuator torque 1.7464 N·m. CSV and plot:
`out/frontflip-rotation-guard/gpu-output/seed-1234/measure/`. Per-support force evidence
is in the independent first-episode `evaluation/*/traces.npz` and `summary.json` files;
the stock joint CSV alone does not measure ground-reaction forces.

The complete Python suite was repeated after the guard changes: 1198 passed, the same
17 pre-existing Windows failures, 27 skipped, and the self-SIGTERM test deselected.
There are **no new suite failures** relative to the untouched historical checkout.
Controller/Rust gates remain explicitly unverified; no hardware claim is made.

## Artifacts and continuation

`out/frontflip-rotation-guard/state.json` records actual remote state. Launchers,
source/input manifests, CPU controls, raw traces, test logs and candidate comparisons
are all under that directory. No task-specific tests were added to `tests/`.

```powershell
kaggle kernels status claytoncrumbly/jumper-frontflip-rotation-guard-20261111
kaggle kernels output claytoncrumbly/jumper-frontflip-rotation-guard-20261111 -p out/frontflip-rotation-guard/gpu-output
.venv\Scripts\python.exe out/frontflip-rotation-guard/evaluate_candidates.py --download out/frontflip-rotation-guard/gpu-output --baseline out/frontflip-rotation-guard/kaggle-inputs/model_1500.pt --out out/frontflip-rotation-guard/evaluation
```

The completed comparisons do not accept the finalist as a deployed replacement.
**Use the +10 candidate as the next training warm start**, while retaining both the
original and +10 checkpoints as comparison controls. Failing promotion criteria does
not mean discarding its preserved rotation and reduced non-pad impacts. No additional
training was launched by this selection change.

The remaining objective is to reduce the worst pad loads without losing nominal
rotation, while improving randomized reliability. Existing Windows suite failures and
unverified controller gates are documented in the first-trial report; this remains
simulation-only work.

## Heading drift found while reviewing the video

The candidate's leftward turn is real, not a camera orbit. A read-only replay matched
the recorded touchdown and final forward rotation exactly. The video camera keeps its
azimuth relative to the first pose while following position; it does not turn around
the robot as its heading changes.

Nominal physics, one environment, seed 1234, 400 Hz physics / 200 Hz control:

| Policy / observation noise | Start-to-settled heading change |
|---|---:|
| +10 candidate / on — exact video trajectory | +23.59° left |
| +10 candidate / off | +24.24° left |
| Original / on | +28.59° left |
| Original / off | +28.77° left |

The reference's heading change is only -0.41°. The candidate slightly improves an
existing bias, but does not preserve its starting heading. Turning noise off does not
remove the drift, so sensor noise is not its main cause in this controlled comparison.
These are single-trajectory diagnostics, not a heading distribution over the holdouts.

An independent, read-only whole-robot momentum calculation shows yaw angular momentum
already present in flight: +0.0279 N·m·s at t=1.0 s, alongside +0.1997 N·m·s of pitch
momentum. Ground contact during launch therefore imparted off-axis spin. The specific
limb/contact responsible has not been isolated. Once upright around touchdown, heading
is +13.0° at t=1.30 s and reaches +23.6° while settling. Do not interpret Euler heading
while inverted as a physical heading-error measurement.

**The previous qualification definition was incomplete for a straight frontflip.**
Its first-touchdown guard checks accumulated forward rotation; landing qualification
also checks upright posture and six supports, but neither requires the original
heading. The existing full-orientation and angular-velocity rewards do discourage yaw,
but they are soft tracking terms, not an explicit heading-preservation requirement.
Thus “64/64 nominal qualified” does not mean “64/64 nominal flips with no heading drift.”

The next fine-tune should still start from +10, retain both comparison checkpoints,
and explicitly measure/preserve settled heading while reducing pad peaks and improving
randomized reliability. A heading acceptance tolerance must be declared for that
separately named experiment; none was retroactively added to the completed comparisons.
Diagnostic scripts, raw quaternion/momentum traces and summaries are in
`out/frontflip-rotation-guard/video/heading-*.json`.

## Watch the candidate

The stock `play --video` recorder produced a side-view nominal-physics example, with
observation noise on and the original 400 Hz physics / 200 Hz control rate:

- [Normal speed](../../../out/frontflip-rotation-guard/video/candidate-plus10-nominal-side.mp4)
- [Quarter speed](../../../out/frontflip-rotation-guard/video/candidate-plus10-nominal-quarter-speed.mp4)

The normal clip is 3.5 s, 1280×720, 60 fps. Quarter speed repeats those same recorded
frames; it does not simulate different dynamics. This individual attempt touched down
at 350.6° and settled on six supports at 355.4°. It is a nominal example, not evidence
that the randomized failures disappeared. Rendering metadata and the clean
first-episode trace are beside the video; a post-timeout reset sample is excluded from
reported final-state telemetry.

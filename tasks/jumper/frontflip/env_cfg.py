"""200 Hz tracking of a qualified, torque-limited frontflip reference."""

from __future__ import annotations

import math
import os
from pathlib import Path

from mjlab.envs import mdp as envs_mdp
from mjlab.managers import TerminationTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from ..common.constants import HOME
from ..common.assets import JUMPER_XML
from ..common.dance.rewards import peak_actuator_force, torque_headroom_cost
from ..jump.env_cfg import env_cfg as jump_env_cfg
from ..jump.mdp.observations import FUTURE_OFFSETS
from . import mdp
from .rl_cfg import agent_cfg
from .physics import (
    CHASSIS_FORCE_LIMIT, CONTROL_DT, PHYSICS_DT, SUPPORT_GEOMS, T_GO, robot_cfg,
    sensors,
)

REFERENCE = Path(__file__).resolve().parent / "ref" / "frontflip.npz"
# Limb impacts end an attempt only above 100 N; between 35 N (the reference's own
# landing peak) and 100 N they cost reward instead. LIMB_FORCE_LIMIT (50 N) stays
# the pass/fail line, read from the peak_limb_force metric.
LIMB_TERMINATION_N = 100.0
# Nov 11 corrected replay (Windows AMD CPU, native, 32 starts, cea9a567) rejected all
# load-sharing fine-tunes: the old -5 squared cost was masked on terminating impacts.
# It charged 102.4 at 99 N but only the flat 50 termination cost at 101 N. Price both.
# The old median landing stack was ~100, above discounted break-even ~62 (gamma=.99,
# 200 Hz), not the previously claimed +47 advantage. Start this separately named trial
# at one tenth the limb weight. CPU probe (same checkpoint, 32 starts, seed 1234,
# observation noise plus startup physics draws, 2026-11-11) reproduced 75.3 N median
# non-pad impact and 343.2 deg turn. Under this objective the measured discounted
# continuation minus an immediate -50 termination was +49.8 median / +29.8 minimum.
# This prices the existing landing; it is not evidence that the new policy improves it.
LIMB_PENALTY_FROM_N, LIMB_PENALTY_WEIGHT = 35.0, -0.5
# Scale-invariant sharing avoids an absolute knee that can reward reducing total support.
# Its maximum episode cost over this 0.6 s window is 10 * 5/6 * .6 = 5, independent of
# force magnitude. Rotation tracking and all physical termination limits remain unchanged.
LOAD_SHARE_WEIGHT = -10.0
LANDING_FROM_S, LANDING_UNTIL_S = 1.0, 1.6
LOAD_SHARE_MIN_ROTATION = math.radians(270.0)
LOAD_SHARE_MIN_TOTAL_N = 1.0
# Separately named pad-heading fine-tune, warm-started from the guarded +10 policy.
# Paired native CPU holdouts found LF/RM medians of 45.0/44.5 N; the exact video
# trajectory finishes 23.59 deg left (24.24 deg with noise off). Sharing alone does
# not price absolute pad force, and the existing orientation reward did not hold yaw.
PAD_LOAD_FROM_S, PAD_LOAD_UNTIL_S = 1.0, 2.0
# Native CPU, +10 warm start, 8 nominal HOME starts, noise on (2026-11-11):
# weight -0.15 charged only 0.427 median per episode against 16.14 heading cost.
# At -1.5, discounted continuation minus immediate termination remains +51.80
# median / +50.55 minimum nominal, and at least +35.55 for the 16/32 properly
# rotated randomized landings. Under-rotated failures are retained, not made soft.
PAD_LOAD_THRESHOLD_N, PAD_LOAD_MEAN_WEIGHT, PAD_LOAD_WEIGHT = 30.0, 0.5, -1.5
HEADING_FROM_S, HEADING_UNTIL_S = 0.5, 3.5
# The third trial reduced heading by only ~2 deg and did not preserve nominal
# qualification. Its warm-start yaw-rate charge was just 0.203 per episode at
# -0.5, versus 16.14 heading cost. Price the gyro/gravity-observable rate at -10;
# the actor has no absolute heading input. This revision still needs GPU validation.
HEADING_WEIGHT, YAW_VELOCITY_WEIGHT = -40.0, -10.0
# Evaluation requirement, not an immediate termination of the 23.6-deg warm start.
HEADING_ACCEPTANCE_RAD = math.radians(10.0)
FINE_TUNE_RANDOMIZATION_HOLD_UPDATES, FINE_TUNE_RANDOMIZATION_RAMP_UPDATES = 20, 80
LANDING_SUPPORT_POWER = 2.0
# First trial, Tesla T4, 256 envs, two seeds x 150 updates; paired native CPU replays,
# 8 nominal starts each (2026-11-11): all four candidates qualified 0/8 vs source 8/8.
# Seed 1234/+50 softened non-pad impacts to 52.6 N but turned only 334.9 deg median.
# Do not let a later ground rotation earn credit for that first under-rotated touchdown.
LANDING_MIN_ROTATION, LANDING_MAX_ROTATION = math.radians(340.0), math.radians(380.0)
LANDING_MIN_AIRTIME_S, LANDING_MIN_AIR_ROTATION = 0.015, math.radians(90.0)
# Flight clock feedback, exported to the robot (deploy/fsm/src/obs.rs); 0 disables. Zero
# residual on lean_turn, 128 HOME starts, seed 50311, native, Windows AMD CPU, off -> 50
# steps/rad: COM +-25 mm (2026-11-13) COM-only 72 -> 90, randomized 54 -> 66; BASE_COM_RANGE
# (2026-11-14) randomized 89 -> 109, hold only (2026-11-15) 114 (seed 7177: 124); a lead skip
# cost 5; nominal 128. max_hold 7-20, gain 70 and start +-0.04-0.08 s were no better.
CLOCK_FEEDBACK_GAIN = 50.0
# The feedback acts over this last stretch of the reference's flight.
CLOCK_FEEDBACK_LEAD_S = 0.24
# All four new nominal holdout losses touched down at 343.8-344.8 deg, inside
# the turn guard, but upright cosine 0.9616-0.9651 failed evaluation's 0.966.
# Latch pose and support deadline too; later recovery must not earn landing credit.
LANDING_MIN_UPRIGHT_COS, LANDING_SUPPORT_DEADLINE_S = 0.966, 0.1
# One-side-only support pair-steps after 1.15 s: zero residual 9, 1700 nominal 22, randomized 31.
# Touchdown only: to the episode end, lifting both feet of a pair would cost nothing.
ASYMMETRY_WEIGHT = -50.0
ASYMMETRY_FROM_S, ASYMMETRY_UNTIL_S = 1.15, 1.45
# The landing, pad-load and asymmetry windows above were tuned against frontflip.npz,
# which touches down at 1.285 s; they shift with the loaded reference's touchdown.
# Touchdowns: frontflip/soft 1.285 s, lean 1.815 s, lean_turn 1.840 s.
TUNED_TOUCHDOWN_S = 1.285
# Limb-on-limb contact in the tuck is penalized above 5 N and ends an attempt at 15 N.
SELF_TERMINATION_N = 15.0
SELF_PENALTY_FROM_N = 5.0
# Fixed cost for non-timeout termination, in addition to unmasked impact costs.
# This alone does not prove a landing wins; compare discounted reward-to-go in the probe.
# Probe, 64 HOME starts, Windows AMD CPU, native (2026-10-26): iteration 1950 earned 21.9 by
# ending on self-contact at 1.19 s, zero residual 12.8 by landing (41/64 then end at touchdown);
# landing wins only above 25. At 50 the same probe gives -47.4 against -35.0.
TERMINATION_PENALTY = 50.0
ROTATION_LAG_RAD = 1.5
# Every start is a HOME start. Run 5 update 1500 at 0.5 held the middle knees 0.41 rad in
# the launch crouch after landing, feet 11 mm up, until the reset (16 HOME starts, Windows
# AMD CPU, native, 2026-11-02); run_flip, fine-tuned from it without RSI, settled to 0.07.
RSI_FRACTION = 0.0
# PPO updates with nominal physics, then over which friction/COM/bias widen to full.
# With zero residual, 64/64 HOME attempts qualify without randomization and 4/64 with it
# (same probe); dropping any one source still leaves 10/64 or fewer.
RANDOMIZATION_START_UPDATES, RANDOMIZATION_RAMP_UPDATES = 500, 1000
RANDOMIZED_EVENTS = ("foot_friction", "encoder_bias", "base_com")
# Clean, dry floor: the silicone estimate in common/constants.py, not the dusty or wet low end.
# Zero residual at full randomization (192 HOME attempts, same probe) lands 3/63 below 0.8
# and 79/129 from 0.8 to 1.5; sampled over this range it lands 107/192.
FOOT_FRICTION_RANGE = (0.8, 1.5)
# Best guess, no robot to measure: +-5 mm whole-robot COM error. The offset moves base_link
# (0.886 of 2.543 kg), so +-15 mm here; mjlab's +-25/25/30 mm default was never chosen.
# Zero residual, feedback 50, 128 HOME starts, seed 50311, native, Windows AMD CPU
# (2026-11-14), all sources randomized: +-10 mm 121, +-15 mm 109 (89 without), +-25 mm 66.
BASE_COM_RANGE = {0: (-0.015, 0.015), 1: (-0.015, 0.015), 2: (-0.018, 0.018)}


def reference_touchdown_s(arrays) -> float:
    """First contact after the reference's longest flight, by qualify()'s airborne test."""
    airborne = (arrays["contact"].sum(axis=1) == 0) & (arrays["body_force"] < 1.0)
    run = longest = end = 0
    for i, value in enumerate(airborne):
        run = run + 1 if value else 0
        if run > longest:
            longest, end = run, i + 1
    if not longest or end >= len(airborne):
        raise ValueError("Reference has no flight followed by a touchdown")
    return float(arrays["t"][end])


def env_cfg(asset: Path | None = None, play: bool = False, reference: Path = REFERENCE,
            fine_tune_step: int | None = None):
    if asset is not None and Path(asset).resolve() != JUMPER_XML.resolve():
        raise ValueError("Frontflip reference is qualified against the default jumper.xml only")
    path = Path(os.environ.get("MJRL_FRONTFLIP_REFERENCE", str(reference))).expanduser()
    diagnostic = os.environ.get("MJRL_FRONTFLIP_ALLOW_DIAGNOSTIC") == "1"
    _, arrays = mdp.read_reference(path, diagnostic)
    touchdown = reference_touchdown_s(arrays)
    shift = touchdown - TUNED_TOUCHDOWN_S
    cfg = jump_env_cfg(asset=asset, play=play)
    cfg.sim.mujoco.timestep = PHYSICS_DT
    cfg.decimation = round(CONTROL_DT / PHYSICS_DT)
    cfg.scene.entities["robot"] = robot_cfg()
    if "foot_friction" in cfg.events:
        cfg.events["foot_friction"].params["asset_cfg"] = SceneEntityCfg(
            "robot", geom_names=SUPPORT_GEOMS,
        )
        cfg.events["foot_friction"].params["ranges"] = FOOT_FRICTION_RANGE
    if "base_com" in cfg.events:
        cfg.events["base_com"].params["ranges"] = dict(BASE_COM_RANGE)

    # A flip may invert freely; shell or hard limb impacts and inter-leg contacts end an attempt.
    cfg.scene.sensors = tuple(
        sensor for sensor in cfg.scene.sensors or ()
        if sensor.name not in ("body_ground_contact", "feet_ground_contact")
    ) + sensors(cfg.decimation) + (
        ContactSensorCfg(
            name="self_collision",
            primary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
            secondary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
            fields=("force",), reduce="maxforce", num_slots=1,
            history_length=cfg.decimation,
        ),
    )
    cfg.commands["jump"] = mdp.FrontflipCommandCfg(
        entity_name="robot", motion_file=str(path), allow_diagnostic=diagnostic,
        command_delay_range=(T_GO, T_GO), rsi_fraction=0.0 if play else RSI_FRACTION,
        landing_min_airtime_s=LANDING_MIN_AIRTIME_S,
        landing_min_air_rotation=LANDING_MIN_AIR_ROTATION,
        clock_feedback_gain=CLOCK_FEEDBACK_GAIN,
        clock_feedback_start_s=touchdown - CLOCK_FEEDBACK_LEAD_S,
    )
    reset = cfg.events["reset_from_reference"]
    reset.func = mdp.reset_from_reference_phase
    reset.params = {"command_name": "jump", "rsi_fraction": 0.0 if play else RSI_FRACTION,
                    "phase_range": (0.0, 0.9)}
    # External pushes obscure the initial physics-reference feasibility measurement.
    cfg.events.pop("push_robot", None)
    if not play:
        steps = agent_cfg().num_steps_per_env
        for name in RANDOMIZED_EVENTS:
            cfg.events[name].mode = "reset"
        if fine_tune_step is not None and fine_tune_step < 0:
            raise ValueError("Fine-tune start step must be nonnegative")
        start_step = RANDOMIZATION_START_UPDATES * steps
        ramp_steps = RANDOMIZATION_RAMP_UPDATES * steps
        if fine_tune_step is not None:
            # Keep the restored environment counter intact; anchor the new ramp to it.
            # Full evaluation ranges remain unchanged, never narrowed to flatter a run.
            start_step = fine_tune_step + FINE_TUNE_RANDOMIZATION_HOLD_UPDATES * steps
            ramp_steps = FINE_TUNE_RANDOMIZATION_RAMP_UPDATES * steps
        cfg.curriculum["randomization"] = CurriculumTermCfg(
            func=mdp.RandomizationRamp,
            params={"events": RANDOMIZED_EVENTS,
                    "start_step": start_step,
                    "ramp_steps": ramp_steps,
                    "nominal_friction": next(iter(
                        cfg.scene.entities["robot"].collisions[0].friction.values()))[0]},
        )
    cfg.terminations.pop("fell_over", None)
    for name, limit in (("chassis_ground_contact", CHASSIS_FORCE_LIMIT),
                        ("body_ground_contact", LIMB_TERMINATION_N),
                        ("self_collision", SELF_TERMINATION_N)):
        cfg.terminations[name] = TerminationTermCfg(
            func=mdp.collision_failure, params={"sensor_name": name, "force_threshold": limit}
        )
    # Up to 1.10 s, standing still lags 2.6 rad (run 2, update 450) and real flips lag 1.05 rad at
    # most (run 1 checkpoints, zero residual with full randomization; same probe as
    # TERMINATION_PENALTY). Later, under-rotated flips near touchdown would end too.
    cfg.terminations["rotation_lag"] = TerminationTermCfg(
        func=mdp.rotation_lag,
        params={"command_name": "jump", "threshold": ROTATION_LAG_RAD, "until_s": 1.10},
    )

    if not play:
        # Training-only constraint. Replay keeps the whole failed attempt visible so a
        # shorter, under-rotated episode cannot appear to have lower landing forces.
        cfg.terminations["landing_rotation"] = TerminationTermCfg(
            func=mdp.landing_rotation_failure,
            params={"command_name": "jump", "min_rotation": LANDING_MIN_ROTATION,
                    "max_rotation": LANDING_MAX_ROTATION},
        )
        cfg.terminations["landing_pose"] = TerminationTermCfg(
            func=mdp.landing_pose_failure,
            params={"command_name": "jump", "min_upright": LANDING_MIN_UPRIGHT_COS,
                    "max_support_delay_s": LANDING_SUPPORT_DEADLINE_S},
        )

    # Starting scales in rad² and (rad/s)², pending measured policy tracking runs.
    cfg.rewards["ref_base_ori"] = RewardTermCfg(
        func=mdp.ref_full_orientation, weight=2.0, params={"sigma_sq": 0.15}
    )
    cfg.rewards["ref_base_ang_vel"] = RewardTermCfg(
        func=mdp.ref_angular_velocity, weight=2.0, params={"sigma_sq": 4.0}
    )
    cfg.rewards["ref_forward_rotation"] = RewardTermCfg(
        func=mdp.ref_forward_rotation, weight=4.0, params={"sigma_sq": 0.1}
    )
    cfg.rewards["ref_contact"] = RewardTermCfg(
        func=mdp.ref_contact_found, weight=2.5,
        params={"command_name": "jump"},
    )
    # Landing cost at weight 60, run 5 and zero residual under full randomization (960 HOME
    # attempts, Windows AMD CPU, native, 2026-10-27): 45-55 N 9.8, 65-75 N 28.5, 75-100 N 50.
    # Weight 60 from scratch never landed (run 6); run 8 fine-tunes run 5 update 1500, its
    # output zeroed, at LIMB_PENALTY_WEIGHT.
    cfg.rewards["limb_impact"] = RewardTermCfg(
        func=mdp.contact_force_excess, weight=LIMB_PENALTY_WEIGHT,
        params={"sensor_name": "body_ground_contact", "threshold": LIMB_PENALTY_FROM_N,
                "power": 2.0},
    )
    cfg.rewards["foot_load"] = RewardTermCfg(
        func=mdp.landing_load_share, weight=LOAD_SHARE_WEIGHT,
        params={"command_name": "jump", "from_s": LANDING_FROM_S + shift,
                "until_s": LANDING_UNTIL_S + shift, "min_rotation": LOAD_SHARE_MIN_ROTATION,
                "min_total_force": LOAD_SHARE_MIN_TOTAL_N},
    )
    cfg.rewards["pad_load"] = RewardTermCfg(
        func=mdp.landing_pad_load, weight=PAD_LOAD_WEIGHT,
        params={"command_name": "jump", "from_s": PAD_LOAD_FROM_S + shift,
                "until_s": PAD_LOAD_UNTIL_S + shift, "threshold": PAD_LOAD_THRESHOLD_N,
                "mean_weight": PAD_LOAD_MEAN_WEIGHT},
    )
    cfg.rewards["heading"] = RewardTermCfg(
        func=mdp.ref_heading_cost, weight=HEADING_WEIGHT,
        params={"command_name": "jump", "from_s": HEADING_FROM_S,
                "until_s": HEADING_UNTIL_S},
    )
    cfg.rewards["yaw_velocity"] = RewardTermCfg(
        func=mdp.ref_yaw_velocity_cost, weight=YAW_VELOCITY_WEIGHT,
        params={"command_name": "jump", "from_s": HEADING_FROM_S,
                "until_s": HEADING_UNTIL_S},
    )
    cfg.rewards["support_asymmetry"] = RewardTermCfg(
        func=mdp.support_asymmetry, weight=ASYMMETRY_WEIGHT,
        params={"command_name": "jump", "from_s": ASYMMETRY_FROM_S + shift,
                "until_s": ASYMMETRY_UNTIL_S + shift},
    )
    cfg.rewards["self_impact"] = RewardTermCfg(
        func=mdp.contact_force_excess, weight=-20.0,
        params={"sensor_name": "self_collision", "threshold": SELF_PENALTY_FROM_N},
    )
    # At most 25 x 2.2 s = 55 from the reference touchdown to the episode end, about the
    # termination penalty: a completed, upright landing is worth what a crash costs.
    # Run 3 settled 0.22-0.32 rad short; at sigma_sq 0.25 that kept 82-66% of this term.
    # Same probe, recomputed: zero residual keeps 32.5 of 33.6, run 3 update 1750 keeps 13.9 of 29.0.
    # Continuous six-support credit instead of a binary three/six-foot switch: one support
    # earns 1/36, three earn 1/4, and six earn full credit. Keep the strict rotation/upright
    # factors; foot count alone must never qualify an under-rotated or reset HOME pose.
    cfg.rewards["flip_landing"] = RewardTermCfg(
        func=mdp.flip_landing, weight=25.0,
        params={"command_name": "jump", "sigma_sq": 0.05,
                "upright_cos": LANDING_MIN_UPRIGHT_COS, "min_feet": 1,
                "support_fraction_power": LANDING_SUPPORT_POWER,
                "min_touchdown_rotation": LANDING_MIN_ROTATION,
                "max_touchdown_rotation": LANDING_MAX_ROTATION,
                "min_touchdown_upright": LANDING_MIN_UPRIGHT_COS,
                "max_support_delay_s": LANDING_SUPPORT_DEADLINE_S},
    )
    cfg.rewards["termination"] = RewardTermCfg(
        func=envs_mdp.is_terminated, weight=-TERMINATION_PENALTY / cfg.sim.mujoco.timestep
        / cfg.decimation,
    )
    cfg.rewards["torque_headroom"] = RewardTermCfg(
        func=torque_headroom_cost, weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", actuator_names=(".*",))},
    )
    cfg.metrics["peak_torque"] = MetricsTermCfg(
        func=peak_actuator_force, reduce="max",
        params={"asset_cfg": SceneEntityCfg("robot", actuator_names=(".*",))},
    )
    cfg.metrics["peak_limb_force"] = MetricsTermCfg(
        func=mdp.contact_force, reduce="max", params={"sensor_name": "body_ground_contact"},
    )
    cfg.metrics["peak_self_force"] = MetricsTermCfg(
        func=mdp.contact_force, reduce="max", params={"sensor_name": "self_collision"},
    )
    # Include the HOME lead-in for attempts that start from HOME instead of a reference phase.
    cfg.episode_length_s = float(arrays["t"][-1]) + 0.5
    cfg.viewer.distance = 1.2

    def reference_contract():
        reference = mdp.FrontflipReference(path, "cpu", list(HOME), diagnostic)
        contract = reference.deploy_contract(FUTURE_OFFSETS, CONTROL_DT, starts_on_entry=True)
        jump = cfg.commands["jump"]
        if jump.clock_feedback_gain > 0:
            # deploy/fsm/src/obs.rs runs the same feedback from the gyro.
            contract["clock_feedback"] = {
                "gain": jump.clock_feedback_gain, "start_s": jump.clock_feedback_start_s,
                "max_hold": jump.clock_feedback_max_hold}
            contract["reference"]["forward_rotation"] = reference.forward_rotation.tolist()
        return contract

    cfg.reference_contract = reference_contract
    return cfg

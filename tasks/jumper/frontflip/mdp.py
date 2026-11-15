"""Task-local flip reference, angular-state reset, tracking and failure checks."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from mjlab.utils.lab_api.math import (
    quat_apply_inverse,
    quat_error_magnitude,
    quat_inv,
    quat_mul,
)

from ..jump.mdp.commands import JumpMotionCommand, JumpMotionCommandCfg
from ..jump.mdp.reference import JumpReference, ref_state
from ..common.assets import JUMPER_XML
from ..common.constants import HOME
from .physics import CONTROL_DT, KD, KP, T_GO, fingerprint, qualify, signed_rotation, support_contacts


def read_reference(path: Path, allow_diagnostic: bool = False) -> tuple[dict, dict]:
    """Reject incompatible, malformed or unqualified references before building a scene."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Frontflip reference missing: {path}. Run tools/search_frontflip.py "
            "first; train only after its physics rollout qualifies. See the task README."
        )
    with np.load(path, allow_pickle=False) as archive:
        meta = json.loads(str(archive["meta"].item()))
        keys = (
            "t", "q", "qd", "q_cmd", "base_pos", "base_quat", "base_lin_vel",
            "base_ang_vel", "foot_pos", "contact", "forward_rotation", "tau",
            "body_force", "self_force", "chassis_force",
        )
        arrays = {key: archive[key] for key in keys}
        if meta.get("schedule") == "rear-claw":
            arrays.update({key: archive[key]
                           for key in ("stage", "foot_force", "com_ahead", "momentum")})
            if meta.get("foot_force_convention") != "world-frame terrain reaction acting on each foot":
                raise ValueError("Rear-claw reference must record terrain reaction acting on the feet")
    names = meta["joint_order"]
    if len(names) != len(set(names)) or set(names) != set(HOME):
        raise ValueError("Reference joint_order must name every robot joint exactly once")
    n, joints = len(arrays["t"]), len(names)
    shapes = {
        "t": (n,), "q": (n, joints), "qd": (n, joints), "q_cmd": (n, joints),
        "base_pos": (n, 3), "base_quat": (n, 4), "base_lin_vel": (n, 3),
        "base_ang_vel": (n, 3), "foot_pos": (n, 6, 3), "contact": (n, 6),
        "forward_rotation": (n,), "tau": (n, joints), "body_force": (n,), "self_force": (n,),
        "chassis_force": (n,),
    }
    if meta.get("schedule") == "rear-claw":
        shapes.update(stage=(n,), foot_force=(n, 6, 3), com_ahead=(n,), momentum=(n, 6))
    for key, array in arrays.items():
        if array.shape != shapes[key] or array.dtype != np.float32:
            raise ValueError(f"{key}: expected float32 {shapes[key]}, got {array.dtype} {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError(f"{key} contains non-finite samples")
    if n < 2 or abs(float(arrays["t"][0])) > 1e-6:
        raise ValueError("Reference must have at least two samples and start at t=0")
    if not np.allclose(np.diff(arrays["t"]), CONTROL_DT, atol=1e-6, rtol=0):
        raise ValueError("Reference times must increase uniformly at the 200 Hz control rate")
    if not np.allclose(np.linalg.norm(arrays["base_quat"], axis=1), 1.0, atol=1e-3):
        raise ValueError("Reference base_quat must contain unit wxyz quaternions")
    if not np.isin(arrays["contact"], (0.0, 1.0)).all():
        raise ValueError("Reference contact must contain six binary foot-contact channels")
    if meta.get("schedule") == "rear-claw":
        if not np.isin(arrays["stage"], np.arange(-1, 7)).all():
            raise ValueError("Rear-claw reference has an unknown controller stage")
        if not np.isin(arrays["com_ahead"], (0., 1.)).all():
            raise ValueError("Rear-claw COM feedback must be binary")
    if meta.get("model_fingerprint") != fingerprint():
        raise ValueError("Reference model/motor fingerprint differs; regenerate it for this checkout")
    for key, expected in (("KP", KP), ("KD", KD), ("rec_hz", 1 / CONTROL_DT),
                          ("control_hz", 1 / CONTROL_DT)):
        if meta.get(key) != expected:
            raise ValueError(f"Reference {key}={meta.get(key)!r}; expected {expected}")
    go_frame = meta["go_frame"]
    if (isinstance(go_frame, bool) or not isinstance(go_frame, int)
            or not 0 <= go_frame < n - 1):
        raise ValueError("Reference go_frame must select a sample before its endpoint")
    if not np.isclose(meta["t_go"], arrays["t"][go_frame], atol=1e-6):
        raise ValueError("Reference t_go must agree with go_frame")
    if (go_frame != round(T_GO / CONTROL_DT)
            or not np.isclose(meta["t_go"], T_GO, atol=1e-6, rtol=0)):
        raise ValueError(f"Reference go timing must match this task's {T_GO:.2f} s launch clock")
    if not np.isfinite(meta["stand_base_z"]):
        raise ValueError("Reference stand_base_z must be finite")
    if not isinstance(meta.get("qualification"), dict):
        raise ValueError("Reference must carry its measured qualification report")
    rotation = signed_rotation(arrays["base_quat"])
    if not np.allclose(arrays["forward_rotation"], rotation, atol=1e-4, rtol=0):
        raise ValueError("Reference accumulated rotation disagrees with its quaternion trajectory")
    limits = {joint.get("name"): np.fromstring(joint.get("range"), sep=" ")
              for joint in ET.parse(JUMPER_XML).iter("joint") if joint.get("range")}
    ranges = np.stack([limits[name] for name in names])
    if ((arrays["q_cmd"] < ranges[:, 0] + .08)
            | (arrays["q_cmd"] > ranges[:, 1] - .08)).any():
        raise ValueError("Reference commanded targets violate the XML limits or 0.08 rad margin")
    violation = float(np.maximum(ranges[:, 0] - arrays["q"], arrays["q"] - ranges[:, 1]).max())
    measured = qualify(dict(arrays, joint_names=names, joint_limit_violation=violation,
                            schedule=meta.get("schedule", "jump")))
    if meta["qualification"].get("checks") != measured["checks"]:
        raise ValueError("Reference qualification report disagrees with measured trajectory")
    qualified = meta.get("qualified") is True and meta.get("diagnostic") is False
    if qualified:
        report = meta["qualification"]
        if report.get("qualified") is not True or not measured["qualified"]:
            raise ValueError("Qualified metadata disagrees with its qualification checks")
    if meta.get("angular_velocity_frame") != "world":
        raise ValueError("Frontflip angular velocity must be recorded in world coordinates")
    if not qualified and not allow_diagnostic:
        raise ValueError(
            "Reference is not a qualified frontflip. Inspect its qualification report "
            "and continue the physics search. MJRL_FRONTFLIP_ALLOW_DIAGNOSTIC=1 "
            "allows an explicit environment smoke test only."
        )
    return meta, arrays


class FrontflipReference(JumpReference):
    """Jump's reference API with measured contacts and angular state for a full revolution."""

    def __init__(self, path: Path, device: str, joint_names: list[str],
                 allow_diagnostic: bool = False):
        meta, arrays = read_reference(path, allow_diagnostic)
        source = meta["joint_order"]
        missing = set(joint_names) - set(source)
        if missing:
            raise ValueError(f"Frontflip reference missing robot joints: {sorted(missing)}")
        perm = [source.index(name) for name in joint_names]
        self.path, self.meta, self.joint_names = path, meta, list(joint_names)
        self.n = len(arrays["t"])
        self.rec_hz, self.control_hz = float(meta["rec_hz"]), float(meta["control_hz"])
        self.dt = 1.0 / self.rec_hz
        self.t_go, self.go_frame = float(meta["t_go"]), int(meta["go_frame"])
        self.duration = float(arrays["t"][-1])
        self.span = self.duration - self.t_go
        self.stand_base_z = float(meta["stand_base_z"])
        for key in arrays:
            value = arrays[key][:, perm] if key in ("q", "qd", "q_cmd") else arrays[key]
            setattr(self, key, torch.as_tensor(value, dtype=torch.float32, device=device))
        rel = self.foot_pos - self.base_pos.unsqueeze(1)
        q = self.base_quat.unsqueeze(1).expand(-1, 6, -1).reshape(-1, 4)
        self.foot_b = quat_apply_inverse(q, rel.reshape(-1, 3)).reshape(rel.shape)
        gravity = self.base_pos.new_tensor([0.0, 0.0, -1.0]).expand(self.n, 3)
        self.proj_g = quat_apply_inverse(self.base_quat, gravity)

    def sample(self, idx: torch.Tensor) -> dict[str, torch.Tensor]:
        sample = super().sample(idx)
        lo = idx.floor().long().clamp(0, self.n - 1)
        hi = (lo + 1).clamp(0, self.n - 1)
        w = (idx - lo.float()).unsqueeze(-1)
        q0, q1 = self.base_quat[lo], self.base_quat[hi]
        # Antipodal quaternions describe the same pose; interpolating through zero does not.
        q1 = torch.where((q0 * q1).sum(-1, keepdim=True) < 0, -q1, q1)
        quat = q0 * (1 - w) + q1 * w
        sample["base_quat"] = quat / quat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        sample["base_ang_vel"] = self.base_ang_vel[lo] * (1 - w) + self.base_ang_vel[hi] * w
        scalar_w = w.squeeze(-1)
        sample["forward_rotation"] = (
            self.forward_rotation[lo] * (1 - scalar_w) + self.forward_rotation[hi] * scalar_w
        )
        return sample


@dataclass(kw_only=True)
class FrontflipCommandCfg(JumpMotionCommandCfg):
    landing_min_airtime_s: float
    landing_min_air_rotation: float
    allow_diagnostic: bool = False
    # Flight clock feedback: held reference steps per rad of forward-turn lag; 0 disables.
    clock_feedback_gain: float = 0.0
    clock_feedback_start_s: float = 0.0
    clock_feedback_max_hold: int = 10

    def __post_init__(self):
        self.class_type = FrontflipCommand

    def build(self, env):
        return FrontflipCommand(self, env)


class FrontflipCommand(JumpMotionCommand):
    def __init__(self, cfg: FrontflipCommandCfg, env):
        # Reuse jump machinery through its per-environment cache, without changing module globals.
        env._jump_reference = FrontflipReference(
            Path(cfg.motion_file), env.device, list(env.scene[cfg.entity_name].joint_names),
            cfg.allow_diagnostic,
        )
        super().__init__(cfg, env)
        self.rotation = torch.zeros(self.num_envs, device=self.device)
        self.previous_quat = self.robot.data.root_link_quat_w.clone()
        self._rotation_stamp = -1
        self.touchdown_rotation = torch.full_like(self.rotation, float("nan"))
        self.touchdown_upright = torch.full_like(self.rotation, float("nan"))
        self.touchdown_step = torch.full_like(self.rotation, -1, dtype=torch.long)
        self.first_six_step = torch.full_like(self.rotation, -1, dtype=torch.long)
        self._landing_air_steps = torch.zeros_like(self.rotation, dtype=torch.long)
        self._landing_flew = torch.zeros_like(self.rotation, dtype=torch.bool)
        self._landing_stamp = -1
        self._clock_target = torch.full_like(self.touchdown_step, -1)
        self._clock_shift = torch.zeros_like(self.touchdown_step)
        if cfg.landing_min_airtime_s <= 0:
            raise ValueError("Landing airtime must be positive")
        self.metrics["forward_rotation_rad"] = self.rotation

    def _resample_command(self, env_ids):
        phase = getattr(self._env, "_jump_rsi_phase", torch.zeros_like(self.rotation))
        phase = phase[env_ids].clone()
        sample = self.reference.sample(self.reference.index_of(phase * self.reference.span))
        super()._resample_command(env_ids)
        self.rotation[env_ids] = sample["forward_rotation"]
        self.previous_quat[env_ids] = sample["base_quat"]
        self.touchdown_rotation[env_ids] = float("nan")
        self.touchdown_upright[env_ids] = float("nan")
        self.touchdown_step[env_ids] = -1
        self.first_six_step[env_ids] = -1
        self._landing_air_steps[env_ids] = 0
        self._landing_flew[env_ids] = False
        self._clock_target[env_ids] = -1
        self._clock_shift[env_ids] = 0
        # A reset changes go_step without advancing common_step_counter.
        self._env._jump_ref_stamp = None

    def measured_rotation(self):
        # Rewards run before command.compute(); update once at the reward's current state.
        stamp = int(self._env.common_step_counter)
        if stamp != self._rotation_stamp:
            quat = self.robot.data.root_link_quat_w
            delta = quat_mul(quat, quat_inv(self.previous_quat))
            delta = torch.where(delta[:, :1] < 0, -delta, delta)
            length = delta[:, 1:].norm(dim=-1)
            angle = 2 * torch.atan2(length, delta[:, 0])
            self.rotation += angle * delta[:, 2] / length.clamp_min(1e-8)
            self.previous_quat.copy_(quat)
            self._rotation_stamp = stamp
        return self.rotation

    def update_landing_state(self):
        """Latch the first post-flight touchdown turn once per attempt, never recovery turn.

        Like measured_rotation(), rewards/terminations and command.compute() may all ask;
        advancing at most once per control step keeps those readers consistent.
        """
        stamp = int(self._env.common_step_counter)
        if stamp == self._landing_stamp:
            return
        turn = self.measured_rotation()
        contacts = support_contacts(self._env)
        airborne = ~contacts.any(dim=1)
        self._landing_air_steps = torch.where(
            airborne, self._landing_air_steps + 1, torch.zeros_like(self._landing_air_steps))
        min_steps = max(1, round(self.cfg.landing_min_airtime_s / self._env.step_dt))
        self._landing_flew |= ((self._landing_air_steps >= min_steps)
                              & (turn >= self.cfg.landing_min_air_rotation))
        first = ~airborne & self._landing_flew & torch.isnan(self.touchdown_rotation)
        self.touchdown_rotation[first] = turn[first]
        quat = self.robot.data.root_link_quat_w
        upright = 1 - 2 * (quat[:, 1].square() + quat[:, 2].square())
        self.touchdown_upright[first] = upright[first]
        self.touchdown_step[first] = stamp
        first_six = ((self.touchdown_step >= 0) & (self.first_six_step < 0)
                     & (contacts.sum(dim=1) == 6))
        self.first_six_step[first_six] = stamp
        self._landing_stamp = stamp

    def _update_command(self, env_ids=None):
        if env_ids is not None:
            return
        z = self.robot.data.root_link_pos_w[:, 2] - self._env.scene.env_origins[:, 2]
        torch.maximum(self._apex_rise, z - self._stance_z, out=self._apex_rise)
        airborne = ~support_contacts(self._env).any(dim=1)
        self._flight_s += airborne.float() * self._env.step_dt
        grounded = ~airborne
        vz = self.robot.data.root_link_lin_vel_w[:, 2]
        self._takeoff_vz[grounded] = torch.maximum(self._takeoff_vz[grounded], vz[grounded])
        self.update_landing_state()
        if self.cfg.clock_feedback_gain > 0:
            self._clock_feedback()

    def _clock_feedback(self):
        """Hold the reference clock when the flip lags, so the un-tuck waits for the turn.

        At the start time, lag sets a hold of gain*lag steps, capped; a lead does nothing.
        deploy/fsm/src/obs.rs mirrors this.
        """
        cfg = self.cfg
        tsg = self.time_since_go
        lag = self.reference.sample(self.reference.index_of(tsg))["forward_rotation"] \
            - self.measured_rotation()
        t_ref = tsg + self.reference.t_go
        fresh = (t_ref >= cfg.clock_feedback_start_s) & (self._clock_target < 0)
        target = (cfg.clock_feedback_gain * lag).round().clamp(0, cfg.clock_feedback_max_hold)
        self._clock_target = torch.where(fresh, target.long(), self._clock_target)
        hold = (self._clock_target >= 0) & (self._clock_shift < self._clock_target)
        self.go_step += hold.float()
        self._clock_shift += hold.long()
        # Observations later this step must sample the moved clock.
        self._env._jump_ref_stamp = None


def reset_from_reference_phase(env, env_ids, command_name="jump", rsi_fraction=1.0,
                               phase_range=(0.0, 0.9)):
    """Restore pose and both world-frame root velocities, including the flip's spin."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    cmd = env.command_manager.get_term(command_name)
    phase = torch.empty(len(env_ids), device=env.device).uniform_(*phase_range)
    use_rsi = torch.rand(len(env_ids), device=env.device) < rsi_fraction
    if not hasattr(env, "_jump_rsi_phase"):
        env._jump_rsi_phase = torch.zeros(env.num_envs, device=env.device)
    env._jump_rsi_phase[env_ids] = torch.where(use_rsi, phase, torch.zeros_like(phase))
    ids = env_ids[use_rsi]
    if not len(ids):
        return
    sample = cmd.reference.sample(cmd.reference.index_of(phase[use_rsi] * cmd.reference.span))
    root = torch.cat((sample["base_pos"] + env.scene.env_origins[ids], sample["base_quat"],
                      sample["base_lin_vel"], sample["base_ang_vel"]), dim=-1)
    robot = env.scene["robot"]
    robot.write_root_state_to_sim(root, env_ids=ids)
    robot.write_joint_state_to_sim(sample["q"], sample["qd"], env_ids=ids)


def ref_full_orientation(env, command_name="jump", sigma_sq=0.15):
    _, sample, _, _ = ref_state(env, command_name)
    error = quat_error_magnitude(env.scene["robot"].data.root_link_quat_w, sample["base_quat"])
    return torch.exp(-error.square() / sigma_sq)


def reference_heading_error(env, command_name="jump"):
    """World-up twist error against the phase-matched reference, also while inverted.

    Projecting the body's forward axis becomes singular during a forward flip.
    Instead remove the reference rotation, then extract its world-z twist. Use
    antipodal-invariant magnitudes; undefined 180-degree swing is not zero error.
    """
    _, sample, _, _ = ref_state(env, command_name)
    current = env.scene["robot"].data.root_link_quat_w
    error = quat_mul(current, quat_inv(sample["base_quat"]))
    w, z = error[:, 0].abs(), error[:, 3].abs()
    angle = 2 * torch.atan2(z, w.clamp_min(1e-8))
    return torch.where(w.square() + z.square() > 1e-8, angle, torch.full_like(angle, torch.pi))


def ref_heading_cost(env, command_name, from_s, until_s):
    """Squared reference-relative yaw; no contact/termination mask to hide a bad landing."""
    reference, _, _, tsg = ref_state(env, command_name)
    t = tsg + reference.t_go
    active = (t >= from_s) & (t < until_s)
    return reference_heading_error(env, command_name).square() * active


def ref_yaw_velocity_cost(env, command_name, from_s, until_s):
    """Price off-axis yaw rate independently of the much larger forward-flip rate."""
    reference, sample, _, tsg = ref_state(env, command_name)
    error = env.scene["robot"].data.root_link_ang_vel_w[:, 2] - sample["base_ang_vel"][:, 2]
    t = tsg + reference.t_go
    return error.square() * ((t >= from_s) & (t < until_s))


def ref_angular_velocity(env, command_name="jump", sigma_sq=4.0):
    _, sample, _, _ = ref_state(env, command_name)
    error = env.scene["robot"].data.root_link_ang_vel_w - sample["base_ang_vel"]
    return torch.exp(-error.square().sum(dim=-1) / sigma_sq)


def ref_forward_rotation(env, command_name="jump", sigma_sq=0.25):
    _, sample, _, _ = ref_state(env, command_name)
    cmd = env.command_manager.get_term(command_name)
    error = cmd.measured_rotation() - sample["forward_rotation"]
    return torch.exp(-error.square() / sigma_sq)


def ref_contact_found(env, command_name="jump"):
    """Match the geometric contacts recorded by the physics search, including unloaded feet."""
    _, sample, _, _ = ref_state(env, command_name)
    actual = support_contacts(env)
    return (actual == sample["contact"].bool()).float().mean(dim=1)


def flip_landing(env, command_name="jump", sigma_sq=0.25, upright_cos=0.866, min_feet=3,
                 from_s=1.30, support_fraction_power=0.0,
                 min_touchdown_rotation=None, max_touchdown_rotation=None,
                 min_touchdown_upright=None, max_support_delay_s=None):
    """Closeness of the accumulated turn to one revolution, while landed on the feet and upright."""
    reference, _, _, tsg = ref_state(env, command_name)
    turn = env.command_manager.get_term(command_name).measured_rotation()
    quat = env.scene["robot"].data.root_link_quat_w
    upright = 1 - 2 * (quat[:, 1].square() + quat[:, 2].square())
    landed = (support_contacts(env).sum(dim=1) >= min_feet) & (upright > upright_cos)
    if min_touchdown_rotation is not None:
        cmd = env.command_manager.get_term(command_name)
        cmd.update_landing_state()
        landed &= torch.isfinite(cmd.touchdown_rotation)
        landed &= cmd.touchdown_rotation >= min_touchdown_rotation
        if max_touchdown_rotation is not None:
            landed &= cmd.touchdown_rotation <= max_touchdown_rotation
        if min_touchdown_upright is not None:
            landed &= cmd.touchdown_upright >= min_touchdown_upright
        if max_support_delay_s is not None:
            delay = (cmd.first_six_step - cmd.touchdown_step).float() * env.step_dt
            landed &= ((cmd.first_six_step >= 0) & (delay <= max_support_delay_s + 1e-8))
    active = tsg + reference.t_go >= from_s
    fraction = (support_contacts(env).float().mean(dim=1) ** support_fraction_power)
    return (torch.exp(-(turn - 2 * torch.pi).square() / sigma_sq)
            * (landed & active).float() * fraction)


def contact_force(env, sensor_name="body_ground_contact"):
    """Largest terrain force on the sensor's geoms across this step's substeps [N]."""
    history = env.scene[sensor_name].data.force_history
    return history.norm(dim=-1).flatten(1).max(dim=1).values


def contact_force_excess(env, sensor_name="body_ground_contact", threshold=35.0, power=1.0):
    """Force above `threshold`, including the impact that terminates an attempt."""
    return (contact_force(env, sensor_name) - threshold).clamp_min(0.0) ** power


def support_force_history(env):
    """[N, 6, H, 3] simultaneous world-frame terrain reactions, newest substep first.

    Sum both claw pads as vectors before taking magnitudes: two pads belong to one limb.
    The sensor stores primary-to-terrain forces; negate for reaction on the robot.
    """
    rear = env.scene["feet_ground_contact"].data.force_history
    front = env.scene["claw_front_ground_contact"].data.force_history
    if rear.shape[1] != 6 or front.shape[1] != 2 or rear.shape[2:] != front.shape[2:]:
        raise ValueError("Frontflip requires six rear-pad and two front-pad force histories")
    return -torch.cat((rear[:, :2] + front, rear[:, 2:]), dim=1)


def support_peak_forces(env):
    """[N, 6] largest simultaneous substep reaction per limb, LF RF LM RM LR RR [N]."""
    return support_force_history(env).norm(dim=-1).amax(dim=-1)


def foot_load_excess(env, threshold=45.0):
    """Squared per-foot force above `threshold`, summed over the six supports."""
    return (support_peak_forces(env) - threshold).clamp_min(0.0).square().sum(dim=1)


def landing_pad_load(env, command_name, from_s, until_s, threshold, mean_weight):
    """Price worst-pad and mean-pad squared overload at simultaneous physics substeps.

    Equal sharing of a large impact must still cost more than a gentle landing.
    Preserve the vector sum across both claw pads, then aggregate magnitudes.
    The phase gate does not depend on turn or termination, so an incorrectly
    rotated, terminating impact inside the window still pays its full cost.
    """
    reference, _, _, tsg = ref_state(env, command_name)
    overload = (support_force_history(env).norm(dim=-1) - threshold).clamp_min(0.0).square()
    cost = overload.amax(dim=1) + mean_weight * overload.mean(dim=1)
    t = tsg + reference.t_go
    return cost.mean(dim=-1) * ((t >= from_s) & (t < until_s))


def landing_load_share(env, command_name, from_s, until_s, min_rotation, min_total_force):
    """Bounded, scale-invariant six-limb imbalance during the landing window.

    Each substep uses simultaneous vertical reactions, not independently timed peaks.
    Zero for equal shares; at most 5/6 when one limb carries the entire load.
    No termination mask: a crash must not erase its own cost.
    """
    reference, _, _, tsg = ref_state(env, command_name)
    forces = support_force_history(env)[..., 2].clamp_min(0.0)
    total = forces.sum(dim=1)
    shares = forces / total.unsqueeze(1).clamp_min(min_total_force)
    imbalance = (shares - 1.0 / 6.0).square().sum(dim=1)
    loaded = total >= min_total_force
    t = tsg + reference.t_go
    turn = env.command_manager.get_term(command_name).measured_rotation()
    active = (t >= from_s) & (t < until_s) & (turn >= min_rotation)
    return (imbalance * loaded).mean(dim=-1) * active


def support_asymmetry(env, command_name="jump", from_s=1.15, until_s=1.45):
    """Left/right support pairs with only one side on the ground, between `from_s` and `until_s`."""
    reference, _, _, tsg = ref_state(env, command_name)
    contacts = support_contacts(env)
    odd = (contacts[:, 0::2] ^ contacts[:, 1::2]).sum(dim=1).float()
    t = tsg + reference.t_go
    return odd * ((t >= from_s) & (t < until_s))


class RandomizationRamp:
    """Scale the reset-time friction, COM and encoder-bias ranges from nominal to full.

    The level is 0 before `start_step`, then rises linearly to 1 over `ramp_steps`.
    """

    def __init__(self, cfg, env):
        p = cfg.params
        self.start, self.ramp = p["start_step"], p["ramp_steps"]
        # The event manager samples from its own deep copy of the term configs.
        self.events = {name: env.event_manager.cfg[name].params for name in p["events"]}
        self.full = {name: _range_copy(params) for name, params in self.events.items()}
        self.nominal_friction = p["nominal_friction"]

    def reset(self, env_ids=None):
        pass

    def __call__(self, env, env_ids, **_):
        level = min(max((env.common_step_counter - self.start) / max(self.ramp, 1), 0.0), 1.0)
        nominal = self.nominal_friction
        lo, hi = self.full["foot_friction"]["ranges"]
        self.events["foot_friction"]["ranges"] = (nominal + level * (lo - nominal),
                                                  nominal + level * (hi - nominal))
        lo, hi = self.full["encoder_bias"]["bias_range"]
        self.events["encoder_bias"]["bias_range"] = (level * lo, level * hi)
        self.events["base_com"]["ranges"] = {
            axis: (level * lo, level * hi) for axis, (lo, hi) in self.full["base_com"]["ranges"].items()
        }
        return torch.tensor(level)


def _range_copy(params):
    return {key: dict(value) if isinstance(value, dict) else value for key, value in params.items()
            if key in ("ranges", "bias_range")}


def rotation_lag(env, command_name="jump", threshold=1.5, until_s=1.10):
    """End an attempt whose forward turn is `threshold` rad from the reference before `until_s`."""
    reference, sample, _, tsg = ref_state(env, command_name)
    cmd = env.command_manager.get_term(command_name)
    lag = (cmd.measured_rotation() - sample["forward_rotation"]).abs()
    return (lag > threshold) & (tsg + reference.t_go <= until_s)


def landing_rotation_failure(env, command_name, min_rotation, max_rotation):
    """Reject incomplete/extra turns at first touchdown, not a later on-ground recovery."""
    cmd = env.command_manager.get_term(command_name)
    cmd.update_landing_state()
    turn = cmd.touchdown_rotation
    return torch.isfinite(turn) & ((turn < min_rotation) | (turn > max_rotation))


def landing_pose_failure(env, command_name, min_upright, max_support_delay_s):
    """Reject tilted first touchdown or six supports missed by the landing deadline.

    Later recovery cannot erase either failure. Replay omits this training term
    so the complete failed attempt remains available for force measurements.
    """
    cmd = env.command_manager.get_term(command_name)
    cmd.update_landing_state()
    touched = cmd.touchdown_step >= 0
    tilted = touched & (cmd.touchdown_upright < min_upright)
    deadline_steps = max(1, round(max_support_delay_s / env.step_dt))
    elapsed = int(env.common_step_counter) - cmd.touchdown_step
    timely = ((cmd.first_six_step >= 0)
              & ((cmd.first_six_step - cmd.touchdown_step) <= deadline_steps))
    late = touched & (elapsed >= deadline_steps) & ~timely
    return tilted | late


def collision_failure(env, sensor_name, force_threshold=1.0):
    sensor = env.scene[sensor_name]
    history = sensor.data.force_history
    if history is None:
        raise ValueError(f"{sensor_name} must retain force history across physics substeps")
    return (history.norm(dim=-1) > force_threshold).flatten(1).any(dim=1)

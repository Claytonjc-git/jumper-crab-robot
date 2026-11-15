"""Keep the physical reset and the one-shot reference clock synchronized."""

import hashlib
import os
from pathlib import Path

import torch

from mjlab.rl.runner import MjlabOnPolicyRunner

# Set to 1 to load a checkpoint trained on another (or an unrecorded) reference.
TRANSFER_ENV = "MJRL_FRONTFLIP_REFERENCE_TRANSFER"


def _reference_tag(env) -> dict:
    path = Path(env.unwrapped.command_manager.get_term("jump").reference.path)
    return {"name": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


class FrontflipRunner(MjlabOnPolicyRunner):
    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        # A random clock would ask a HOME pose to track mid-flight. Instead lengthen each
        # first episode's pre-go HOME wait, so rollouts do not all hold one flip phase.
        if init_at_random_ep_len:
            env = self.env.unwrapped
            env.episode_length_buf = -torch.randint_like(
                env.episode_length_buf, int(env.max_episode_length))
            print("[frontflip] staggered first episodes by a random HOME wait")
        super().learn(num_learning_iterations, init_at_random_ep_len=False)

    def save(self, path: str, infos=None) -> None:
        super().save(path, {**(infos or {}), "frontflip_reference": _reference_tag(self.env)})

    def load(self, path: str, *args, **kwargs) -> dict:
        """Refuse a checkpoint whose recorded reference differs from the loaded one."""
        saved = (torch.load(path, map_location="cpu", weights_only=False).get("infos") or {}
                 ).get("frontflip_reference")
        current = _reference_tag(self.env)
        if saved is None or saved["sha256"] != current["sha256"]:
            found = "no reference" if saved is None else f"{saved['name']} {saved['sha256'][:12]}"
            message = (f"{path} records {found}; this environment loaded "
                       f"{current['name']} {current['sha256'][:12]}")
            if os.environ.get(TRANSFER_ENV) != "1":
                raise ValueError(f"[frontflip] {message}. Set MJRL_FRONTFLIP_REFERENCE to the "
                                 f"checkpoint's reference, or {TRANSFER_ENV}=1 to transfer.")
            print(f"[frontflip] reference transfer: {message}")
        return super().load(path, *args, **kwargs)

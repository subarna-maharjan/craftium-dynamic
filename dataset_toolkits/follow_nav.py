"""Keep ONE dynamic agent inside the player's frame, without staring at it.

The guided tour in `guided_nav.py` turns to face each mob in turn, which makes the
player rotate constantly and parks every mob dead centre. This controller is looser and
is what the dataset wants:

  * one target mob is chosen at random per episode (so it differs between games);
  * a comfort distance is drawn per episode, so the mob is not always the same size in
    frame - some games follow from 4 nodes away, some from 14;
  * the camera only turns when the mob is about to leave the frame. Inside that dead
    zone the player is free to look wherever it likes, so the mob drifts around the
    frame instead of being centred;
  * the approach direction is unconstrained - the player closes distance by walking
    forward and otherwise strafes or idles, so it follows from the side, at an angle,
    or from in front just as often as from behind.

The target is only required to be *in frame*, never centred and never chased.
"""
from __future__ import annotations

import math
import random

from guided_nav import GuidedNavigator, _wrap


class FollowOneNavigator(GuidedNavigator):
    def __init__(
        self,
        run_dir: str,
        actions,
        action_shape,
        *,
        follow_min_distance: float = 3.5,
        follow_max_distance: float = 14.0,
        frame_margin_deg: float = 12.0,
        distance_slack: float = 2.5,
        rng: random.Random | None = None,
        **kwargs,
    ):
        super().__init__(run_dir, actions, action_shape, **kwargs)
        # the base class resolves a few action indices but does not keep the table
        self._actions0 = list(actions[0]) if actions else []
        self.rng = rng or random.Random()
        # One comfort distance for the whole episode: varies game to game.
        self.want_dist = self.rng.uniform(follow_min_distance, follow_max_distance)
        self.near = max(1.5, self.want_dist - distance_slack)
        self.far = self.want_dist + distance_slack
        # Turn only once the mob is within this of the frame edge.
        self.keep_half = max(math.radians(5.0), self.fov / 2.0 - math.radians(frame_margin_deg))
        self.target_slot: int | None = None
        self._strafe_for = 0
        self._strafe_idx = None

        self.a_backward = self._idx0_safe("backward", 2)
        self.a_left = self._idx0_safe("left", 3)
        self.a_right = self._idx0_safe("right", 4)

    def _idx0_safe(self, name, default):
        try:
            return self._actions0.index(name) + 1
        except Exception:
            return default

    # ------------------------------------------------------------------ target
    def _pick_target(self, agents):
        present = [a.get("slot") for a in agents if a.get("present", 0) == 1]
        present = [s for s in present if s is not None]
        if present:
            self.target_slot = self.rng.choice(present)
        return self.target_slot

    def _find(self, agents):
        for a in agents:
            if a.get("slot") == self.target_slot and a.get("present", 0) == 1:
                return a
        return None

    # --------------------------------------------------------------------- act
    def act(self):
        import numpy as np

        action = np.zeros(self.num_groups, dtype=np.int64)
        rec = self._read_latest_frame()
        if rec is None:
            return action                      # no data yet: stand still

        p = rec["player_pos"]
        px, pz = float(p["x"]), float(p["z"])
        yaw = float(rec.get("player", {}).get("yaw", 0.0))
        pitch_rot = rec.get("player", {}).get("rotation")
        pitch = float(pitch_rot.get("x", 0.0)) if isinstance(pitch_rot, dict) else 0.0
        self._calibrate(yaw, pitch)

        agents = rec.get("agents", [])
        if self.target_slot is None:
            self._pick_target(agents)
        mob = self._find(agents)
        if mob is None:                        # target gone: adopt another one
            self._pick_target(agents)
            mob = self._find(agents)
        if mob is None:
            return action

        ap = mob.get("pos", {})
        dx, dz = float(ap.get("x", 0.0)) - px, float(ap.get("z", 0.0)) - pz
        dist = math.hypot(dx, dz)
        yaw_err = _wrap(self._desired_yaw(dx, dz) - yaw)

        # --- framing: act only when it is about to leave the view ---------------
        if abs(yaw_err) > self.keep_half:
            action[2] = self._turn_action(yaw_err > 0.0)

        # --- distance: keep it legible, never lock on ---------------------------
        if dist > self.far:
            action[0] = self.a_forward
            self._strafe_for = 0
        elif dist < self.near:
            action[0] = self.a_backward
            self._strafe_for = 0
        else:
            # comfortable: drift sideways in bouts so the viewing angle keeps changing
            if self._strafe_for <= 0:
                self._strafe_for = self.rng.randint(8, 24)
                self._strafe_idx = self.rng.choice(
                    [self.a_left, self.a_right, self.a_forward, 0, 0])
            self._strafe_for -= 1
            if self._strafe_idx:
                action[0] = self._strafe_idx
        return action

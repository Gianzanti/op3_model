import os
from typing import Dict, Tuple, Union

import numpy as np
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from gymnasium.utils import EzPickle

DEFAULT_CAMERA_CONFIG = {"trackbodyid": 0}


def mass_center(model, data):
    mass = np.expand_dims(model.body_mass, axis=1)
    xpos = data.xipos
    com = np.sum(mass * xpos, axis=0) / np.sum(mass)
    return com[0:3].copy()


information = {
    "_pos_x": "px",
    "_pos_y": "py",
    "_pos_z": "pz",
    "_vel_x": "vx",
    "_vel_y": "vy",
    "_vel_z": "vz",
    "distance": "dfo",
    "r_health": "rh",
    "r_forward": "rf",
    "r_knee_flex": "rkf",
    "r_feet_up": "rfu",
    "p_not_parallel": "pnp",
    "p_control": "pc",
}


class DarwinOp3Env(MujocoEnv, EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }

    def __init__(
        self,
        frame_skip: int = 5,
        default_camera_config: Dict[str, Union[float, int]] = DEFAULT_CAMERA_CONFIG,
        healthy_z_range: Tuple[float, float] = (0.270, 0.300),  # expected z: 0.285
        keep_alive_reward: float = 1.0,
        ctrl_cost_weight: float = 1e-3,
        target_distance: float = 100.0,
        forward_velocity_weight: float = 3.0,
        reach_target_reward: float = 100.0,
        knee_flex_reward: float = 1e-3,
        feet_up_reward: float = 1e-3,
        not_parallel_penalty: float = 0.05,
        motor_max_torque: float = 3.0,
        reset_noise_scale: float = 1e-2,
        **kwargs,
    ):
        EzPickle.__init__(
            self,
            frame_skip,
            default_camera_config,
            keep_alive_reward,
            healthy_z_range,
            ctrl_cost_weight,
            forward_velocity_weight,
            target_distance,
            reach_target_reward,
            knee_flex_reward,
            feet_up_reward,
            not_parallel_penalty,
            motor_max_torque,
            reset_noise_scale,
            **kwargs,
        )

        xml_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "model", "scene.xml"
        )

        self._keep_alive_reward: float = keep_alive_reward
        self._healthy_z_range: Tuple[float, float] = healthy_z_range
        self._ctrl_cost_weight: float = ctrl_cost_weight
        self._fw_vel_rew_weight: float = forward_velocity_weight
        self._target_distance: float = target_distance
        self._reach_target_reward: float = reach_target_reward
        self._knee_flex_reward: float = knee_flex_reward
        self._feet_up_reward: float = feet_up_reward
        self._not_parallel_penalty: float = not_parallel_penalty
        self._motor_max_torque = motor_max_torque
        self._reset_noise_scale: float = reset_noise_scale

        MujocoEnv.__init__(
            self,
            xml_path,
            frame_skip,
            observation_space=None,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

        self.action_space = Box(
            low=-1, high=1, shape=self.action_space.shape, dtype=np.float32
        )

        obs_size = self.data.qpos[2:].size + self.data.qvel[2:].size
        obs_size += self.data.sensordata.size

        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

    def _get_obs(self):
        position = self.data.qpos[2:].flatten()
        velocity = self.data.qvel[2:].flatten()
        gyro = self.data.sensordata[0:3].flatten()
        acc = self.data.sensordata[3:6].flatten()
        mag = self.data.sensordata[6:9].flatten()

        return np.concatenate((position, velocity, gyro, acc, mag))

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )

        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )

        self.set_state(qpos, qvel)

        return self._get_obs()

    # @property
    def is_healthy(self, z_pos) -> bool:
        min_z, max_z = self._healthy_z_range
        return min_z < z_pos < max_z

    def _get_rew(self, velocity, position_before, position_after):
        [x_velocity, y_velocity, z_velocity] = velocity
        health_reward = self._keep_alive_reward * self.is_healthy(position_after[2])
        control_cost = self._ctrl_cost_weight * np.sum(np.square(self.data.ctrl))
        forward_reward = self._fw_vel_rew_weight * x_velocity
        knee_flex_reward = self._knee_flex_reward * (
            abs(self.data.qvel[10]) + abs(self.data.qvel[16])
        )
        feet_up_reward = self._feet_up_reward * (
            self.data.geom_xpos[32][2] + self.data.geom_xpos[44][2]
        )
        not_parallel_penalty = (
            self._not_parallel_penalty * self.check_not_parallel_penalty()
        )

        reward = (
            health_reward
            + forward_reward
            + knee_flex_reward
            + feet_up_reward
            - not_parallel_penalty
            - control_cost
        )

        if self.data.qpos[0] >= self._target_distance:
            health_reward = 0
            forward_reward = 0
            control_cost = 0
            knee_flex_reward = 0
            feet_up_reward = 0
            not_parallel_penalty = 0
            reward = self._reach_target_reward

        reward_info = {
            information["r_health"]: health_reward,
            information["r_forward"]: forward_reward,
            information["p_control"]: control_cost,
            information["r_knee_flex"]: knee_flex_reward,
            information["r_feet_up"]: feet_up_reward,
            information["p_not_parallel"]: not_parallel_penalty,
        }

        return reward, reward_info

    def termination(self, z_pos):
        if not self.is_healthy(z_pos):
            return True

        if self.data.qpos[0] >= self._target_distance:
            return True

        return False

    def step(self, normalized_action):
        # print("step")
        # get the current position of the robot, before action
        # position_before = mass_center(self.model, self.data)
        position_before = mass_center(self.model, self.data)

        # denormalize the action to the range of the motors
        action = normalized_action * self._motor_max_torque
        self.do_simulation(action, self.frame_skip)
        position_after = mass_center(self.model, self.data)

        velocity = (position_after - position_before) / self.dt
        distance_from_origin = np.linalg.norm(self.data.qpos[0:2], ord=2)

        observation = self._get_obs()
        reward, reward_info = self._get_rew(velocity, position_before, position_after)

        info = {
            information["_pos_x"]: position_after[0],
            information["_pos_y"]: position_after[1],
            information["_pos_z"]: position_after[2],
            information["_vel_x"]: velocity[0],
            information["_vel_y"]: velocity[1],
            information["_vel_z"]: velocity[2],
            information["distance"]: distance_from_origin,
            **reward_info,
        }

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit`
        # wrapper added during `make`
        return observation, reward, self.termination(position_after[2]), False, info

    def _check_contact(self, foot):
        """
        Checks if two bodies are in contact.

        Args:
            body_name1 (str): The name of the first body.
            body_name2 (str): The name of the second body.

        Returns:
            bool: True if the bodies are in contact, False otherwise.
        """
        if foot == "l_foot":
            geom_foot = 32
        elif foot == "r_foot":
            geom_foot = 44

        # Iterate through all contacts in the current simulation step.
        for i in range(self.data.ncon):
            contact = self.data.contact[i]

            if (contact.geom1 == 0 and contact.geom2 == geom_foot) or (
                contact.geom1 == geom_foot and contact.geom2 == 0
            ):
                # Contact detected between geom_id_A and geom_id_B
                # print(f"Contact detected between world and {foot}", contact.pos)
                return True

        # If the loop completes, no contact was found.
        return False

    def check_not_parallel_penalty(self):
        """
        Calculate a penalty based on the orientation of the robot's feet
        when they are in contact with the ground.

        The penalty is higher when the feet are less parallel to the ground
        (i.e., their local Z-axes do not align with the global Z-axis).

        Returns:
            float: The calculated not parallel penalty.
        """

        # 1. Get the rotation of the foot bodies
        left_foot_rot = self.data.geom("l_foot").xmat.reshape(3, 3)
        print("Left foot rotation matrix:\n", left_foot_rot)
        right_foot_rot = self.data.geom("r_foot").xmat.reshape(3, 3)
        print("Right foot rotation matrix:\n", right_foot_rot)

        # 2. Extract their local Z-axes (the "up" vector for each foot)
        left_foot_z_axis = left_foot_rot[:, 2]
        print("Left foot Z-axis:", left_foot_z_axis)
        right_foot_z_axis = right_foot_rot[:, 2]
        print("Right foot Z-axis:", right_foot_z_axis)

        # 3. Calculate the dot product with the global Z-axis (0, 0, 1)
        # Measure how aligned the foot's Z-axis is with the global Z-axis
        # A value of 1 means perfectly aligned (parallel), 0 means perpendicular
        # The dot product with (0,0,1) is just the z-component of the vector.
        # So we just get that directly.
        left_alignment = left_foot_z_axis[2]
        print("Left alignment:", left_alignment)
        right_alignment = right_foot_z_axis[2]
        print("Right alignment:", right_alignment)

        # 4. Check for contact with the world (the ground)
        # Note: 'world' is the default name for the root body in MuJoCo.
        is_left_foot_on_ground = self._check_contact("l_foot")
        # print("Left foot on ground:", is_left_foot_on_ground)
        is_right_foot_on_ground = self._check_contact("r_foot")
        # print("Right foot on ground:", is_right_foot_on_ground)

        # 4. Calculate the penalty based on alignment and contact
        # Only apply the penalty if the foot is in contact with the ground
        # The penalty is higher when the foot is less aligned (i.e., less parallel)
        # We use the alignment directly as the reward, so higher is better (more parallel)
        left_parallel_penalty = (1 - left_alignment) if is_left_foot_on_ground else 0
        print("Left parallel penalty:", left_parallel_penalty)
        right_parallel_penalty = (1 - right_alignment) if is_right_foot_on_ground else 0
        print("Right parallel penalty:", right_parallel_penalty)

        not_parallel_penalty = left_parallel_penalty + right_parallel_penalty
        print(f"Not Parallel Penalty: {not_parallel_penalty}")
        return not_parallel_penalty

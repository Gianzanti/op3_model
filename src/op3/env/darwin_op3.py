import os
from typing import Dict, Tuple, Union

import numpy as np
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from gymnasium.utils import EzPickle

DEFAULT_CAMERA_CONFIG = {"trackbodyid": 0}


def mass_center_position(model, data):
    mass = np.expand_dims(model.body_mass, axis=1)
    xpos = data.xipos
    com = np.sum(mass * xpos, axis=0) / np.sum(mass)
    return com[0:3].copy()


def mass_center_velocity(before, after, delta_time):
    return (after - before) / delta_time


information = {
    "_pos_x": "px",
    "_pos_y": "py",
    "_pos_z": "pz",
    "_vel_x": "vx",
    "_vel_y": "vy",
    "_vel_z": "vz",
    "info_dst_org": "dfo",
    "info_knee_angvel": "ikav",
    "info_feet_height": "ifh",
    "info_feet_misalignment": "ifm",
    "info_control": "ic",
    "r_health": "rh",
    "r_forward": "rf",
    "r_knee_flex": "rkf",
    "r_feet_up": "rfu",
    "p_feet_misalignment": "pfm",
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
        keep_alive_weight: float = 1.0,
        control_weight: float = 1e-3,
        target_distance: float = 100.0,
        velocity_weight: float = 3.0,
        reach_target_reward: float = 100.0,
        knee_flex_weight: float = 1e-3,
        feet_up_weight: float = 1e-3,
        feet_misalign_weight: float = 0.05,
        motor_max_torque: float = 3.0,
        reset_noise_scale: float = 1e-2,
        **kwargs,
    ):
        EzPickle.__init__(
            self,
            frame_skip,
            default_camera_config,
            keep_alive_weight,
            healthy_z_range,
            control_weight,
            velocity_weight,
            target_distance,
            reach_target_reward,
            knee_flex_weight,
            feet_up_weight,
            feet_misalign_weight,
            motor_max_torque,
            reset_noise_scale,
            **kwargs,
        )

        xml_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "model", "scene.xml"
        )

        self._keep_alive_weight: float = keep_alive_weight
        self._healthy_z_range: Tuple[float, float] = healthy_z_range
        self._control_weight: float = control_weight
        self._velocity_weight: float = velocity_weight
        self._target_distance: float = target_distance
        self._reach_target_reward: float = reach_target_reward
        self._knee_flex_weight: float = knee_flex_weight
        self._feet_up_weight: float = feet_up_weight
        self._feet_misalignment_weight: float = feet_misalign_weight
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

    def is_healthy(self, z_pos) -> bool:
        min_z, max_z = self._healthy_z_range
        return min_z < z_pos < max_z

    def control_effort(self):
        return np.sum(np.square(self.data.ctrl))

    def knees_angular_velocity(self):
        return abs(self.data.qvel[10]) + abs(self.data.qvel[16])

    def feet_height(self):
        return self.data.geom_xpos[32][2] + self.data.geom_xpos[44][2]

    def _get_rew(self, velocity, position_before, position_after):
        # [x_velocity, y_velocity, z_velocity] = velocity
        health_reward = self._keep_alive_weight * self.is_healthy(position_after[2])
        forward_reward = self._velocity_weight * velocity[0]
        info_knee_angvel = self.knees_angular_velocity()
        info_feet_height = self.feet_height()
        info_control = self.control_effort()
        info_feet_misalign = self.feets_ground_misalignment()

        knee_flex_reward = self._knee_flex_weight * info_knee_angvel
        feet_up_reward = self._feet_up_weight * info_feet_height
        control_penalty = self._control_weight * info_control
        feet_misalign_penalty = self._feet_misalignment_weight * info_feet_misalign

        reward = (
            health_reward
            + forward_reward
            + knee_flex_reward
            + feet_up_reward
            - feet_misalign_penalty
            - control_penalty
        )

        if self.data.qpos[0] >= self._target_distance:
            health_reward = 0
            forward_reward = 0
            knee_flex_reward = 0
            feet_up_reward = 0
            control_penalty = 0
            feet_misalign_penalty = 0
            reward = self._reach_target_reward

        reward_info = {
            information["info_dst_org"]: self.distance_from_origin(),
            information["info_knee_angvel"]: info_knee_angvel,
            information["info_feet_height"]: info_feet_height,
            information["info_control"]: info_control,
            information["info_feet_misalignment"]: info_feet_misalign,
            information["r_health"]: health_reward,
            information["r_forward"]: forward_reward,
            information["r_knee_flex"]: knee_flex_reward,
            information["r_feet_up"]: feet_up_reward,
            information["p_control"]: control_penalty,
            information["p_feet_misalignment"]: feet_misalign_penalty,
        }

        return reward, reward_info

    def termination(self, z_pos):
        if not self.is_healthy(z_pos):
            return True

        if self.data.qpos[0] >= self._target_distance:
            return True

        return False

    def distance_from_origin(self):
        return np.linalg.norm(self.data.qpos[0:2], ord=2)

    def step(self, normalized_action):
        # get the current position of the robot, before action
        mc_before = mass_center_position(self.model, self.data)

        # denormalize the action to the range of the motors
        action = normalized_action * self._motor_max_torque
        self.do_simulation(action, self.frame_skip)
        mc_after = mass_center_position(self.model, self.data)

        velocity = mass_center_velocity(mc_before, mc_after, self.dt * self.frame_skip)
        observation = self._get_obs()
        reward, reward_info = self._get_rew(velocity, mc_before, mc_after)

        info = {
            information["_pos_x"]: mc_after[0],
            information["_pos_y"]: mc_after[1],
            information["_pos_z"]: mc_after[2],
            information["_vel_x"]: velocity[0],
            information["_vel_y"]: velocity[1],
            information["_vel_z"]: velocity[2],
            **reward_info,
        }

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit`
        # wrapper added during `make`
        return observation, reward, self.termination(mc_after[2]), False, info

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

    def feets_ground_misalignment(self):
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
        right_foot_rot = self.data.geom("r_foot").xmat.reshape(3, 3)

        # 2. Extract their local Z-axes (the "up" vector for each foot)
        left_foot_z_axis = left_foot_rot[:, 2]
        right_foot_z_axis = right_foot_rot[:, 2]

        # 3. Calculate the dot product with the global Z-axis (0, 0, 1)
        # Measure how aligned the foot's Z-axis is with the global Z-axis
        # A value of 1 means perfectly aligned (parallel), 0 means perpendicular
        # The dot product with (0,0,1) is just the z-component of the vector.
        # So we just get that directly.
        left_alignment = left_foot_z_axis[2]
        right_alignment = right_foot_z_axis[2]

        # 4. Check for contact with the world (the ground)
        # Note: 'world' is the default name for the root body in MuJoCo.
        is_left_foot_on_ground = self._check_contact("l_foot")
        is_right_foot_on_ground = self._check_contact("r_foot")

        # 4. Calculate the penalty based on alignment and contact
        # Only apply the penalty if the foot is in contact with the ground
        # The penalty is higher when the foot is less aligned (i.e., less parallel)
        # We use the alignment directly as the reward, so higher is better (more parallel)
        left_misalignment = (1 - left_alignment) if is_left_foot_on_ground else 0
        right_misalignment = (1 - right_alignment) if is_right_foot_on_ground else 0

        return left_misalignment + right_misalignment

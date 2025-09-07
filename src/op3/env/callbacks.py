import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from op3.env.darwin_op3 import information


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        # self.episode_info = {}

    # def _init_callback(self) -> None:
    #     """
    #     This method is called once when the callback is initialized.
    #     """
    #     print("001 - Callback Initialized")
    #     print("N Calls:", self.n_calls)
    #     print("Num Timesteps:", self.num_timesteps)

    def reset_episode_info(self):
        self.episode_info = {}
        for key in information.keys():
            self.episode_info[key] = []

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        # print("002 - Training Started")
        # print("N Calls:", self.n_calls)
        # print("Num Timesteps:", self.num_timesteps)

        self.reset_episode_info()

        # reset episode info
        # for key in information.keys():
        #     self.episode_info[key] = []

        # self.episode_info = {
        #     information["position"]["body"]["x"]: [],
        #     information["position"]["body"]["y"]: [],
        #     information["position"]["body"]["z"]: [],
        #     information["position"]["mc"]["x"]: [],
        #     information["position"]["mc"]["y"]: [],
        #     information["position"]["mc"]["z"]: [],
        #     information["velocity"]["body"]["x"]: [],
        #     information["velocity"]["body"]["y"]: [],
        #     information["velocity"]["body"]["z"]: [],
        #     information["velocity"]["mc"]["x"]: [],
        #     information["velocity"]["mc"]["y"]: [],
        #     information["velocity"]["mc"]["z"]: [],
        #     information["xtras"]["orientation"]: [],
        #     information["xtras"]["distance"]: [],
        #     information["rewards"]["health"]: [],
        #     information["rewards"]["forward"]: [],
        #     information["penalties"]["control"]: [],
        # }

    # def _on_training_end(self) -> None:
    #     """
    #     This method is called after training is finished.
    #     """
    #     print("003 - Training Ended")
    #     print("N Calls:", self.n_calls)
    #     print("Num Timesteps:", self.num_timesteps)

    # def _on_rollout_start(self) -> None:
    #     """
    #     This event is triggered before collecting new samples.
    #     """
    #     print("004 - Rollout Started")
    #     print("N Calls:", self.n_calls)
    #     print("Num Timesteps:", self.num_timesteps)

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: If the callback returns False, training is aborted early.
        """
        # print("N Calls:", self.n_calls)
        # print("Num Timesteps:", self.num_timesteps)

        for env_idx in range(self.training_env.num_envs):
            info = self.locals["infos"][env_idx]

            for key, value in information.items():
                if value in info:
                    self.episode_info[key].append(info[value])

            # self.episode_info["x_positions"].append(info["x_position"])
            # self.episode_info["y_positions"].append(info["y_position"])
            # self.episode_info["z_positions"].append(info["z_position"])
            # self.episode_info["x_velocities"].append(info["x_velocity"])
            # self.episode_info["y_velocities"].append(info["y_velocity"])
            # self.episode_info["health_rewards"].append(info["health_reward"])
            # self.episode_info["control_costs"].append(info["control_cost"])
            # self.episode_info["forward_rewards"].append(info["forward_reward"])
            # self.episode_positions['pos_deviation_costs'].append(info['pos_deviation_cost'])
            # self.episode_positions['lateral_velocity_costs'].append(info['lateral_velocity_cost'])
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        # print("005 - Rollout Ended")

        # print("N Calls:", self.n_calls)
        # print("Num Timesteps:", self.num_timesteps)

        if self.episode_info:
            for key, values in self.episode_info.items():
                if values:
                    self.logger.record(f"mean_episode/{key}", np.mean(np.array(values)))

            # x_values = np.array(self.episode_info["x_positions"])
            # self.logger.record("mean_episode/pos_x", np.mean(x_values))

            # y_values = np.array(self.episode_info["y_positions"])
            # self.logger.record("mean_episode/pos_y", np.mean(y_values))

            # z_values = np.array(self.episode_info["z_positions"])
            # self.logger.record("mean_episode/pos_z", np.mean(z_values))

            # x_vel_values = np.array(self.episode_info["x_velocities"])
            # self.logger.record("mean_episode/vel_x", np.mean(x_vel_values))

            # y_vel_values = np.array(self.episode_info["y_velocities"])
            # self.logger.record("mean_episode/vel_y", np.mean(y_vel_values))

            # health_values = np.array(self.episode_info["health_rewards"])
            # self.logger.record("mean_episode/health_reward", np.mean(health_values))

            # control_costs = np.array(self.episode_info["control_costs"])
            # self.logger.record("mean_episode/control_cost", np.mean(control_costs))

            # forward_values = np.array(self.episode_info["forward_rewards"])
            # self.logger.record("mean_episode/forward_reward", np.mean(forward_values))

            # pos_deviation_costs = np.array(self.episode_positions['pos_deviation_costs'])
            # self.logger.record('mean_episode/pos_deviation_cost', np.mean(pos_deviation_costs))

            # lateral_velocity_costs = np.array(self.episode_positions['lateral_velocity_costs'])
            # self.logger.record('mean_episode/lateral_velocity_cost', np.mean(lateral_velocity_costs))

        # self.episode_info = {
        #     "x_positions": [],
        #     "y_positions": [],
        #     "z_positions": [],
        #     "x_velocities": [],
        #     "y_velocities": [],
        #     "health_rewards": [],
        #     "control_costs": [],
        #     "forward_rewards": [],
        #     # "pos_deviation_costs": [],
        #     # "lateral_velocity_costs": [],
        # }
        self.reset_episode_info()

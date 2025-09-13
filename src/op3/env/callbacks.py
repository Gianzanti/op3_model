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
        self.reset_episode_info()

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
        # if self.locals["infos"]["TimeLimit.truncated"]:
        #     print("Episode truncated @ top")

        for env_idx in range(self.training_env.num_envs):
            info = self.locals["infos"][env_idx]
            for key, value in information.items():
                if value in info:
                    self.episode_info[key].append(info[value])

        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """

        if self.episode_info:
            for key, values in self.episode_info.items():
                if values:
                    self.logger.record(f"mean_episode/{key}", np.mean(np.array(values)))

        self.reset_episode_info()

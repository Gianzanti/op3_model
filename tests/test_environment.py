import unittest

import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm
from op3.env.callbacks import TensorboardCallback
from op3.env.darwin_op3 import DarwinOp3Env
from op3 import exported_env


def evaluate(
    model: BaseAlgorithm,
    num_episodes: int = 100,
    deterministic: bool = False,
) -> float:
    """
    Evaluate an RL agent for `num_episodes`.

    :param model: the RL Agent
    :param env: the gym Environment
    :param num_episodes: number of episodes to evaluate it
    :param deterministic: Whether to use deterministic or stochastic actions
    :return: Mean reward for the last `num_episodes`
    """
    # This function will only work for a single environment
    vec_env = model.get_env()
    obs = vec_env.reset()
    all_episode_rewards = []
    for _ in range(num_episodes):
        episode_rewards = []
        done = False
        # Note: SB3 VecEnv resets automatically:
        # https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#vecenv-api-vs-gym-api
        # obs = vec_env.reset()
        while not done:
            # _states are only useful when using LSTM policies
            # `deterministic` is to use deterministic actions
            action, _states = model.predict(obs, deterministic=deterministic)
            # here, action, rewards and dones are arrays
            # because we are using vectorized env
            obs, reward, done, _info = vec_env.step(action)
            episode_rewards.append(reward)

        all_episode_rewards.append(sum(episode_rewards))

    mean_episode_reward = np.mean(all_episode_rewards)
    print(f"Mean reward: {mean_episode_reward:.2f} - Num episodes: {num_episodes}")


class DarwinOp3_TestEnv(unittest.TestCase):
    def test_checkenv(self):
        import gymnasium as gym

        from gymnasium.utils.env_checker import check_env
        env = gym.make(exported_env, render_mode="rgb_array", width=1920, height=1080)
        check_env(env.unwrapped)

        from stable_baselines3.common.env_checker import check_env
        env = gym.make(exported_env, render_mode="rgb_array", width=1920, height=1080)
        check_env(env.unwrapped)
        
    def test_rewards(self):
        import gymnasium as gym
        env = gym.make(exported_env, render_mode="rgb_array", width=1920, height=1080)

        observation, info = env.reset()
        # print(f"Observation space: {env.observation_space}")
        # print(f"Action space: {env.action_space}")
        # print(f"Info: {info}")
        # print(f"Action sample: {env.action_space.sample()}")

        episode_over = False
        counter = 0
        action = np.zeros(env.action_space.shape[0])

        while not episode_over:
            observation, reward, terminated, truncated, info = env.step(action)
            print(f"Observation: {observation}")
            print(f"Reward: {reward}")
            # print(f"Terminated: {terminated}")
            # print(f"Truncated: {truncated}")
            print(f"Info: {info}")
            episode_over = counter > 3000 or terminated
            counter += 1

        env.close()

        print("Rewards check successful!")

    def test_(self):
        import gymnasium as gym
        from stable_baselines3 import PPO
        from stable_baselines3.ppo.policies import MlpPolicy

        env = gym.make(exported_env, render_mode="rgb_array", width=1920, height=1080)
        model = PPO(MlpPolicy, env, verbose=1)
        mean_reward_before_train = evaluate(model, num_episodes=100)

    def test_actions(self):
        import gymnasium as gym
        env = gym.make(exported_env, render_mode="rgb_array", width=1920, height=1080)

        observation, info = env.reset()

        episode_over = False
        counter = 0

        # joint_ranges = {
        #     "l_sho_pitch": {"min": -3.14, "max": 3.14, "range": 6.28},
        #     "l_sho_roll": {"min": -0.60, "max": 1.90, "range": 2.50},
        #     "l_el": {"min": -3.00, "max": 0.50, "range": 3.50},
        #     "r_sho_pitch": {"min": -3.14, "max": 3.14, "range": 6.28},
        #     "r_sho_roll": {"min": -1.90, "max": 0.60, "range": 2.50},
        #     "r_el": {"min": -0.50, "max": 3.00, "range": 3.50},
        #     "l_hip_yaw": {"min": -0.3, "max": 0.3, "range": 0.6},
        #     "l_hip_roll": {"min": -0.3, "max": 0.3, "range": 0.6},
        #     "l_hip_pitch": {"min": -2.00, "max": 1.0, "range": 3.0},
        #     "l_knee": {"min": -0.08, "max": 3.00, "range": 3.08},
        #     "l_ank_pitch": {"min": -0.80, "max": 0.80, "range": 1.60},
        #     "l_ank_roll": {"min": -0.80, "max": 0.80, "range": 1.60},
        #     "r_hip_yaw": {"min": -0.3, "max": 0.3, "range": 0.6},
        #     "r_hip_roll": {"min": -0.3, "max": 0.3, "range": 0.6},
        #     "r_hip_pitch": {"min": -1.0, "max": 2.00, "range": 3.0},
        #     "r_knee": {"min": -3.00, "max": 0.08, "range": 3.08},
        #     "r_ank_pitch": {"min": -0.80, "max": 0.80, "range": 1.60},
        #     "r_ank_roll": {"min": -0.80, "max": 0.80, "range": 1.60},
        # }

        # find the 0 value for each normalized joint
        # joint_zeros = []
        # for joint in joint_ranges:
        #     zero_point = 0.00 - joint_ranges[joint]["min"]
        #     joint_zeros.append(zero_point / joint_ranges[joint]["range"])
        # for i, joint_range in enumerate(self.joint_ranges.values()):
        #     action[i] = normalized_action[i] * (joint_range["range"]/2)
        action = np.zeros(env.action_space.shape[0])

        # action = np.array(joint_zeros)

        target_left_shoulder = 0.80
        target_right_shoulder = -0.80

        target_left_hip = -0.45
        target_left_knee = 0.75
        target_left_ankle = 0.25

        target_right_hip = 0.45
        target_right_knee = -0.75
        target_right_ankle = -0.25
        
        velocity = 0.001
        
        while not episode_over:
            observation, reward, terminated, truncated, info = env.step(action)

            if observation[6] < target_left_shoulder: # 1.30
                action[1] += velocity

            if observation[9] > target_right_shoulder: # -1.30
                action[4] -= velocity

            if observation[13] > target_left_hip: # -0.25
                action[8] -= velocity
            
            if observation[14] < target_left_knee: # 0.50
                action[9] += velocity

            if observation[15] < target_left_ankle: # 0.25
                action[10] += velocity

            if observation[19] < target_right_hip: # 0.25
                action[14] += velocity
            
            if observation[20] > target_right_knee: # -0.50
                action[15] -= velocity

            if observation[21] > target_right_ankle: # -0.25
                action[16] -= velocity
           
            # print(f"Observation: {observation}")
            print(f"Reward: {reward}")
            # print(f"dt: {env.spec}")
            # print(f"Terminated: {terminated}")
            # print(f"Truncated: {truncated}")
            print(f"Info: {info}")
            episode_over = counter > 1000
            counter += 1

        env.close()

        print("Environment check successful!")

    def test_callback(self):
        import gymnasium as gym
        from stable_baselines3 import PPO
        from stable_baselines3.ppo.policies import MlpPolicy

        # env = gym.make(exported_env, render_mode="rgb_array", width=1920, height=1080)
        env = gym.make(exported_env)
        model = PPO(MlpPolicy, env, verbose=1, device="cpu", n_steps=2048, batch_size=64, n_epochs=10)
        # mean_reward_before_train = evaluate(model, num_episodes=100)

        model.learn(total_timesteps=10000, callback=TensorboardCallback())
        # mean_reward_after_train = evaluate(model, num_episodes=100)

        # assert mean_reward_after_train > mean_reward_before_train, "Callback failed to improve the mean reward"

    def test_multi_env(self):
        # import gymnasium as gym
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env

        # from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
        from stable_baselines3.ppo.policies import MlpPolicy

        env = make_vec_env(exported_env, n_envs=4)
        model = PPO(MlpPolicy, env, verbose=1, device="cpu", n_steps=2048, batch_size=64, n_epochs=10)

        model.learn(total_timesteps=10000, callback=TensorboardCallback())
        # mean_reward_after_train = evaluate(model, num_episodes=100)

        # assert mean_reward_after_train > mean_reward_before_train, "Callback failed to improve the mean reward"

    def make_env(self):
        import gymnasium as gym
        env = gym.make(exported_env)
        env = gym.wrappers.TimeLimit(env, 5)  # new time limit
        return env

    def test_time_limit(self):
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.ppo.policies import MlpPolicy
        import gymnasium as gym

        env = make_vec_env(self.make_env, n_envs=4)
        model = PPO(MlpPolicy, env, verbose=1, device="cpu", n_steps=2048, batch_size=64, n_epochs=10)

        model.learn(total_timesteps=10000, callback=TensorboardCallback())
     
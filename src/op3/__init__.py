from importlib.metadata import version

from gymnasium.envs.registration import find_highest_version, register

env_name = "DarwinOp3"
env = f"{env_name}-v3"
env_id = find_highest_version(ns=None, name=env_name)

if env_id is None:
    register(
        id=env,
        entry_point="robofei.env:DarwinOp3Env",
        nondeterministic=True,
    )
    print(f"Registered environment {env}")

print(f"DarwinOp3 Env version: {version('robofei')}")

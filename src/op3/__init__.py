from importlib.metadata import version

from gymnasium.envs.registration import find_highest_version, register

env_name = "DarwinOp3"
exported_env = f"{env_name}-v3"
env_id = find_highest_version(ns=None, name=env_name)

if env_id is None:
    register(
        id=exported_env,
        entry_point="op3.env:DarwinOp3Env",
        nondeterministic=True,
    )
    print(f"Registered environment {exported_env}")

print(f"{exported_env} Env version: {version('op3')}")

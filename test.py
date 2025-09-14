from env.robomimic.robomimic_env import RobomimicEnv


env = RobomimicEnv(env_name="can-ph")
env.reset()
env.render()
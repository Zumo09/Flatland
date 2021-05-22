from flatland.envs.observations import TreeObsForRailEnv, LocalObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import complex_rail_generator
from flatland.envs.schedule_generators import complex_schedule_generator
from flatland.utils.rendertools import RenderTool

from simple_DQN import VanillaDQN, obs_to_tensor


TreeObservation = TreeObsForRailEnv(max_depth=2, predictor=ShortestPathPredictorForRailEnv())
LocalGridObs = LocalObsForRailEnv(view_height=10, view_width=2, center=2)
env = RailEnv(width=20, height=20,
              rail_generator=complex_rail_generator(nr_start_goal=10,
                                                    nr_extra=2,
                                                    min_dist=8,
                                                    max_dist=99999,
                                                    seed=1),
              schedule_generator=complex_schedule_generator(),
              number_of_agents=1
              #obs_builder_object=TreeObservation
              )

obs, _ = env.reset()

input_ = obs_to_tensor(obs)
print(input_.shape)

env_renderer = RenderTool(env)

# Initialize the agent with the parameters corresponding to the environment and observation_builder
controller = VanillaDQN(input_.shape[1:], 5)
n_trials = 5
steps = 100

# Empty dictionary for all agent action
print("Starting Training...")

for trials in range(1, n_trials + 1):
    # Reset environment and get initial observations for all agents
    obs, info = env.reset()

    for idx in range(env.get_num_agents()):
        tmp_agent = env.agents[idx]
        tmp_agent.speed_data["speed"] = 1 / (idx + 1)
    env_renderer.reset()
    # Here you can also further enhance the provided observation by means of normalization
    # See training navigation example in the baseline repository

    score = 0
    # Run episode
    for step in range(steps):
        # Chose an action for each agent in the environment
        actions = controller.get_actions(observations=obs)

        # Environment step which returns the observations for all agents, their corresponding
        # reward and whether their are done
        next_obs, all_rewards, done, _ = env.step(actions)

        controller.save_history(next_obs, all_rewards, done)

        env_renderer.render_env(show=True, show_observations=False, show_predictions=False)

        # Update replay buffer and train agent
        controller.update()

        obs = next_obs.copy()

        if done['__all__']:
            break

        print(f'\rStep {step}: ', end='')

    if controller.end_episode(score):
        break
    print('Episode Nr. {}\t Score = {}'.format(trials, score))

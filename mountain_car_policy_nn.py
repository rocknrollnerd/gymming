import gym
import optparse
from lasagne import nonlinearities
import numpy as np

from algo.policy_nn import NNGaussianGradientPolicy

gamma = 0.95  # discount factor for reward
policy_variance = 1  # fixed variance for gaussian policy


def discount_rewards(r):
    return r


# def discount_rewards(r):
#     discounted_r = np.zeros_like(r)
#     running_sum = 0
#     discount = 1
#     for i in xrange(len(r)):
#         running_sum += discount * r[i]
#         discount *= gamma
#         discounted_r[i] = running_sum
#     # discounted_r = discounted_r[::-1]
#     discounted_r -= np.mean(discounted_r)
#     discounted_r /= np.std(discounted_r)
#     return discounted_r


def parse_options():
    parser = optparse.OptionParser()
    parser.add_option('-e', '--episodes',
                      dest="n_episodes",
                      type=int,
                      default=1000
                      )
    parser.add_option('-s', '--steps',
                      dest="n_steps",
                      type=int,
                      default=500
                      )
    options, remainder = parser.parse_args()
    return options


def run():
    options = parse_options()
    env = gym.make('MountainCarContinuous-v0')

    policy = NNGaussianGradientPolicy(input_size=2, network_schema=(
        {'n_neurons': 20, 'nonlinearity': nonlinearities.leaky_rectify},
        {'n_neurons': 10, 'nonlinearity': nonlinearities.leaky_rectify},
        {'n_neurons': 1, 'nonlinearity': nonlinearities.linear}
    ), learning_rate=0.01)

    solved = False
    total_rewards = []
    threshold = 500

    max_position = 0.6

    for i_episode in range(options.n_episodes):
        observations = []
        actions = []
        rewards = []

        observation = env.reset()
        for t in range(options.n_steps):
            # if solved:
            if solved or i_episode > threshold:
                env.render()

            action_proba = policy.act(observation)
            action = [np.random.normal(loc=action_proba, scale=policy_variance)]
            old_observation = observation
            observation, reward, done, info = env.step(action)

            # reward: (is position == max_position then 100 else 0) - sum(all actions taken from the beginning)
            # this is never positive and at first encourages to do nothing, so we'll try to make a finctional reward
            if observation[0] != max_position:
                reward = (observation[0] / max_position) #* 100 + reward
            # print reward, max(observation[0], 0)
            # raw_input()

            observations.append(old_observation)
            actions.append(action[0])
            rewards.append(reward)
            if i_episode > threshold:
                print reward, observation[0]
            if done:
                break
        if i_episode > threshold:
            raw_input()
        total_rewards.append(np.sum(rewards))
        avg_reward = np.mean(total_rewards[-100:])
        print 'episode', i_episode, 'steps', t, 'avg reward', avg_reward, 'episode reward', total_rewards[-1]
        # if avg_reward >= 90.0 and not solved:
        #     print 'SOLVED! :-)'
        #     solved = True
            # break
        policy.update(np.vstack(observations), np.vstack(actions), discount_rewards(np.array(rewards)))

    if not solved:
        print 'FAILED :-(', 'avg reward', avg_reward


if __name__ == "__main__":
    run()

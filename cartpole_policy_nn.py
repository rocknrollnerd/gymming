import gym
import optparse
from lasagne import nonlinearities
import numpy as np

from algo.policy_nn import NeuralNetworkGradientPolicy

gamma = 0.95  # discount factor for reward


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


def discount_rewards(r):
    discounted_r = np.zeros_like(r)
    running_sum = 0
    discount = 1
    for i in xrange(len(r)):
        running_sum += discount * r[i]
        discount *= gamma
        discounted_r[i] = running_sum
    discounted_r = discounted_r[::-1]
    discounted_r -= np.mean(discounted_r)
    discounted_r /= np.std(discounted_r)
    return discounted_r


def run():
    options = parse_options()
    env = gym.make('CartPole-v0')
    env.monitor.start('/tmp/cartpole-experiment-1', force=True)

    policy = NeuralNetworkGradientPolicy(input_size=4, network_schema=(
        {'n_neurons': 20, 'nonlinearity': nonlinearities.leaky_rectify},
        {'n_neurons': 10, 'nonlinearity': nonlinearities.leaky_rectify},
        {'n_neurons': 2, 'nonlinearity': nonlinearities.softmax}
    ))

    solved = False
    total_rewards = []

    for i_episode in range(options.n_episodes):
        observations = []
        actions = []
        rewards = []

        observation = env.reset()
        for t in range(options.n_steps):
            # if solved:
            env.render()

            action_proba = policy.act(observation)
            action_blank = np.zeros(2)
            action = 0 if np.random.uniform() < action_proba[0] else 1
            action_blank[action] = 1
            old_observation = observation
            observation, reward, done, info = env.step(action)

            observations.append(old_observation)
            actions.append(action_blank)
            rewards.append(reward)
            if done:
                break

        total_rewards.append(np.sum(rewards))
        avg_reward = np.mean(total_rewards[-100:])
        print 'episode', i_episode, 'steps', t, 'avg reward', avg_reward
        if avg_reward >= 195.0 and not solved:
            print 'SOLVED! :-)'
            solved = True
            break
        policy.update(np.vstack(observations), np.vstack(actions), discount_rewards(np.array(rewards)))

    if not solved:
        print 'FAILED :-(', 'avg reward', avg_reward

    env.monitor.close()


if __name__ == "__main__":
    run()

from agent import StudentAgent
import tensorflow as tf
from env import CanvasEnvironment
import random
import utils as U


EP_START = 1.0
EP_END = 0.001
T_EP_END = 100000
T_TRAIN_START = 5000
MAX_T = 500000


def main():
    with tf.Session() as sess:
        env = CanvasEnvironment()
        agent = StudentAgent(sess=sess, t_train_start=T_TRAIN_START, env=env)

        _global_step = agent.global_step()

        terminal = True

        for _ in range(_global_step, MAX_T + T_TRAIN_START):
            global_step = agent.global_step()
            ep = U.calc_ep(global_step, EP_START, EP_END, T_EP_END)

            if terminal:
                rgoal = random.randint(0, 4)
                sub_goal, observation, _, _ = env.new_canvas(goal=rgoal)

            action = agent.predict(observation, sub_goal, ep)

            sub_goal, next_observation, reward, terminal = env.draw(action)
            agent.perceive(sub_goal, observation,action,reward,next_observation,terminal)

            observation = next_observation


        # Testing


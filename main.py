import argparse
import time
import os
import gym
from collections import deque
from CarRacingDQNAgent import CarRacingDQNAgent
from common_functions import process_state_image
from common_functions import generate_state_frame_stack_from_queue
import numpy as np
import math
import cv2


class Point():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def getx(self):
        return self.x

    def gety(self):
        return self.y


def GetCross(p1, p2, p):
    return (p2.x - p1.x) * (p.y - p1.y) - (p.x - p1.x) * (p2.y - p1.y)


class Getlen:
    def __init__(self, p1, p2):
        self.x = p1.getx() - p2.getx()
        self.y = p1.gety() - p2.gety()
        # use math.sqrt() to get the square root 用math.sqrt（）求平方根
        self.len = math.sqrt((self.x ** 2) + (self.y ** 2))

    # define the function of getting the length of line 定义得到直线长度的函数
    def getlen(self):
        return self.len


def IsPointInMatrix(p1, p2, p3, p4, p):
    isPointIn = GetCross(p1, p2, p) * GetCross(p3, p4, p) >= 0 and GetCross(p2, p3, p) * GetCross(p4, p1, p) >= 0
    return isPointIn


def getDis(p1, p2, p3, p4, p):
    # define the object 定义对象
    l1 = Getlen(p1, p2)
    l2 = Getlen(p1, p3)
    l3 = Getlen(p2, p3)
    # get the length of two points/获取两点之间直线的长度
    d1 = l1.getlen()
    d2 = l2.getlen()
    d3 = l3.getlen()


def isInTrack(position, trackList):
    x, y = position
    pp = Point(x, y)
    for i in range(len(trackList)):
        p1 = Point(trackList[i][0][0][0], trackList[i][0][0][1])
        p2 = Point(trackList[i][0][1][0], trackList[i][0][1][1])
        p3 = Point(trackList[i][0][2][0], trackList[i][0][2][1])
        p4 = Point(trackList[i][0][3][0], trackList[i][0][3][1])
        if IsPointInMatrix(p1, p2, p3, p4, pp):
            return True
    return False


if __name__ == '__main__':
    # labels=np.load("record/out2000/labels.npy")

    name = 0
    safes = []
    parser = argparse.ArgumentParser(description='Play CarRacing by the trained model.')
    parser.add_argument('-i', '--input', default='save/trial_200.h5', help='the path of saved model')
    parser.add_argument('-o', '--output', default='record/train', help='output path, dir\'s name')
    parser.add_argument('-n', '--numbers', type=int, default=100000, help='The number of episodes should the model plays.')
    args = parser.parse_args()

    train_model = ["save/trial_100.h5","save/trial_200.h5","save/trial_300.h5","save/trial_400.h5","save/trial_500.h5","save/trial_600.h5"]
    print(train_model)
    outdir = args.output

    if not os.path.exists(outdir + '/'):  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(outdir + '/')
    out2 = outdir+"/img/"
    if not os.path.exists(out2 + '/'):  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(out2 + '/')

    out3 = outdir+"/npz/"
    if not os.path.exists(out3 + '/'):  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(out3 + '/')
    play_numbers = args.numbers

    env = gym.make('CarRacing-v0')
    env.seed(999)
    np.random.seed(999)
    agent = CarRacingDQNAgent(epsilon=0)  # Set epsilon to 0 to ensure all actions are instructed by the agent
    serise=[]
    for i in range(len(train_model)):
        agent.load(train_model[i])
        guard=0
        safety=[]

        recording_obs = []
        recording_action = []
        recording_safe = []
        recording_position = []
        recording_map = []
        for t in range(15):



            init_state = env.reset()

            init_state = process_state_image(init_state)

            total_reward = 0
            punishment_counter = 0
            state_frame_stack_queue = deque([init_state] * agent.frame_stack_num, maxlen=agent.frame_stack_num)
            time_frame_counter = 1
            for j in range(50):
                env.render()
                current_state_frame_stack = generate_state_frame_stack_from_queue(state_frame_stack_queue)
                action = agent.act(current_state_frame_stack)
                next_state, reward, done, info = env.step(action)
                total_reward += reward
                next_state = process_state_image(next_state)
                state_frame_stack_queue.append(next_state)

            while True:
                env.render()

                current_state_frame_stack = generate_state_frame_stack_from_queue(state_frame_stack_queue)
                action = agent.act(current_state_frame_stack)

                recording_action.append(action)

                next_state, reward, done, info = env.step(action)
                cv2.imwrite(out2+ str(name) + ".jpg", next_state)
                name = name + 1
                guard=guard+1
                posx, posy = info[0]
                safety = isInTrack(info[0], info[1])
                print(safety)
                if safety:
                    safes.append(1)

                    recording_safe.append(1)
                else:
                    safes.append(0)
                    recording_safe.append(0)

                recording_obs.append(next_state)
                recording_position.append(info[0])
                total_reward += reward

                next_state = process_state_image(next_state)
                state_frame_stack_queue.append(next_state)

                if done:

                    serise.append(len(recording_obs))
                    recording_map = (info[1])
                    recording_obs = np.array(recording_obs, dtype=np.uint8)
                    recording_action = np.array(recording_action, dtype=np.float16)
                    recording_position=np.array(recording_position, dtype=np.float16)
                    recording_map=np.array(recording_map)
                    tmp=time.strftime("%Y%m%d%H%M%S", time.localtime())
                    np.savez_compressed(out3+"/"+tmp+".npz", obs=recording_obs, action=recording_action,safe=recording_safe,map=recording_map,position=recording_position)

                    recording_obs = []
                    recording_action = []
                    recording_safe = []
                    recording_position = []
                    recording_map = []
                    # print('Episode: {}/{}, Scores(Time Frames): {}, Total Rewards: {:.2}'.format(e + 1, play_episodes,
                    #                                                                              time_frame_counter,
                    #                                                                              float(total_reward)))
                    break
                time_frame_counter += 1

            print(recording_obs)
            print(recording_action)


        # print(tmp)
    np.save(outdir + "/labels.npy", safes)
    np.save(outdir + "/serise.npy", serise)

    # if len(recording_obs)==1000:
    # print("end")

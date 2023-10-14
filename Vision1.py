# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Header.Vision1
import gymnasium as gym
import numpy as np
import sys
from datetime import datetime
import argparse
import json
import brica1.brica_gym
import brical
import tensorflow as tf
import cv2


# Periphery vision to saliency:Generates the saliency map
class Periphery2Saliency(brica1.brica_gym.Component):
    def __init__(self, config):
        super().__init__()
        self.grid_size = config["env"]["grid_size"]
        self.stage_size = config["env"]["stage_size"]
        self.scene_size = self.stage_size * 2 - 1
        self.scene_image_size = self.scene_size * self.grid_size
        self.mid_img_size = self.scene_image_size // 2
        self.tf_log_writer = config['train']["tf_log_writer"]
        self.edge_detection = config['agent']['Periphery2Saliency']['edge_detection']
        self.weights = np.array(config['agent']['Periphery2Saliency']['weights'])
        self.make_in_port('observation', self.scene_image_size * self.scene_image_size * 3)
        self.make_in_port('token_in', 1)
        self.make_out_port('saliency_map', self.scene_size * self.scene_size)
        self.make_out_port('token_out', 1)
        self.prev_intensity_map = np.zeros((self.scene_size, self.scene_size), dtype=np.int8)

    def fire(self):
        img = self.get_in_port('observation').buffer
        # Gray-scale conversion
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        intensity_map = cv2.resize(img, dsize=(self.scene_size, self.scene_size), interpolation=cv2.INTER_LINEAR)
        # Reduce size (1/2)
        half_img = cv2.resize(img, dsize=(self.mid_img_size, self.mid_img_size), interpolation=cv2.INTER_LINEAR)
        log_images = np.reshape(half_img, (-1, self.mid_img_size, self.mid_img_size, 1))
        # Differential/Edge detection
        if self.edge_detection == "Laplacian":
            edges = np.abs(cv2.Laplacian(half_img, cv2.CV_32F))
        elif self.edge_detection == "Sobel":
            edges = cv2.Sobel(half_img, cv2.CV_32F, 1, 1, ksize=3)
        elif self.edge_detection == "Canny":
            edges = cv2.Canny(half_img, 100, 200)
        else:
            edges = half_img
        edges = np.clip(edges, 0, np.iinfo(np.uint8).max).astype(np.uint8)
        edge_map = cv2.resize(edges, dsize=(self.scene_size, self.scene_size), interpolation=cv2.INTER_LINEAR)
        # Time differential
        increment = np.clip(intensity_map - self.prev_intensity_map, 0, None)
        self.prev_intensity_map = intensity_map
        # Adding intensity map, edge_map, and increment
        weights = self.weights / np.sum(self.weights)
        img = (intensity_map * weights[0] + edge_map * weights[1] + increment * weights[2]).astype(np.uint8)
        if self.tf_log_writer is not None:
            # Dump images for TensorBoard
            dump_img = np.reshape(cv2.resize(img, dsize=(self.mid_img_size, self.mid_img_size),
                                                         interpolation=cv2.INTER_AREA),
                                              (-1, self.mid_img_size, self.mid_img_size, 1))
            log_images = np.append(log_images, dump_img, axis=0)
            with self.tf_log_writer.as_default():
                tf.summary.image("Training data", log_images, max_outputs=2, step=0)
        self.results['saliency_map'] = img

    def reset(self):
        self.token = 0
        self.inputs['token_in'] = np.array([0])
        self.results['token_out'] = np.array([0])
        self.get_in_port('token_in').buffer = self.inputs['token_in']
        self.get_out_port('token_out').buffer = self.results['token_out']
        self.prev_intensity_map = np.zeros((self.scene_size, self.scene_size), dtype=np.int8)


# Saliency to gaze control:Generates gaze control signals from the saliency map
class PriorityMap2Gaze(brica1.brica_gym.Component):
    def __init__(self, config):
        super().__init__()
        self.grid_size = config["env"]["grid_size"]
        self.stage_size = config["env"]["stage_size"]
        self.decrement_rate = config['agent']['PriorityMap2Gaze']['decrement_rate']
        self.threshold = config['agent']['PriorityMap2Gaze']['threshold']
        self.noise_max = config['agent']['PriorityMap2Gaze']['noise_max']
        self.scene_size = self.stage_size * 2 - 1
        self.scene_image_size = self.scene_size * self.grid_size
        self.make_in_port('saliency_map', self.scene_size * self.scene_size)
        self.make_in_port('token_in', 1)
        self.make_out_port('action', 2)
        self.make_out_port('token_out', 1)
        self.state = np.zeros((self.scene_size, self.scene_size), dtype=np.int8)

    def fire(self):
        sm = self.get_in_port('saliency_map').buffer
        sm = sm + np.random.randint(0, self.noise_max + 1, (self.scene_size, self.scene_size))
        self.state = sm + (self.state * self.decrement_rate).astype(int)
        am = np.argmax(self.state)
        saccade = np.zeros(2, dtype=np.int8)
        if np.max(sm) > self.threshold:
            center = self.scene_size // 2
            saccade = (am // self.scene_size - center, am % self.scene_size - center)
            self.state = np.zeros((self.scene_size, self.scene_size), dtype=np.int8)
        self.results['action'] = np.asarray(saccade)

    def reset(self):
        self.token = 0
        self.inputs['token_in'] = np.array([0])
        self.results['token_out'] = np.array([0])
        self.get_in_port('token_in').buffer = self.inputs['token_in']
        self.get_out_port('token_out').buffer = self.results['token_out']
        self.state = np.zeros((self.scene_size, self.scene_size), dtype=np.int8)


def main():
    parser = argparse.ArgumentParser(description='An agent with minimal active vision')
    parser.add_argument('--dump', help='dump file path')
    parser.add_argument('--episode_count', type=int, default=1, metavar='N',
                        help='Number of training episodes (default: 1)')
    parser.add_argument('--max_steps', type=int, default=30, metavar='N',
                        help='Max steps in an episode (default: 20)')
    parser.add_argument('--config', type=str, default='Vision1.json', metavar='N',
                        help='Model configuration (default: Vision1.json')
    parser.add_argument('--brical', type=str, default='Vision1.brical.json', metavar='N',
                        help='a BriCAL json file')
    args = parser.parse_args()

    with open(args.config) as config_file:
        config = json.load(config_file)

    tf_logidir = config['tf_logidir']
    # Sets up a timestamped log directory.
    logdir = tf_logidir + datetime.now().strftime("%Y%m%d-%H%M%S")
    # Creates a file writer for the log directory.
    tf_log_writer = tf.summary.create_file_writer(logdir)

    nb = brical.NetworkBuilder()
    f = open(args.brical)
    nb.load_file(f)
    if not nb.check_consistency():
        sys.stderr.write("ERROR: " + args.brical + " is not consistent!")
        exit(-1)

    if not nb.check_grounding():
        sys.stderr.write("ERROR: " + args.brical + " is not grounded!")
        exit(-1)

    train = {"episode_count": args.episode_count, "max_steps": args.max_steps, "tf_log_writer": tf_log_writer}
    config['train'] = train

    env = gym.make(config['env']['name'], config=config['env'], render_mode="human")

    nb.unit_dic['Vision1.Periphery2Saliency'].__init__(config)
    nb.unit_dic['Vision1.PriorityMap2Gaze'].__init__(config)
    nb.make_ports()

    agent_builder = brical.AgentBuilder()
    model = nb.unit_dic['Vision1.CognitiveArchitecture']
    model.make_in_port('reward', 1)
    model.make_in_port('done', 1)
    agent = agent_builder.create_gym_agent(nb, model, env)
    scheduler = brica1.VirtualTimeSyncScheduler(agent)

    for i in range(train["episode_count"]):
        last_token = 0
        print ('Cycle: ', i)
        '''
        if i == 0:
            nb.unit_dic['Vision1.PriorityMap2Gaze'].gaze = np.array([0, 0])
        elif i == 1:
            nb.unit_dic['Vision1.PriorityMap2Gaze'].gaze = np.array([2, 2])
        elif i == 2:
            nb.unit_dic['Vision1.PriorityMap2Gaze'].gaze = np.array([-2, -2])
        elif i == 3:
            nb.unit_dic['Vision1.PriorityMap2Gaze'].gaze = np.array([-2, 2])
        else:
            nb.unit_dic['Vision1.PriorityMap2Gaze'].gaze = np.array([2, -2])
        print("main", nb.unit_dic['Vision1.PriorityMap2Gaze'].gaze)
        '''
        for j in range(train["max_steps"]):
            scheduler.step()
            current_token = agent.get_out_port('token_out').buffer[0]
            if last_token + 1 == current_token:
                if last_token >= 1:
                    env.render()
                last_token = current_token
                # TODO: WRITE END OF ENV CYCLE CODE HERE!!
            if agent.env.done:
                break
        agent.env.flush = True
        nb.unit_dic['Vision1.Periphery2Saliency'].reset()
        nb.unit_dic['Vision1.PriorityMap2Gaze'].reset()
        # TODO: WRITE END OF EPISODE CODE (component reset etc.) HERE!!
        agent.env.reset()
        # agent.env.out_ports['token_out'] = np.array([0])
        agent.env.done = False

    # nb.unit_dic['Vision1.Periphery2Saliency'].close()
    # nb.unit_dic['Vision1.PriorityMap2Gaze'].close()
    print("Close")
    env.close()


if __name__ == '__main__':
    main()

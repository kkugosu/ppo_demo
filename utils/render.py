import torch
import random
import numpy as np
from NeuralNetwork import NN
from control import BASE, policy
from utils import converter
from control import policy
import time
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# self.converter = converter.Converter(self.envname)


class Render(BASE.BasePolicy):
    def __init__(self, p, *args):
        super().__init__(*args)
        self.policy = p

    def rend(self):
        n_p_o = self.env.reset()
        n_o = self.env.reset()
        t = 0
        local = 0
        color = np.random.randint(256, size=3)
        board = 1
        self.env.render(board, color, n_p_o, n_o)
        while t < 1000:
            board = 0
            n_a = self.policy.select_action(n_p_o)
            n_o, n_r, n_d, info = self.env.step(n_a)
            self.env.render(board, color, n_p_o, n_o)
            time.sleep(0.02)
            t = t + 1
            n_d = 0
            if t % 50 == 0:
                n_d = 1
            local = local + 1
            if n_d:
                board = 1
                n_o = self.env.reset()
                # self.env.render(board, color, n_o, n_o)
                print("Episode finished after {} timesteps".format(local+1))
                local = 0
            n_p_o = n_o



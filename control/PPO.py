from control import BASE, policy
import gym
import torch
import numpy as np
import sys
from torchvision.transforms import ToTensor, Lambda
from torch import nn
from NeuralNetwork import NN
from utils import buffer
from torch.distributions.categorical import Categorical
import random
import torch.onnx as onnx
GAMMA = 0.98


class PPOPolicy(BASE.BasePolicy):
    def __init__(self, *args) -> None:
        super().__init__(*args)
        self.updatedPG = NN.ProbNN(self.o_s, self.h_s, self.a_s).to(self.device)
        self.basePG = NN.ProbNN(self.o_s, self.h_s, self.a_s).to(self.device)
        self.updatedDQN = NN.ValueNN(self.o_s, self.h_s, self.a_s).to(self.device)
        self.baseDQN = NN.ValueNN(self.o_s, self.h_s, self.a_s).to(self.device)
        self.basePG.eval()
        self.baseDQN.eval()
        self.policy = policy.Policy(self.cont, self.updatedPG, self.env_n)
        self.buffer = buffer.Simulate(self.env, self.policy, step_size=self.e_trace)
        self.optimizer_p = torch.optim.SGD(self.updatedPG.parameters(), lr=self.lr/5)
        self.optimizer_q = torch.optim.SGD(self.updatedDQN.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss(reduction='mean')
        self.kl_loss = nn.KLDivLoss(reduction="mean")

    def get_policy(self):
        return self.policy

    def training(self, load=int(0)):

        if int(load) == 1:
            print("loading")
            self.updatedPG.load_state_dict(torch.load(self.PARAM_PATH + "/1.pth"))
            self.updatedDQN.load_state_dict(torch.load(self.PARAM_PATH + "/2.pth"))
            self.baseDQN.load_state_dict(self.updatedDQN.state_dict())
            self.basePG.load_state_dict(self.updatedPG.state_dict())
            self.baseDQN.eval()
            self.basePG.eval()
            print("loading complete")
        else:
            pass
        i = 0
        while i < self.t_i:
            print(i)

            i = i + 1
            self.buffer.renewal_memory(self.ca, self.data, self.dataloader)
            pg_loss, dqn_loss = self.train_per_buff()
            self.writer.add_scalar("pg/loss", pg_loss, i)
            self.writer.add_scalar("dqn/loss", dqn_loss, i)
            #torch.save(self.updatedPG.state_dict(), self.PARAM_PATH + "/1.pth")
            #torch.save(self.updatedDQN.state_dict(), self.PARAM_PATH + '/2.pth')
            self.baseDQN.load_state_dict(self.updatedDQN.state_dict())
            self.basePG.load_state_dict(self.updatedPG.state_dict())
            self.baseDQN.eval()
            self.basePG.eval()

        for param in self.updatedDQN.parameters():
            print("----------dqn-------------")
            print(param)
        for param in self.updatedPG.parameters():
            print("----------pg--------------")
            print(param)

        self.env.close()
        self.writer.flush()
        self.writer.close()

    def train_per_buff(self):
        i = 0
        dqn_loss = None
        pg_loss = None
        while i < self.m_i:
            # print(i)
            n_p_o, n_a, n_o, n_r, n_d = next(iter(self.dataloader))
            t_p_o = torch.tensor(n_p_o, dtype=torch.float32).to(self.device)
            t_a_index = self.converter.act2index(n_a, self.b_s).unsqueeze(axis=-1)
            t_o = torch.tensor(n_o, dtype=torch.float32).to(self.device)
            t_r = torch.tensor(n_r, dtype=torch.float32).to(self.device)
            t_p_weight = torch.gather(self.updatedPG(t_p_o), 1, t_a_index)
            t_p_base_weight = torch.gather(self.basePG(t_p_o), 1, t_a_index)
            t_p_weight_clip = torch.clamp(t_p_weight/t_p_base_weight, max=1.1)
            t_p_weight_clip = t_p_weight_clip*t_p_base_weight
            t_p_qvalue = torch.gather(self.updatedDQN(t_p_o), 1, t_a_index)
            weight = torch.transpose(t_p_weight_clip, 0, 1)

            entropy_bonus = -torch.sum(torch.log(self.updatedPG(t_p_o)))/self.b_s
            pg_loss = -torch.matmul(weight, t_p_qvalue) - entropy_bonus*0.1

            with torch.no_grad():
                n_a_expect = self.policy.select_action(n_o)
                t_a_index = self.converter.act2index(n_a_expect, self.b_s).unsqueeze(-1)
                t_qvalue = torch.gather(self.baseDQN(t_o), 1, t_a_index)
                t_qvalue = t_qvalue*(GAMMA**self.e_trace) + t_r.unsqueeze(-1)

            dqn_loss = self.criterion(t_p_qvalue, t_qvalue)

            self.optimizer_p.zero_grad()
            pg_loss.backward(retain_graph=True)
            for param in self.updatedPG.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer_p.step()

            self.optimizer_q.zero_grad()
            dqn_loss.backward(retain_graph=True)
            for param in self.updatedDQN.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer_q.step()

            i = i + 1
        print("loss1 = ", pg_loss)
        print("loss2 = ", dqn_loss)

        return pg_loss, dqn_loss

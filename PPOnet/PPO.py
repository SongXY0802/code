import torch
import gurobipy as gp
import os
import numpy as np
from helper import state_vnode_represent
import sys
import math
from sklearn.utils import shuffle
import torch.nn.functional as F

INFINITY = 1e+20  # 定义无穷大
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PPO:
    def __init__(self, actor_lr, critic_lr, gamma, actor, critic, eps, lmbda, memory_size, horizon, action_range,
                 entropy_coef, imi_coef, destroy_degree):
        # 变化的参数
        self.ins_low_bound = []  # 原问题下界
        self.ins_up_bound = []  # 原问题上界
        self.variable_number = []  # 问题变量数
        self.problem_model = []  # 原问题模型
        self.instance_name = []  # 问题名称
        self.nor_obj = None  # 归一化原问题的系数
        self.constraint_features = []
        self.edge_index = []
        self.variable_features_static = []
        self.edge_attr = []
        self.initial_sol = []  # 初始解
        self.entropy_coef = entropy_coef  # 交叉熵系数
        self.imi_coef = imi_coef  # 模仿学习系数
        self.destroy_degree = destroy_degree  # 破坏程度（邻域半径）

        # 不变的参数
        self.memory_size = memory_size
        self.horizon = horizon
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.actor = actor
        self.critic = critic
        self.eps = eps  # PPO中截断范围的参数
        self.lmbda = lmbda
        self.action_range = action_range
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        # memory
        self.memory_state_id = []
        self.memory_state_old_solution = []
        self.memory_td = []
        self.memory_advantage = []

    def initialize(self, batch, DIR_INS):
        gp.setParam('LogToConsole', 0)  # 是否输出控制台信息
        batch = batch.to(device)
        self.old_obj_ori = torch.FloatTensor(np.array(batch.sols['objs'])).reshape(-1).numpy()  # 初始解
        var_beginid = 0
        con_beginid = 0
        attr_begin = 0
        ini_solution = torch.FloatTensor(np.array(batch.sols['sols']))  # 当前解(初始解)
        for i in range(len(batch)):
            self.initial_sol.append(torch.FloatTensor(np.array(batch.sols['sols']))[i].reshape(-1).numpy())
            # name = batch.files[i].split("/")[4].replace('.bg', '')
            name = batch.files[i].split("\\")[1].replace('.bg', '')
            ins_location = os.path.join(DIR_INS, name)  # sample原始问题的地址
            self.instance_name.append(name)
            m = gp.read(ins_location)  # 读取原始问题
            self.problem_model.append(m)
            variable = m.getVars()  # 获得全部变量

            var_endid = var_beginid + len(variable)
            con_endid = con_beginid + len(m.getConstrs()) + 1  # 约束数目

            self.variable_number.append(len(variable))
            up_bound = [INFINITY] * len(variable)  # 记录原始的上下界
            low_bound = [-1 * INFINITY] * len(variable)
            for index, var in enumerate(variable):  # 每个变量对应的batch里面的编号
                up_bound[index] = var.ub
                low_bound[index] = var.lb
            self.ins_up_bound.append(up_bound)
            self.ins_low_bound.append(low_bound)

            self.constraint_features.append(batch.constraint_features[con_beginid:con_endid])
            self.edge_index.append(batch.edge_ind[i * 2:(i + 1) * 2])
            self.variable_features_static.append(batch.variable_features[var_beginid:var_endid])
            end_attr = attr_begin + len(batch.edge_ind[i * 2:(i + 1) * 2][0])
            self.edge_attr.append(batch.edge_attr[attr_begin:end_attr])
            # print(self.actor.forward(self.constraint_features[i], self.edge_index[i], self.edge_attr[i],
            # self.variable_features_static[i]))
            var_beginid = var_endid
            con_beginid = con_endid
            attr_begin = end_attr
        self.nor_obj = batch.nor_ob

    def choose_action(self, ins_id, old_solution):
        constraint_features = self.constraint_features[ins_id]
        edge_indices = self.edge_index[ins_id]
        edge_features = self.edge_attr[ins_id]
        probabilities = self.actor.forward(constraint_features, edge_indices, edge_features,
                                           state_vnode_represent(self.variable_features_static[ins_id], old_solution))
        # probabilities = torch.clamp(probabilities, self.action_range[0], self.action_range[1])
        # action = torch.tensor(np.random.binomial(1, probabilities.cpu().detach().numpy()))
        action = torch.zeros(self.variable_number[ins_id])
        if torch.any(torch.isnan(probabilities)):
            sys.exit()
        _, indices = torch.topk(probabilities, k=math.ceil(len(probabilities) * self.destroy_degree))
        action.index_fill_(0, indices.cpu(), 1)
        return action

    def step(self, action, ins_id, old_solution):
        m = self.problem_model[ins_id].copy()
        variable = m.getVars()  # 获得全部变量
        for index, var in enumerate(variable):  # 改变上下界
            if action[index] == 0:
                var.Start = old_solution[index]
                var.ub = old_solution[index]
                var.lb = old_solution[index]
                m.update()
            else:
                var.Start = old_solution[index]
                var.ub = self.ins_up_bound[ins_id][index]
                var.lb = self.ins_low_bound[ins_id][index]
                m.update()
        m.update()
        m.setParam('LogToConsole', 0)  # 是否输出控制台信息
        # 计算新的解
        # m.Params.PoolSolutions = 2
        m.Params.PoolSearchMode = 2
        # m.Params.SolutionLimit = 2
        m.Params.TimeLimit = 1
        m.optimize()
        current_solution = m.Xn
        reward = np.dot(self.nor_obj[ins_id], old_solution) - np.dot(self.nor_obj[ins_id], current_solution)
        # print('reward', reward)

        # 计算当前状态的价值
        constraint_features = self.constraint_features[ins_id]
        edge_indices = self.edge_index[ins_id]
        edge_features = self.edge_attr[ins_id]
        value = self.critic.forward(constraint_features, edge_indices, edge_features,
                                    state_vnode_represent(self.variable_features_static[ins_id], old_solution))
        return reward, current_solution, value, m.PoolObjVal

    def save(self, trajectory, old_solution, ins_id):
        # 计算最后一个value
        constraint_features = self.constraint_features[ins_id]
        edge_indices = self.edge_index[ins_id]
        edge_features = self.edge_attr[ins_id]
        value = self.critic.forward(constraint_features, edge_indices, edge_features,
                                    state_vnode_represent(self.variable_features_static[ins_id], old_solution))
        trajectory.values.append(value)
        # 计算advantage function
        trajectory.advantage(self.gamma, self.lmbda)
        for i in range(len(trajectory.td_target)):
            self.memory_td.append(trajectory.td_target[i])
            self.memory_advantage.append(trajectory.advantage[i])

    def get_mini_batch(self, batch_id, batch, mini_batch_size):
        begin = batch_id * mini_batch_size
        end = (batch_id + 1) * mini_batch_size
        if end > self.horizon * self.memory_size:
            end = self.horizon * self.memory_size
        state_id = batch[0][begin:end]
        old_solution = batch[1][begin:end]
        td_target = batch[2][begin:end]
        advantage = batch[3][begin:end]
        return state_id, old_solution, td_target, advantage

    def learn(self, mini_batch_size, epoch_num, iteration, imitation_net, actor_old):
        # 当前actor的参数复制给旧actor
        actor_old.load_state_dict(self.actor.state_dict())
        epoch_cycles_nm = math.ceil(self.horizon * self.memory_size / mini_batch_size)
        self.memory_advantage = (self.memory_advantage - np.mean(self.memory_advantage)) / np.std(self.memory_advantage)
        # print('learn')
        # print("begin:{}".format(torch.cuda.memory_allocated(0)))
        IMLOSS = 0
        RLLOSS = 0
        for epoch in range(epoch_num):
            batch = shuffle(self.memory_state_id, self.memory_state_old_solution, self.memory_td, self.memory_advantage)
            for cycle in range(epoch_cycles_nm):
                # print('cycle')
                state_id, old_solution, td_target, advantage = self.get_mini_batch(batch_id=cycle, batch=batch,
                                                                                   mini_batch_size=mini_batch_size)

                ac_lo = 0
                cri_lo = 0
                imi_loss = 0
                for i in range(len(state_id)):
                    # 旧的log概率
                    old_prob = actor_old.forward(self.constraint_features[state_id[i]], self.edge_index[state_id[i]],
                                                 self.edge_attr[state_id[i]],
                                                 state_vnode_represent(self.variable_features_static[state_id[i]],
                                                                       old_solution[i])).double()
                    action = torch.zeros(self.variable_number[state_id[i]])
                    if torch.any(torch.isnan(old_prob)):
                        sys.exit()
                    _, indices = torch.topk(old_prob, k=math.ceil(len(old_prob) * self.destroy_degree))
                    # 旧的action
                    action.index_fill_(0, indices.cpu(), 1)
                    action = action.to(device)
                    log_old_prob = torch.log(
                        torch.mul(old_prob, action) + torch.mul(1 - old_prob, 1 - action)).detach()
                    # 新的log概率
                    prob = self.actor.forward(self.constraint_features[state_id[i]], self.edge_index[state_id[i]],
                                              self.edge_attr[state_id[i]],
                                              state_vnode_represent(self.variable_features_static[state_id[i]],
                                                                    old_solution[i])).double()
                    log_prob = torch.log(torch.mul(prob, action) + torch.mul(1 - prob, 1 - action))
                    radio = torch.exp(log_prob - log_old_prob)
                    new_value = self.critic.forward(self.constraint_features[state_id[i]], self.edge_index[state_id[i]],
                                                    self.edge_attr[state_id[i]],
                                                    state_vnode_represent(self.variable_features_static[state_id[i]],
                                                                          old_solution[i]))
                    surr1 = advantage[i] * radio
                    surr2 = torch.clamp(radio, 1 - self.eps, 1 + self.eps) * advantage[i]
                    # 专家网络输出
                    imi_pro = imitation_net.forward(self.constraint_features[state_id[i]], self.edge_index[state_id[i]],
                                                    self.edge_attr[state_id[i]],
                                                    state_vnode_represent(self.variable_features_static[state_id[i]],
                                                                          old_solution[i]))
                    imi_action = torch.zeros(self.variable_number[state_id[i]])
                    _, indices = torch.topk(imi_pro, k=math.ceil(len(imi_pro) * self.destroy_degree))
                    imi_action.index_fill_(0, indices.cpu(), 1)
                    imi_action = imi_action.to(device)
                    # 模仿的loss
                    imi_loss += -torch.sum(torch.log(torch.mul(prob, imi_action) + torch.mul(1 - prob, 1 - imi_action)))
                    # PPO loss
                    ac_lo = ac_lo - torch.min(surr1, surr2) - self.entropy_coef * torch.mul(prob,
                                                                                            torch.log(prob)).mean()
                    cri_lo = cri_lo + F.mse_loss(new_value, torch.tensor(td_target[i]).to(device))
                imi_loss = imi_loss / (len(state_id) * 300)  # 每个变量的平均模仿loss
                rl_loss = ac_lo.mean() / len(state_id)
                # actor loss
                ac_lo = rl_loss + self.imi_coef * imi_loss
                '''
                if iteration < 40:
                    ac_lo = rl_loss + self.imi_coef * imi_loss
                else:
                    ac_lo = rl_loss
                '''
                # critic loss
                cri_lo = cri_lo / len(state_id)

                # loss记录
                IMLOSS += imi_loss.item()
                RLLOSS += rl_loss.item()
                '''
                print('imi_loss',imi_loss)
                print('rl_loss',rl_loss)
                print('cri_lo',cri_lo)'''
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                ac_lo.backward()
                cri_lo.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.step()
        return IMLOSS, RLLOSS

    def clean(self):  # 还原agent
        self.instance_name.clear()
        self.problem_model.clear()
        self.variable_number.clear()
        self.ins_low_bound.clear()
        self.ins_up_bound.clear()
        self.nor_obj = None
        self.constraint_features.clear()
        self.edge_index.clear()
        self.variable_features_static.clear()
        self.edge_attr.clear()
        self.initial_sol.clear()
        self.memory_state_id.clear()
        self.memory_state_old_solution.clear()
        self.memory_td.clear()
        self.memory_advantage = []
        self.old_obj_ori = []

    def save_models(self, ckpt_dir, iteration):
        self.actor.save_checkpoint(ckpt_dir + 'PPO_policy_{}.pth'.format(iteration))
        self.critic.save_checkpoint(ckpt_dir + 'PPO_value_{}.pth'.format(iteration))
        print('Saved the policy network successfully!')

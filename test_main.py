import torch
import os
from sklearn.utils import shuffle
from PPOnet.GCN import GraphDataset
from PPOnet.GCN import PolicyNet, ValueNet
from PPOnet.PPO import PPO
from PPOnet.Trajectory import Trajectory
import torch_geometric
from valid import valid
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import time

# 参数
actor_lr = 1e-6  # actor学习率 im：-7
critic_lr = 1e-6  # critic学习率
gamma = 0.98  # 折扣因子
lmbda = 0.95
BATCH_SIZE = 16
NUM_WORKERS = 0
nb_iteration = 1500
embedding_size = 64
cons_feats_num = 4  # 约束特征数
edge_feats_num = 1  # 边特征数
var_feats_num = 6  # helper中得到的节点特征数
eps = 0.2  # PPO中截断范围的参数
memory_size = BATCH_SIZE
horizon = 4  # 每个trajectory的长度4
destroy_degree = 0.1  # destroy的程度
epoch_num = 3  # 一批数据学几次（抽取几个mini_batch）
mini_batch_size = 32  # 从经验池中抽mini_batch的大小
entropy_coef = 0.01
action_range = (0.2, 0.8)  # action概率的裁剪
imi_coef = 0.1
# epoch_cycles_nm =4  # 每个epoch训练几个minibatch horizon*memory_size

TaskName = "SC"
DIR_BG = f'./dataset/{TaskName}/train/BG'  # 二分图存储路径
DIR_SOL = f'./dataset/{TaskName}/train/solution'  # 解存储路径
DIR_INS = f'./instance/{TaskName}/train'  # 原问题的存储位置
ckpt_dir = f'./rl_model/{TaskName}'  # 神经网络模型参数存储路径
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
pre_ckpt_dir = f'./im_checkpoints/{TaskName}'  # 预训练神经网络模型参数存储路径

if __name__ == '__main__':
    # 加载数据地址
    train_names = os.listdir(DIR_BG)
    train_files = [(os.path.join(DIR_BG, name), os.path.join(DIR_SOL, name).replace('bg', 'sol')) for name in
                   train_names]
    instance_files = [os.path.join(DIR_INS, name).replace('.bg', '') for name in train_names]
    # 初始化GCN
    critic = ValueNet(emb_size=embedding_size, cons_nfeats=cons_feats_num, edge_nfeats=edge_feats_num,
                      var_nfeats=var_feats_num + 1)
    actor = PolicyNet(emb_size=embedding_size, cons_nfeats=cons_feats_num, edge_nfeats=edge_feats_num,
                      var_nfeats=var_feats_num + 1)
    # 旧网络
    actor_old = PolicyNet(emb_size=embedding_size, cons_nfeats=cons_feats_num, edge_nfeats=edge_feats_num,
                      var_nfeats=var_feats_num + 1)
    imitation_net = PolicyNet(emb_size=embedding_size, cons_nfeats=cons_feats_num, edge_nfeats=edge_feats_num,
                              var_nfeats=var_feats_num + 1)
    actor.initialize()
    critic.initialize()
    imitation_net.load_checkpoint(pre_ckpt_dir + 'imitation_policy_LB800.pth')
    # actor.load_checkpoint(pre_ckpt_dir + 'imitation_policy_LB800.pth')
    # 定义agent
    agent = PPO(actor_lr=actor_lr, critic_lr=critic_lr, gamma=gamma, actor=actor, critic=critic, eps=eps, lmbda=lmbda,
                memory_size=memory_size, horizon=horizon, action_range=action_range, entropy_coef=entropy_coef,
                imi_coef=imi_coef, destroy_degree=destroy_degree)
    shuffle(train_files, instance_files, random_state=0)  # 打乱顺序
    train_data = GraphDataset(train_files)  # 调用类函数
    train_loader = torch_geometric.loader.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,
                                                     num_workers=NUM_WORKERS)
    train_rewards = []
    valid_rewards = []
    rl_loss_rec = []
    im_loss_rec = []

    for iteration in range(nb_iteration):  # 总共迭代轮次
        rewards_rec = 0
        imi_loss = 0
        rl_loss = 0
        print('iteration:', iteration)
        # 抽取需要采集轨迹更新的batch
        for step, batch in enumerate(train_loader):
            # agent初始化
            agent.initialize(batch, DIR_INS)
            # 收集trajectory
            tra_number = 0  # 轨迹计数
            while tra_number < memory_size:
                ins_id = tra_number % len(agent.instance_name)  # instance在batch中的id
                trajectory = Trajectory(instance=agent.instance_name[ins_id], ins_id=ins_id)
                old_solution = agent.initial_sol[ins_id]
                inisol = agent.old_obj_ori[ins_id]  # 没有归一化的目标函数值
                # 轨迹采集
                for t in range(horizon):
                    action = agent.choose_action(ins_id, old_solution)
                    reward, current_solution, value, obg_ori = agent.step(action, ins_id, old_solution)
                    trajectory.save(reward=reward, value=value)
                    # 记录数据
                    agent.memory_state_id.append(ins_id)  # 记录问题id
                    agent.memory_state_old_solution.append(old_solution)  # 记录旧解
                    old_solution = current_solution
                    rewards_rec += inisol - obg_ori
                    inisol = obg_ori
                agent.save(trajectory, old_solution, ins_id)
                tra_number = tra_number + 1
            # 学习
            IMLOSS, RLLOSS = agent.learn(mini_batch_size, epoch_num, iteration, imitation_net,actor_old)
            imi_loss += IMLOSS
            rl_loss += RLLOSS
            # agent reset
            agent.clean()
        rec = valid(TaskName, agent, horizon, destroy_degree)
        if (iteration + 1) % 30 == 0:
            agent.save_models(ckpt_dir, iteration)
        train_rewards.append(rewards_rec)
        valid_rewards.append(rec)
        rl_loss_rec.append(rl_loss)
        im_loss_rec.append(imi_loss)
        print(rec)
        print('total rewards:', rewards_rec)
        if (iteration + 1) % 100 == 0:
            plt.figure()
            # 创建画布
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            X = np.arange(start=0, stop=len(train_rewards), step=1)

            axs[0].plot(X, train_rewards)
            axs[0].set_title('train_reward')

            # func1 = interp1d(X, train_rewards, kind='cubic')
            # y_new = func1(X)
            # axs[0].plot(X, y_new)

            axs[1].plot(X, valid_rewards)
            axs[1].set_title('valid_rewards')
            # func2 = interp1d(X, valid_rewards, kind='cubic')
            # y_new2 = func2(X)
            # axs[1].plot(X, y_new2)

            # method: 插值方法: 可选 {‘linear’, ‘nearest’, ‘cubic’} 之一
            # ‘linear’: 分段线性, ‘nearest’: 最近邻点, ‘cubic’: 三次样条（cubic spline）插值
            # func = interp1d(xdata, ydata, kind='cubic')
            # x_new = np.linspace(start=min(xdata), stop=max(xdata), num=10)
            # y_new = func(x_new)
            # plt.scatter(x_new, y_new)

            # 调整子图之间的间距
            fig.tight_layout()

            # 显示图表
            plt.show()
            # plt.savefig(f"{iteration}_iteration.jpg")

            plt.figure()
            # 创建画布
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            X = np.arange(start=0, stop=len(rl_loss_rec), step=1)

            axs[0].plot(X, rl_loss_rec)
            axs[0].set_title('PPO_actor_loss')

            axs[1].plot(X, im_loss_rec)
            axs[1].set_title('imitation_loss')

            # 调整子图之间的间距
            fig.tight_layout()

            # 显示图表
            plt.show()

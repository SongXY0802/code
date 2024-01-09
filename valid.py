import os
VALID_BATCH_SIZE = 1
from sklearn.utils import shuffle
from PPOnet.GCN import GraphDataset
import torch_geometric
def valid(TaskName,agent,horizon,destroy_degree):
    DIR_BG = f'./dataset/{TaskName}/valid/BG'  # 二分图存储路径
    DIR_SOL = f'./dataset/{TaskName}/valid/solution'  # 解存储路径
    DIR_INS = f'./instance/{TaskName}/valid'  # 原问题的存储位置
    valid_names = os.listdir(DIR_BG)
    valid_files = [(os.path.join(DIR_BG, name), os.path.join(DIR_SOL, name).replace('bg', 'sol')) for name in
                   valid_names]
    instance_files = [os.path.join(DIR_INS, name).replace('.bg', '') for name in valid_names]
    shuffle(valid_files, instance_files, random_state=0)  # 打乱顺序
    valid_data = GraphDataset(valid_files)  # 调用类函数
    valid_loader = torch_geometric.loader.DataLoader(valid_data, batch_size=1, shuffle=True,
                                                     num_workers=0)
    rec = 0
    for step, batch in enumerate(valid_loader):
        agent.initialize(batch, DIR_INS)
        for i in range(len(batch)):
            inisol = agent.old_obj_ori[i]  # 没有归一化的目标函数值
            old_solution = agent.initial_sol[i]
            for t in range(horizon):
                action = agent.choose_action(i, old_solution)
                reward, current_solution, value, obg_ori = agent.step(action, i, old_solution)
                old_solution = current_solution
                rec = rec+inisol-obg_ori
                #rec +=reward
                inisol=obg_ori
        agent.clean()
    return rec



import torch
import os
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from PPOnet.GCN import GraphDataset
import torch_geometric
from PPOnet.GCN import PolicyNet
import sys
import math

Radius = 30
TaskName = "SC"
bg_dir = f'./imitation_ins/{TaskName}/BG'  # 二分图存储路径
action_dir = f'./imitation_ins/{TaskName}/action'  # action存储路径
ckpt_dir = f'./im_checkpoints/{TaskName}'  # 神经网络模型参数存储路径
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DIR_INS = f'./instance/train/{TaskName}'
model_path = f'./im_state/model_{TaskName}_r={Radius}_IL.pth'
BATCH_SIZE = 16  # 每个batch有多少个样本
NUM_WORKERS = 0  # 有几个进程来处理data loading
LEARNING_RATE = 0.001
NB_EPOCHS = 1200
degree = 0.1
embedding_size = 64
cons_feats_num = 4  # 约束特征数
edge_feats_num = 1  # 边特征数
var_feats_num = 6  # helper中得到的节点特征数


def train(gnn, data_loader, epoch, optimizer=None):
    """
    This function will process a whole epoch of training or validation, depending on whether an optimizer is provided.
    """
    if optimizer:
        gnn.train()
    else:
        gnn.eval()

    mean_loss = 0
    mean_accurate = 0
    n_samples_processed = 0
    with torch.set_grad_enabled(optimizer is not None):
        for step, batch in enumerate(data_loader):
            batch = batch.to(DEVICE)
            action_pro = gnn(
                batch.constraint_features,
                batch.edge_index,
                batch.edge_attr,
                batch.variable_features,
            )

            if torch.any(torch.isnan(action_pro)):
                sys.exit()
            action_pro = action_pro.double()
            n_samples = len(batch)

            loss = 0
            startindx = 0
            tau = 0.07
            acurrate = 0
            for j in range(n_samples):  ####each sample in this batch
                nvars = batch.ntvars[j]
                pi = action_pro[startindx:startindx + nvars]

                _, indices = torch.topk(pi, k=math.ceil(nvars * degree))
                startindx += nvars

                p = torch.DoubleTensor(batch.sols[j]).to(DEVICE)  # 选择的是destroy的节点
                pp = p.clone()

                for i in range(math.ceil(nvars * degree)):
                    if pp[indices[i].item()].item() == 1:
                        acurrate = acurrate + 1

                l = -torch.sum(torch.log(torch.mul(pi, p) + torch.mul(1 - pi, 1 - p)))
                if not torch.isnan(l):
                    loss += l
            if optimizer is not None:
                optimizer.zero_grad()
                if loss == 0:
                    continue
                loss.backward()
                optimizer.step()
            if type(loss) != int:
                mean_loss += loss.item()
            mean_accurate = mean_accurate + acurrate
            n_samples_processed += batch.num_graphs
    mean_loss /= n_samples_processed
    mean_accurate /= n_samples_processed
    torch.save(gnn.state_dict(), model_path)

    return mean_loss, mean_accurate


def main():
    train_names = os.listdir(bg_dir)

    train_files = [(os.path.join(bg_dir, name), os.path.join(action_dir, name).replace('bg', 'act')) for name in
                   train_names]
    # print(len(train_files))
    # instance_files = [os.path.join(DIR_INS, name + '.lp') for name in [name.split("_step")[0] for name in train_names]]
    # print(instance_files)
    shuffle(train_files, random_state=0)  # 打乱顺序
    train_files = train_files[: int(0.9 * len(train_files))]
    valid_files = train_files[int(0.9 * len(train_files)):]
    train_data = GraphDataset(train_files)
    train_loader = torch_geometric.loader.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,
                                                     num_workers=NUM_WORKERS)
    valid_data = GraphDataset(valid_files)
    valid_loader = torch_geometric.loader.DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=False,
                                                     num_workers=NUM_WORKERS)
    policy = PolicyNet(emb_size=embedding_size, cons_nfeats=cons_feats_num, edge_nfeats=edge_feats_num,
                       var_nfeats=var_feats_num + 1)
    policy.initialize()
    # policy.load_checkpoint(ckpt_dir + 'Reinforce_policy_{}.pth'.format(100))
    optimizer = torch.optim.Adam(policy.parameters(), lr=LEARNING_RATE)
    record_train_loss = []
    record_va_loss = []
    record_train_accurate = []
    record_va_accurate = []
    for epoch in range(NB_EPOCHS):
        torch.cuda.empty_cache()
        train_loss, train_accurate = train(policy, train_loader, epoch, optimizer)
        print(f"Epoch {epoch} Train loss: {train_loss:0.3f} Train Accurate:{train_accurate:0.3f}")
        valid_loss, valid_accurate = train(policy, valid_loader, epoch, None)
        print(f"Epoch {epoch} Valid loss: {valid_loss:0.3f} Valid Accurate:{valid_accurate:0.3f}")
        if epoch != 0:
            record_train_loss.append(train_loss)
            record_va_loss.append(valid_loss)
            record_train_accurate.append(train_accurate)
            record_va_accurate.append(valid_accurate)

    policy.save_checkpoint(ckpt_dir + 'imitation_policy_LB{}.pth'.format(NB_EPOCHS))
    print('Saved the policy network successfully!')
    plt.figure()
    X = np.arange(start=0, stop=len(record_train_loss), step=1)
    plt.plot(X, record_train_loss, label='train_loss')
    plt.plot(X, record_va_loss, label='valid_loss')
    plt.legend(loc='upper right', fontsize='medium')
    plt.show()
    plt.figure()
    '''
    X = np.arange(start=0, stop=len(record_train_accurate), step=1)
    plt.plot(X, record_train_accurate, label='train_loss')
    plt.plot(X, record_va_accurate, label='valid_loss')
    plt.show()'''


if __name__ == '__main__':
    main()

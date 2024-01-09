import gurobipy as gp
# import pyscipopt as scp
import numpy as np
import torch

INFINITY = 1e+20  # 定义无穷大
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_a_new2(ins_name):
    gp.setParam('LogToConsole', 0)  # 是否输出log信息
    m = gp.read(ins_name)
    A_in = [[], []]  # [[约束编号][非零系数索引]]
    A_value = []
    nor_co = []  # 归一化系数

    # 构建约束特征
    c_map = {}  # 约束名字编号
    constr = m.getConstrs()  # 获取全部约束
    number_cons = len(constr)  # 约束数目
    c_nodes = np.zeros((number_cons + 1, 4))  # 约束特征共四个[,,约束右端项,]
    for index, var in enumerate(constr):
        c_map[var.ConstrName] = index
        c_nodes[index][2] = var.rhs  # 约束右端项
        if var.sense == '=':
            c_nodes[index][3] = 2
        elif var.sense == '>':
            c_nodes[index][3] = 1

    # 构建节点特征
    mvars = m.getVars()  # 获取全部变量
    nvars = len(mvars)  # 变量总数
    v_map = {}  # 变量编号
    v_nodes = np.zeros((nvars, 6))  # 不变的节点特征[在目标函数中系数,系数均值,变量出现在约束的次数,最大系数,最小系数,是否是二元变量]
    b_vars = []  # 是否是二元变量

    for index, var in enumerate(mvars):  # 求特定变量在约束中平均系数和出现次数
        v_map[var.varName] = index  # 建立索引
        a = m.getCol(mvars[index])  # 获取mvars[index]变量的所有约束
        if var.vType == 'B':  # 是否是二元变量
            b_vars.append(index)
            v_nodes[index][5] = 1
        var_co_add = 0
        co = []
        for i in range(a.size()):
            # 保存该变量在约束中的所有系数
            var_co_add = var_co_add + a.getCoeff(i)
            co.append(a.getCoeff(i))
            # 保存约束对应的系数
            co_indicase = c_map[a.getConstr(i).ConstrName]  # 约束对应的编号
            co_number = c_nodes[co_indicase][1]
            c_nodes[co_indicase][0] = (c_nodes[co_indicase][0] * co_number + a.getCoeff(i)) / (co_number + 1)
            c_nodes[co_indicase][1] = co_number + 1
            A_in[0].append(co_indicase + 1)
            A_in[1].append(index)
            A_value.append(a.getCoeff(i))
        if len(co) == 0:
            v_nodes[index][3] = INFINITY
            v_nodes[index][4] = -1 * INFINITY
        else:
            v_nodes[index][3] = max(co)
            v_nodes[index][4] = min(co)
            v_nodes[index][1] = sum(co) / number_cons
            v_nodes[index][2] = len(range(a.size()))

    # 获取目标函数
    obj = m.getObjective()
    c_nodes[number_cons][1] = obj.size()
    for i in range(obj.size()):
        c_nodes[number_cons][0] = c_nodes[number_cons][0] + obj.getCoeff(i)
        A_in[0].append(0)
        A_in[1].append(v_map[obj.getVar(i).VarName])
        A_value.append(obj.getCoeff(i))
        v_nodes[v_map[obj.getVar(i).VarName]][0] = obj.getCoeff(i)
        nor_co.append(obj.getCoeff(i))

    c_nodes[number_cons][0] = c_nodes[number_cons][0] / obj.size()

    # 获取tensor
    v_nodes = torch.as_tensor(v_nodes, dtype=torch.float32)
    c_nodes = torch.as_tensor(c_nodes, dtype=torch.float32)
    b_vars = torch.as_tensor(b_vars, dtype=torch.int32)
    A = torch.sparse_coo_tensor(A_in, A_value, (number_cons + 1, nvars))

    # 对属性值进行归一化
    clip_max = [20000, 1, torch.max(v_nodes, 0)[0][2].item()]  # torch.max(v_nodes, 0)[0][2].item()约束中出现次数最多的变量出现的次数
    clip_min = [0, -1, 0]
    # v_nodes[:, 0]是v_nodes的第一维即目标中系数
    # clamp（）函数的功能将输入input张量每个元素的值压缩到区间 [min,max]，并返回结果到一个新张量。
    # 压缩目标函数的系数值

    nor_co = nor_co / np.linalg.norm(nor_co)
    v_nodes[:, 0] = torch.clamp(v_nodes[:, 0], clip_min[0], clip_max[0])
    maxs = torch.max(v_nodes, 0)[0]
    mins = torch.min(v_nodes, 0)[0]
    diff = maxs - mins  # 每一维的最大和最小的差
    for ks in range(diff.shape[0]):
        if diff[ks] == 0:
            diff[ks] = 1
    v_nodes = v_nodes - mins
    v_nodes = v_nodes / diff
    v_nodes = torch.clamp(v_nodes, 1e-5, 1)

    maxs = torch.max(c_nodes, 0)[0]
    mins = torch.min(c_nodes, 0)[0]
    diff = maxs - mins
    c_nodes = c_nodes - mins
    c_nodes = c_nodes / diff
    c_nodes = torch.clamp(c_nodes, 1e-5, 1)
    return A, v_map, v_nodes, c_nodes, b_vars, nor_co


def state_vnode_represent(v_nodes, cur_sol_val):
    v_state_nodes = [torch.cat((nodefeat, torch.tensor([cur_sol_val[idx]]).to(device)), dim=0) for idx, nodefeat in
                     enumerate(v_nodes)]
    v_state_nodes = torch.stack(v_state_nodes)
    return v_state_nodes

def keep_largest(nums, k):
    max_nums = sorted(range(len(nums)),key=lambda i:nums[i], reverse=True)[:k] # 获取最大的k个数
    result = [1 if i in max_nums else 0 for i in range(len(nums))]  # 将不在最大k个数中的元素设为0
    return result
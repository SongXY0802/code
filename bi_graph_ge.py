import argparse
import os.path
from multiprocessing import Process, Queue  # 多进程模块
import gurobipy as gp
import numpy as np
import pickle
import time
from helper import get_a_new2


def solve_grb(filepath, settings):
    gp.setParam('LogToConsole', 0)  # 是否输出控制台参数信息
    m = gp.read(filepath)

    m.Params.PoolSolutions = settings['maxsol']  # 决定存储可行解的数量，不改变求解器求解过程以及最终能够获得的解的数量，只负责从最终得到的解中保留用户指定数量的解
    m.Params.PoolSearchMode = settings['mode']  # 改编搜索模式，0：只专注于寻找最优解，1：尽可能搜索用户指定数量的可行解但不能保证解的质量，2：寻找最好用的一批可行解
    # m.Params.TimeLimit = settings['maxtime']
    m.Params.SolutionLimit = settings['solutionLimit']
    m.Params.Threads = settings['threads']  # 设置使用的核数

    m.optimize()

    sols = []  # 解
    objs = []  # 目标函数值
    sol_number = m.getAttr('SolCount')  # 输出Pool中获得的解的数量
    # 获取变量
    mvars = m.getVars()
    # 获取变量名
    var_names = [var.varName for var in mvars]

    for sn in range(sol_number):
        m.Params.SolutionNumber = sn
        sols.append(np.array(m.Xn))  # Pool中特定解的决策变量值
        objs.append(m.PoolObjVal)  # Pool中特定解的目标函数值

    sols = np.array(sols, dtype=np.float64)
    objs = np.array(objs, dtype=np.float64)
    sol_data = {
        'var_names': var_names,
        'sols': sols,
        'objs': objs,
    }
    return sol_data


def collect(ins_dir, q, sol_dir, bg_dir, settings):
    while True:
        filename = q.get()
        if not filename:
            break
        filepath = os.path.join(ins_dir, filename)  # 数据地址
        sol_data = solve_grb(filepath, settings)
        if len(sol_data['sols']) != 0:  # 去除无解的实例
            A, v_map, v_nodes, c_nodes, b_vars, nor_co = get_a_new2(filepath)  # 获取初始二分图
            BG_data = [A, v_map, v_nodes, c_nodes, b_vars, nor_co]

            # save data
            pickle.dump(sol_data, open(os.path.join(sol_dir, filename + '.sol'), 'wb'))
            pickle.dump(BG_data, open(os.path.join(bg_dir, filename + '.bg'), 'wb'))


if __name__ == '__main__':
    sizes = ["SC"]
    start = time.time()
    parser = argparse.ArgumentParser()  # 创建解析器
    # 添加参数 type - 命令行参数应当被转换成的类型，default - 当参数未在命令行中出现时使用的值。
    parser.add_argument('--dataDir', type=str, default='./')  # dataDir存放数据路径
    parser.add_argument('--nWorkers', type=int, default=3)  # 线程数
    parser.add_argument('--maxTime', type=int, default=2)  # 最大求解时间
    parser.add_argument('--maxStoredSol', type=int, default=1)  # 最多存储的解数
    parser.add_argument('--threads', type=int, default=4)  # 核数
    parser.add_argument('--solutionLimit', type=int, default=1)  # 可行解数目
    args = parser.parse_args()  # 解析参数

    # 读取数据
    for size in sizes:
        dataDir = args.dataDir
        INS_DIR = os.path.join(dataDir, f'instance/{size}/train')

        if not os.path.isdir(f'./dataset/{size}'):
            os.mkdir(f'./dataset/{size}')
        if not os.path.isdir(f'./dataset/{size}/train/solution'):
            os.mkdir(f'./dataset/{size}/train/solution')
        if not os.path.isdir(f'./dataset/{size}/train/BG'):
            os.mkdir(f'./dataset/{size}/train/BG')

        SOL_DIR = f'./dataset/{size}/train/solution'
        BG_DIR = f'./dataset/{size}/train/BG'

        os.makedirs(SOL_DIR, exist_ok=True)
        os.makedirs(BG_DIR, exist_ok=True)

        N_WORKERS = args.nWorkers
        # gurobi settings
        SETTINGS = {
            'maxtime': args.maxTime,
            'mode': 0,
            'maxsol': args.maxStoredSol,
            'threads': args.threads,
            'solutionLimit': args.solutionLimit,
        }

        filenames = os.listdir(INS_DIR)  # 训练集数据文件名
        # 进程间通信multiproccessing.Queue()和线程间通信queue.Queue()
        # 1.queue.Queue()是进程内的非阻塞队列。2.multiprocessing.Queue()是跨进程通信的队列。
        q = Queue()
        # 添加训练数据
        for filename in filenames:
            if not os.path.exists(os.path.join(BG_DIR, filename + '.bg')):
                q.put(filename)

        # 添加停止符号
        for i in range(N_WORKERS):
            q.put(None)

        ps = []
        for i in range(N_WORKERS):
            # target:如果传递了函数的引用，子进程就会执行这个函数的代码，args:给target指定的函数传递的参数，以元组的方式传递
            p = Process(target=collect, args=(INS_DIR, q, SOL_DIR, BG_DIR, SETTINGS))
            p.start()  # 启动子进程实例（创建子进程）
            ps.append(p)
        for p in ps:
            p.join()  # 使用p.join()函数可以让主线程等待p的结束，避免卡住主进程
        print('done')
    print('Time:', time.time() - start)

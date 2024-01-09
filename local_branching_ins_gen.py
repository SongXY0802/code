import gurobipy as gp
import os
from io import StringIO
from contextlib import redirect_stdout
import numpy as np
import math
import torch
from helper import keep_largest
from helper import state_vnode_represent
from helper import get_a_new2
import pickle

TaskName = ["SC"]
DIR_INS = f'./instance/train/{TaskName}'
degree = 0.1
T = 4
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def LB(prob_name, num_traj, degree_destroy):
    gp.setParam('LogToConsole', 0)
    ins_file = f'./instance/{prob_name}/train'
    ins_names = os.listdir(ins_file)
    ins_files = [os.path.join(ins_file, name) for name in ins_names]
    sol_dir = f'./imitation_ins/{prob_name}/action'
    if not os.path.isdir(f'./imitation_ins/{prob_name}/action'):
        os.mkdir(f'./imitation_ins/{prob_name}/action')
    bg_dir = f'./imitation_ins/{prob_name}/BG'
    if not os.path.isdir(f'./imitation_ins/{prob_name}/BG'):
        os.mkdir(f'./imitation_ins/{prob_name}/BG')
    diff_best = 0
    diff_LB = 0
    for idx_ins, ins_name in enumerate(ins_files):
        print(f'{idx_ins}th instance----------')
        output_buffer = StringIO()
        with redirect_stdout(output_buffer):
            m = gp.read(ins_name)
            mm = gp.read(ins_name)
        m.setParam("OutputFlag", 0)
        mvars = m.getVars()
        mm.optimize()

        # initial solution
        m.setParam('SolutionLimit', 1)
        m.optimize()
        cur_sol_val = [v.X for v in mvars]
        ave_solution = np.array(cur_sol_val.copy())
        best_solution = np.array(cur_sol_val.copy())
        cur_sol = m.PoolObjVal
        best_sol = m.PoolObjVal
        lb_selection = [0] * len(cur_sol_val)
        fixtimes = [0] * len(cur_sol_val)
        print('最优解值', mm.PoolObjVal)
        print('初始解值', m.PoolObjVal)
        diff_best = diff_best + m.PoolObjVal - mm.PoolObjVal

        for step in range(num_traj):
            times = [step] * len(lb_selection)
            print(f'{step}th_step of the {idx_ins}th_instance')
            output_buffer = StringIO()
            with redirect_stdout(output_buffer):
                m = gp.read(ins_name)
            mvars = m.getVars()

            lb = gp.quicksum([v for i, v in enumerate(mvars) if cur_sol_val[i] < 0.5 and v.VType == 'B']) \
                 + gp.quicksum([1 - v for i, v in enumerate(mvars) if cur_sol_val[i] > 0.5 and v.VType == 'B'])
            m.addLConstr(lb, gp.GRB.LESS_EQUAL, len(mvars) * degree_destroy)
            start = 0
            for v in mvars:
                v.Start = cur_sol_val[start]
                start = start + 1
            m.update()
            m.optimize()
            print('当前解值', m.PoolObjVal)
            LBsol = [v.X for v in mvars]  # 新的解
            if cur_sol - m.PoolObjVal > 0:
                diff_LB += cur_sol - m.PoolObjVal
                A, v_map, v_nodes, c_nodes, b_vars, nor_co = get_a_new2(ins_name)
                v_nodes = v_nodes.to(device)
                v_nodes = state_vnode_represent(v_nodes, cur_sol_val)
                # v_nodes = state_vnode_represent(v_nodes, best_solution)
                # v_nodes = state_vnode_represent(v_nodes, ave_solution / (step + 1))
                # v_nodes = state_vnode_represent3(v_nodes, cur_sol, len(LBsol))
                # v_nodes = state_vnode_represent3(v_nodes, best_sol, len(LBsol))
                # v_nodes = state_vnode_represent(v_nodes, torch.Tensor(fixtimes))
                # v_nodes = state_vnode_represent(v_nodes, torch.Tensor(lb_selection))
                # v_nodes = state_vnode_represent(v_nodes, torch.Tensor(times))
                v_nodes = v_nodes.cpu().to(torch.float32)
                # 特征值归一化
                '''
                max = torch.max(v_nodes, 1, keepdim=False, out=None).values.unsqueeze(1)
                min = torch.min(v_nodes, 1, keepdim=False, out=None).values.unsqueeze(1)
                v_nodes = (v_nodes - min) / (max - min)
                v_nodes = torch.clamp(v_nodes, 1e-5, 1).cpu().to(torch.float32)'''
                BG = [A, v_map, v_nodes, c_nodes, b_vars, nor_co]
                pickle.dump(BG, open(os.path.join(bg_dir, f'instance_{idx_ins}_step{step}' + '.bg'), 'wb'))

                best_solution = LBsol
                best_sol = m.PoolObjVal
            diff = [np.abs(LBsol[i] - cur_sol_val[i]) for i in range(len(mvars))]
            lb_selection = keep_largest(diff, math.ceil(len(mvars) * degree_destroy))
            ave_solution = ave_solution + LBsol
            cur_sol_val = LBsol.copy()

            if cur_sol - m.PoolObjVal > 0:
                # save training data
                pickle.dump(lb_selection,
                            open(os.path.join(sol_dir, f'instance_{idx_ins}_step{step}' + '.act'), 'wb'))

            cur_sol = m.PoolObjVal
            fixtimes = np.sum([fixtimes, lb_selection], axis=0).tolist()
    print("diff best:", diff_best)
    print("diff LB:", diff_LB)


for problem in TaskName:
    LB(problem, T, degree)

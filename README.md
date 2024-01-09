generate_instances_SC.py：产生instance，存放于instance文件夹下

bi_graph_ge.py：产生二分图，存放于dataset文件夹下

test_main.py：主程序

PPOnet文件夹下包含了GCN网络，agent定义（PPO.py）和轨迹的类(Trajectory.py)

helper.py包含了两个函数，一个产生某个instance的二分图，一个连接节点静态和动态特征

local_branching_ins_gen.py产生local branching专家数据

imitation_train.py模仿网络训练

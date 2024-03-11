import gc
import subprocess
from multiprocessing import Process, Queue
import numpy as np
import math
from utils.bump_variables import bump_variables
from utils.cal_j import cal_j
from utils.change_wrf import change_wrf
from utils.flatten_variables import flatten_variables
from utils.parameter import delta, verification, q_dim, t_scale, p_scale, q_scale, u_dim, v_dim, t_dim, p_dim, dim_big
from utils.read_wrf import read_wrf
from utils.scale_variable import scale_variable
from utils.split_x_big import split_x_big


def run_proc(typhoon_name, q, x_small, times):
    a = np.load(f'sample_{typhoon_name}_{times}.npy')
    a = a * x_small[times]
    q.put(a)
    del a
    gc.collect()


def set_veri(x):
    x[:, :verification[2], :] = 0
    x[:, verification[3]:, :] = 0
    x[:, :, :verification[0]] = 0
    x[:, :, verification[1]:] = 0


def define_verification(u, v, t, p, q):
    set_veri(u)
    set_veri(v)
    set_veri(t)
    p[:verification[2], :] = 0
    p[verification[3]:, :] = 0
    p[:, :verification[0]] = 0
    p[:, verification[1]:] = 0
    set_veri(q)
    return [u, v, t, p, q]


class PSO:
    def __init__(self, typhoon_name='Maysak', dim=2, dim_big=238356, iter_max=30, swarm_size=10, w_max=0.9, w_min=0.4, c1=2, c2=2, v_max=0.1):
        self.delta = delta
        self.swarm_size = swarm_size
        self.dim = dim
        self.w_max = w_max
        self.w_min = w_min
        self.v_max = v_max
        self.w = w_max
        self.c1 = c1
        self.c2 = c2
        self.typhoon_name = typhoon_name
        self.X = np.zeros((swarm_size, dim))
        self.V = np.zeros((swarm_size, dim))
        self.p_best = np.zeros((swarm_size, dim))
        self.p_aff = np.zeros(swarm_size)
        self.fun = np.ones(swarm_size)
        self.global_params = [0 for _ in range(dim)]
        self.global_opt = float("inf")
        self.global_best_array = np.zeros(iter_max)
        self.iter_num = 0
        self.iter_max = iter_max
        self.solution = np.zeros(dim_big)

    def stopping_condition(self):
        status = bool(self.iter_num >= self.iter_max)
        return status


    def cal_solution(self, x_small):
        print('cal_solution')
        q = Queue()
        solution = np.zeros(dim_big)
        p = []
        read_one_time = 10
        loop = math.ceil(self.dim / read_one_time)
        for i in range(loop):
            start = read_one_time * i
            if i == loop - 1:
                end = self.dim
            else:
                end = read_one_time * (i + 1)
            for times in range(start, end):
                print(times)
                p.append(Process(target=run_proc, args=(self.typhoon_name, q, x_small, times)))
                p[times].start()
            for times in range(start, end):
                solution += q.get()
        del q
        gc.collect()
        return solution
        # solution = np.zeros(dim_big)
        # for i in range(self.dim):
        #     print(i)
        #     sample = np.load(f'sample_{self.typhoon_name}_{i}.npy')
        #     solution += sample * x_small[i]
        # return solution

    def obj_function(self, x_small):
        project_dir = '/home/users/wangxingzhou/cnop_wxz'
        subprocess.call(f'cp -r {project_dir}/wrfinput_d01 {project_dir}/wrfinput_d02', shell=True)
        x_big = self.cal_solution(x_small)
        scale_variable(x_big, x_small, self.delta)
        # ����wrfinput_d01��solution����wrfinput_d02
        [u_change, v_change, t_change, p_change, q_change] = split_x_big(x_big)
        t_change = t_change / np.sqrt(t_scale)
        p_change = p_change / np.sqrt(p_scale)
        q_change = q_change / np.sqrt(q_scale)
        [u_change, v_change, t_change, p_change, q_change] = bump_variables(u_change, v_change, t_change, p_change,
                                                                            q_change)
        change_wrf(u_change, v_change, t_change, p_change, q_change, 'wrfinput_d02')
        # ִ��wrf
        subprocess.call(f'sh {project_dir}/pso/CNOP.sh', shell=True)
        # ��ȡ���ɵ�wrfout_d01_fin���ʼ��wrfout_d01_ori����
        [u_ori, v_ori, t_ori, p_ori, q_ori] = read_wrf(f'{project_dir}/wrfout_d01_ori_cnop')
        [u_fin, v_fin, t_fin, p_fin, q_fin] = read_wrf(f'{project_dir}/wrfout_d01_fin_cnop')
        [u_ori, v_ori, t_ori, p_ori, q_ori] = define_verification(u_ori, v_ori, t_ori, p_ori, q_ori)
        [u_fin, v_fin, t_fin, p_fin, q_fin] = define_verification(u_fin, v_fin, t_fin, p_fin, q_fin)
        [u_ori, v_ori, t_ori, p_ori, q_ori] = flatten_variables(u_ori, v_ori, t_ori, p_ori, q_ori)
        [u_fin, v_fin, t_fin, p_fin, q_fin] = flatten_variables(u_fin, v_fin, t_fin, p_fin, q_fin)
        u_diff = u_fin - u_ori
        v_diff = v_fin - v_ori
        t_diff = (t_fin - t_ori) * np.sqrt(t_scale)
        p_diff = (p_fin - p_ori) * np.sqrt(p_scale)
        q_diff = (q_fin - q_ori) * np.sqrt(q_scale)
        j = cal_j(u_diff, v_diff, t_diff, p_diff, q_diff)
        j = -j
        print('function value: ' + str(j))
        return j

    def init_introduction(self):
        print('PSO initial finished, dimension: ' + str(self.dim) +
              ', swarm_size: ' + str(self.swarm_size) + ',iter_max: ' + str(self.iter_max) + '.')
        print('----------------------')
        print('global_params: ')
        print(self.global_params)
        print('global_opt: ')
        print(self.global_opt)

    def iter_introduction(self):
        print('--------' + str(self.iter_num + 1) + ' step--------')
        print('global_params: ')
        print(self.global_params)
        print('global_opt: ')
        print(self.global_opt)

    def end_introduction(self):
        print('PSO finished.')
        print('global_params: ')
        print(self.global_params)
        print('global_opt: ')
        print(self.global_opt)
        print('global best function value of steps: ')
        print(self.global_best_array)

    def increase_iter_num(self):
        self.iter_num += 1

    def init_swarm(self):
        self.iter_num = 0
        for i in range(self.swarm_size):
            print('swarm' + str(i))
            self.X[i] = np.random.randn(self.dim)
            self.V[i] = np.random.uniform(-self.v_max, self.v_max, self.dim)
            self.fun[i] = self.obj_function(self.X[i])
            self.p_best[i] = np.copy(self.X[i])
            self.p_aff[i] = self.fun[i]
            if self.fun[i] < self.global_opt:
                self.global_opt = self.fun[i]
                self.global_params = np.copy(self.X[i][:])
            print('function value: ')
            print(self.fun)
        self.solution = self.cal_solution(self.global_params)
        self.save_result()

    def continue_swarm(self):
        print('execute PSO')
        self.iter_num = np.loadtxt(f'iter_num_{self.typhoon_name}.txt')
        print('step ' + str(self.iter_num + 1))
        # self.delta = np.loadtxt('new_delta.txt')
        self.X = np.loadtxt(f'terminal_X_{self.typhoon_name}.txt')
        self.V = np.loadtxt(f'terminal_V_{self.typhoon_name}.txt')
        self.fun = np.loadtxt(f'terminal_fun_{self.typhoon_name}.txt')
        self.p_best = np.loadtxt(f'terminal_p_best_{self.typhoon_name}.txt')
        self.p_aff = np.loadtxt(f'terminal_p_aff_{self.typhoon_name}.txt')
        self.global_params = np.loadtxt(f'terminal_global_params_{self.typhoon_name}.txt')
        self.global_opt = np.loadtxt(f'terminal_global_opt_{self.typhoon_name}.txt')
        self.global_opt = float(self.global_opt)
        self.global_best_array = np.loadtxt(f'global_best_arr_{self.typhoon_name}.txt')

    def scale_v(self):
        self.V = np.where(self.V > self.v_max, self.v_max, self.V)
        self.V = np.where(self.V < -self.v_max, -self.v_max, self.V)

    def change_w(self):
        self.w = self.w_max - ((self.w_max - self.w_min) /
                               self.iter_max) * self.iter_num
        print('w changed: ')
        print(self.w)

    def cal_pso(self):
        print('step ' + str(self.iter_num + 1))
        for p in range(self.swarm_size):
            print('swarm ' + str(p))
            for q in range(self.dim):
                self.V[p][q] = self.w * self.V[p][q] + self.c1 * np.random.random() * (
                        self.p_best[p][q] - self.X[p][q]) + self.c2 * np.random.random() * (
                                       self.global_params[q] - self.X[p][q])
                self.scale_v()
                self.X[p][q] = self.X[p][q] + self.V[p][q]
            aff = self.obj_function(self.X[p])
            self.fun[p] = aff
            if aff < self.p_aff[p]:
                self.p_aff[p] = aff
                self.p_best[p] = np.copy(self.X[p])
            if aff < self.global_opt:
                self.global_opt = aff
                self.global_params = np.copy(self.X[p])
            print('function value: ')
            print(self.fun)
        self.global_best_array[int(self.iter_num)] = self.global_opt
        self.solution = self.cal_solution(self.global_params)
        self.increase_iter_num()
        self.save_result()

    def save_result(self):
        np.savetxt(f'iter_num_{self.typhoon_name}.txt', [self.iter_num])
        np.savetxt(f'terminal_p_best_{self.typhoon_name}.txt', self.p_best)
        np.savetxt(f'terminal_X_{self.typhoon_name}.txt', self.X)
        np.savetxt(f'terminal_V_{self.typhoon_name}.txt', self.V)
        np.savetxt(f'terminal_global_params_{self.typhoon_name}.txt', self.global_params)
        np.save(f'solution_{self.typhoon_name}.npy', self.solution)
        np.savetxt(f'terminal_global_opt_{self.typhoon_name}.txt', [self.global_opt])
        np.savetxt(f'terminal_fun_{self.typhoon_name}.txt', self.fun)
        np.savetxt(f'terminal_p_aff_{self.typhoon_name}.txt', self.p_aff)
        np.savetxt(f'global_best_arr_{self.typhoon_name}.txt', self.global_best_array)
        # np.savetxt('new_delta.txt', [self.delta])

    def execute_pso(self):
        if not self.stopping_condition():
            self.change_w()
            self.cal_pso()
            self.iter_introduction()
        if self.stopping_condition():
            self.end_introduction()

    def get_pso(self):
        self.continue_swarm()
        self.init_introduction()
        self.execute_pso()

    def init_pso(self):
        self.init_swarm()
        self.init_introduction()

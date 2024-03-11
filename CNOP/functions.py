import dataclasses
import functools
import os
import subprocess

import pandas as pd

from CNOP.parameters import variables_to_modify, lon_grid, lat_grid, level_grid
from graphcast import autoregressive
from graphcast import casting
from graphcast import checkpoint
from graphcast import data_utils
from graphcast import graphcast
from graphcast import normalization
from graphcast import rollout
from graphcast import xarray_jax
from graphcast import xarray_tree
import haiku as hk
import jax
import numpy as np
import xarray


def construct_wrapped_graphcast(
        model_config: graphcast.ModelConfig,
        task_config: graphcast.TaskConfig,
        diffs_stddev_by_level,
        mean_by_level,
        stddev_by_level):
    """Constructs and wraps the GraphCast Predictor."""
    # Deeper one-step predictor.
    # 构建基本预测器
    predictor = graphcast.GraphCast(model_config, task_config)

    # Modify inputs/outputs to `graphcast.GraphCast` to handle conversion to
    # from/to float32 to/from BFloat16.
    # 对预测器进行包装 以处理从 float32 到 BFloat16 的类型转换
    predictor = casting.Bfloat16Cast(predictor)

    # Modify inputs/outputs to `casting.Bfloat16Cast` so the casting to/from
    # BFloat16 happens after applying normalization to the inputs/targets.
    # 进一步包装预测器 以在类型转换之后应用标准化
    predictor = normalization.InputsAndResiduals(
        predictor,
        diffs_stddev_by_level=diffs_stddev_by_level,
        mean_by_level=mean_by_level,
        stddev_by_level=stddev_by_level)

    # Wraps everything so the one-step model can produce trajectories.
    # 预测器被包装在一个自回归预测器中 这允许它产生时间轨迹
    # 在训练过程中使用梯度检查点技术 优化内存使用
    predictor = autoregressive.Predictor(predictor, gradient_checkpointing=True)
    return predictor


# 前向传播 构建并运行构建的预测器 返回预测器的输出
@hk.transform_with_state
def run_forward(model_config, task_config, inputs, targets_template, forcings,
                diffs_stddev_by_level, mean_by_level, stddev_by_level):
    predictor = construct_wrapped_graphcast(model_config, task_config,
                                            diffs_stddev_by_level, mean_by_level, stddev_by_level)
    return predictor(inputs, targets_template=targets_template, forcings=forcings)


# 损失计算 返回计算的损失和诊断信息
@hk.transform_with_state
def loss_fn(model_config, task_config, inputs, targets, forcings,
            diffs_stddev_by_level, mean_by_level, stddev_by_level):
    predictor = construct_wrapped_graphcast(model_config, task_config,
                                            diffs_stddev_by_level, mean_by_level, stddev_by_level)
    loss, diagnostics = predictor.loss(inputs, targets, forcings)
    return xarray_tree.map_structure(
        lambda x: xarray_jax.unwrap_data(x.mean(), require_jax=True),
        (loss, diagnostics))


# 计算梯度 返回损失值、诊断信息、更新后的状态和计算出的梯度
def grads_fn(params, state, model_config, task_config, inputs, targets, forcings):
    def _aux(params, state, i, t, f):
        (loss, diagnostics), next_state = loss_fn.apply(
            params, state, jax.random.PRNGKey(0), model_config, task_config,
            i, t, f)
        return loss, (diagnostics, next_state)

    (loss, (diagnostics, next_state)), grads = jax.value_and_grad(
        _aux, has_aux=True)(params, state, inputs, targets, forcings)
    return loss, diagnostics, next_state, grads


# Jax 似乎不喜欢通过 jit 将配置作为参数传递
# 通过部分传递（而不是通过闭包捕获）会迫使 jax 在更改配置时使 jit 缓存失效

# 这段代码实现了一些函数的包装和 JIT 编译 以优化 JAX 中的性能和缓存管理
# 它主要针对之前定义的模型构建、损失函数和梯度计算函数

# 提前绑定参数
# 如果配置发生变化 JAX 将使 JIT 缓存失效 重新编译函数
def with_configs(fn, model_config, task_config, diffs_stddev_by_level, mean_by_level, stddev_by_level):
    return functools.partial(
        fn, model_config=model_config, task_config=task_config, diffs_stddev_by_level=diffs_stddev_by_level,
        mean_by_level=mean_by_level, stddev_by_level=stddev_by_level)


# 预先绑定 简化函数调用
def with_params(fn, params, state):
    return functools.partial(fn, params=params, state=state)


# 由于模型是无状态的 该函数仅返回预测结果 忽略状态
# 这对于滚动代码和一般简化很有用
def drop_state(fn):
    return lambda **kw: fn(**kw)[0]


def parse_gpu_memory(s):
    s = s.replace(' ', '').replace('MiB', '')
    return int(s)


def select_best_gpu():
    cmd = "nvidia-smi --query-gpu=index,memory.used,memory.free,memory.total --format=csv,noheader"
    output = subprocess.check_output(cmd, shell=True).decode("utf-8").strip()
    gpu_info = []
    for line in output.split("\n"):
        index, used, free, total = map(parse_gpu_memory, line.split(","))
        gpu_info.append((index, free))
    gpu_info.sort(key=lambda x: x[0], reverse=True)  # sort by index in reverse order (from last to first GPU)
    gpu_info.sort(key=lambda x: x[1], reverse=True)  # sort by free memory in descending order
    best_gpu = gpu_info[0][0]
    print(f'now, best GPU is {best_gpu}')
    return best_gpu


def predict(dataset_file_path, eval_steps):
    gpu = select_best_gpu()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    params_file_path = '/data2/wxz/developer/graphcast/params/params_GraphCast_operational - ERA5-HRES 1979-2021 - resolution 0.25 - pressure levels 13 - mesh 2to6 - precipitation output only.npz'
    with open(params_file_path, "rb") as f:
        ckpt = checkpoint.load(f, graphcast.CheckPoint)
    params = ckpt.params
    state = {}
    model_config = ckpt.model_config
    task_config = ckpt.task_config
    # 本地 stats 文件夹路径
    stats_folder_path = '/data2/wxz/developer/graphcast/stats/'
    # 从本地加载三个不同的统计数据集：差异的标准差、平均值和标准差 每个都按层级划分
    # 加载差异的标准差
    with open(f"{stats_folder_path}/diffs_stddev_by_level.nc", "rb") as f:
        diffs_stddev_by_level = xarray.load_dataset(f).compute()
    # 加载平均值
    with open(f"{stats_folder_path}/mean_by_level.nc", "rb") as f:
        mean_by_level = xarray.load_dataset(f).compute()
    # 加载标准差
    with open(f"{stats_folder_path}/stddev_by_level.nc", "rb") as f:
        stddev_by_level = xarray.load_dataset(f).compute()
    # 加载数据
    example_batch = xarray.open_dataset(dataset_file_path, engine='netcdf4').compute()
    # 从加载的 example_batch 数据集中提取训练和评估数据
    # 提取评估数据
    eval_inputs, eval_targets, eval_forcings = data_utils.extract_inputs_targets_forcings(
        example_batch, target_lead_times=slice("6h", f"{eval_steps * 6}h"),
        **dataclasses.asdict(task_config))

    # 打印数据维度信息
    print("All Examples:  ", example_batch.dims.mapping)
    print("Eval Inputs:   ", eval_inputs.dims.mapping)
    print("Eval Targets:  ", eval_targets.dims.mapping)
    print("Eval Forcings: ", eval_forcings.dims.mapping)

    run_forward_jitted = drop_state(with_params(jax.jit(with_configs(
        run_forward.apply, model_config, task_config, diffs_stddev_by_level, mean_by_level, stddev_by_level)), params,
        state))

    # 打印输入、目标和强迫数据的维度信息
    print("Inputs:  ", eval_inputs.dims.mapping)
    print("Targets: ", eval_targets.dims.mapping)
    print("Forcings:", eval_forcings.dims.mapping)

    # 执行自回归预测
    predictions = rollout.chunked_prediction(
        # 进行预测 这个函数接受 JIT 编译后的 run_forward_jitted 函数作为参数
        run_forward_jitted,
        # 创建了一个随机数生成器的种子
        rng=jax.random.PRNGKey(0),
        # 预测过程的输入、目标模板和强迫数据
        inputs=eval_inputs,
        targets_template=eval_targets * np.nan,
        forcings=eval_forcings)
    return eval_targets, predictions


def generate_perturbation(nc_file):
    ds = xarray.open_dataset(nc_file)
    # 计算并应用扰动
    for var in variables_to_modify:
        if var in ds.data_vars:
            # 只对具有batch, time, level, lat, lon这些维度的变量进行操作
            data = ds[var].sel(lat=slice(0, 50), lon=slice(100, 150), time=ds.time[:2])
            perturbation = np.zeros_like((data))
            for time_step in range(2):
                if len(data.shape) == 5:  # 检查是否为三维变量
                    # 遍历所有高度层
                    for lvl in range(data.shape[2]):
                        # 计算当前层的平均值
                        layer_range = data.isel(level=lvl).max() - data.isel(level=lvl).min()
                        # 生成针对该层的扰动，扰动的标准差是层平均值的千分之一
                        layer_perturbation = np.random.normal(loc=0, scale=layer_range / 200, size=data.shape[3:])
                        # 将扰动应用到相应的层
                        perturbation[0, time_step, lvl, :, :] = layer_perturbation
                if len(data.shape) == 4:  # 检查是否为二维变量
                    all_range = data.max() - data.min()
                    # 生成扰动值
                    perturbation[0, time_step, :, :] = np.random.normal(loc=0, scale=all_range / 200, size=data.shape[2:])
                perturbation = np.nan_to_num(perturbation)
            # 应用扰动
            perturbed_data = data + perturbation
            ds[var].loc[dict(time=ds.time[:2], lat=slice(0, 50), lon=slice(100, 150))] = perturbed_data
    return ds


def judge_constrain(perturbation, beta=0.3):
    grid_num = lon_grid * lat_grid * level_grid
    limit = np.sqrt(beta * grid_num)
    value = np.linalg.norm(perturbation)
    if value < limit:
        print(f'约束为：{limit}，值为：{value}， 满足约束')
    else:
        print(f'约束为：{limit}，值为：{value}， 违反约束')
        ratio = limit / value
        perturbation = perturbation * ratio
    return perturbation

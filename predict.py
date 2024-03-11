import dataclasses
import datetime
import functools
import math
import os
import subprocess
from typing import Optional

from CNOP.parameters import eval_step_dictionary
from graphcast import autoregressive
from graphcast import casting
from graphcast import checkpoint
from graphcast import data_utils
from graphcast import graphcast
from graphcast import normalization
from graphcast import rollout
from graphcast import xarray_jax
from graphcast import xarray_tree
from IPython.display import HTML
import haiku as hk
import jax
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import xarray


# 构建和包装 GraphCast 预测器
def construct_wrapped_graphcast(
        #  接受 model_config（模型配置）和 task_config（任务配置）作为参数
        model_config: graphcast.ModelConfig,
        task_config: graphcast.TaskConfig):
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
def run_forward(model_config, task_config, inputs, targets_template, forcings):
    predictor = construct_wrapped_graphcast(model_config, task_config)
    return predictor(inputs, targets_template=targets_template, forcings=forcings)


# 损失计算 返回计算的损失和诊断信息
@hk.transform_with_state
def loss_fn(model_config, task_config, inputs, targets, forcings):
    predictor = construct_wrapped_graphcast(model_config, task_config)
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
def with_configs(fn):
    return functools.partial(
        fn, model_config=model_config, task_config=task_config)


# 预先绑定 简化函数调用
def with_params(fn):
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


if __name__ == '__main__':
    gpu = select_best_gpu()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    for typhoon_name in eval_step_dictionary.keys():
        print(typhoon_name)
        # 加载模型
        eval_steps = eval_step_dictionary[typhoon_name]
        # dataset文件路径
        dataset_file_path = f'./dataset/{typhoon_name}.nc'
        # dataset_file_path = './dataset/source-era5_date-2022-01-01_res-0.25_levels-13_steps-12.nc'
        params_file_path = './params/params_GraphCast_operational - ERA5-HRES 1979-2021 - resolution 0.25 - pressure levels 13 - mesh 2to6 - precipitation output only.npz'

        with open(params_file_path, "rb") as f:
            ckpt = checkpoint.load(f, graphcast.CheckPoint)
        params = ckpt.params
        state = {}
        model_config = ckpt.model_config
        task_config = ckpt.task_config
        print("Model description:\n", ckpt.description, "\n")
        print("Model license:\n", ckpt.license, "\n")
        # 本地 stats 文件夹路径
        stats_folder_path = './stats'
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

        # 确保所选数据集文件对给定的模型和任务配置有效
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
            run_forward.apply))))

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
        data = {
            # “目标”、“预测”和“差异”（目标和预测之间的差异）
            "Targets": eval_targets,
            "Predictions": predictions,
            "Diff": eval_targets - predictions,
        }
        print(data["Diff"])
        # 存储和显示预测结果
        predictions.to_netcdf(f'./forecast/predictions_{typhoon_name}.nc')

        print(typhoon_name + 'over')

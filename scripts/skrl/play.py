# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
使用 skrl 播放强化学习代理检查点的脚本。

该脚本提供了完整的模型播放功能，支持多种 RL 算法和机器学习框架。
访问 skrl 文档 (https://skrl.readthedocs.io) 查看更多结构化的示例。
"""

"""首先启动 Isaac Sim 模拟器。"""

import argparse

from isaaclab.app import AppLauncher

# 添加命令行参数解析器
# 用于配置播放过程的各种参数和选项
parser = argparse.ArgumentParser(description="播放 skrl 训练的 RL 代理检查点。")

# 视频录制相关参数
parser.add_argument("--video", action="store_true", default=False, 
                   help="播放过程中录制视频。")
parser.add_argument("--video_length", type=int, default=200, 
                   help="录制视频的长度（以仿真步数为单位）。")

# 控制和数据处理参数
parser.add_argument("--control_mode", type=str, default="ACCBR", 
                   help="代理的控制模式。")
parser.add_argument("--save_plots", action="store_true", default=False, 
                   help="播放过程中保存数据图表。")

# 系统配置参数
parser.add_argument("--disable_fabric", action="store_true", default=False, 
                   help="禁用 fabric 并使用 USD I/O 操作。")
parser.add_argument("--seed", type=int, default=None, 
                   help="用于环境的随机种子，确保实验可重现。")
parser.add_argument("--num_envs", type=int, default=None, 
                   help="要模拟的环境数量。")
parser.add_argument("--task", type=str, default=None, 
                   help="任务名称，用于指定要播放的具体任务。")

# 模型和框架配置参数
parser.add_argument("--checkpoint", type=str, default=None, 
                   help="模型检查点的文件路径。")
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="用于训练 skrl 代理的机器学习框架。"
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["PPO", "IPPO", "MAPPO"],
    help="用于训练 skrl 代理的 RL 算法。"
)

# 实时运行参数
parser.add_argument("--real-time", action="store_true", default=False, 
                   help="尽可能以实时速度运行。")

# 添加 AppLauncher 的命令行参数
# 这些参数用于配置 Isaac Sim 模拟器
AppLauncher.add_app_launcher_args(parser)

# 解析所有命令行参数
args_cli = parser.parse_args()

# 如果启用了视频录制，自动启用摄像头功能
if args_cli.video:
    args_cli.enable_cameras = True

# 启动 Omniverse 应用程序
# AppLauncher 负责初始化 Isaac Sim 模拟器环境
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""以下为播放的主要逻辑部分。"""

import gymnasium as gym
import os
import random
import time
import torch

import skrl
from packaging import version

# 导入自定义绘图工具
from MARL_mav_carry_ext.plotting_tools import DirectMARLPlotter

# 注册 gym 环境（虽然代码中有注释，但实际可能在其他地方完成）

# 检查 skrl 版本兼容性
# 确保使用的 skrl 版本不低于最低要求版本
SKRL_VERSION = "1.4.1"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"不支持的 skrl 版本: {skrl.__version__}。"
        f"请使用 'pip install skrl>={SKRL_VERSION}' 安装支持的版本"
    )
    exit()

# 根据指定的机器学习框架导入对应的 Runner 类
# Runner 是 skrl 提供的训练/播放执行器
if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

# 导入自定义的环境包装器
from isaaclab_rl.skrl import SkrlVecEnvWrapper

# 导入 IsaacLab 任务模块
import isaaclab_tasks  # noqa: F401

# 导入环境和工具类
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent  # 多智能体环境和转换工具
from isaaclab.utils.dict import print_dict      # 字典打印工具
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg  # 配置工具

# 配置快捷变量
algorithm = args_cli.algorithm.lower()


def main():
    """
    使用 skrl 代理进行播放的主要函数。
    
    该函数负责加载训练好的模型检查点，在环境中执行推理，
    支持实时播放、视频录制和数据可视化功能。
    """
    
    # 配置机器学习框架到全局 skrl 变量
    # JAX 后端可以选择纯 JAX 或 NumPy 模式
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    # 解析环境配置
    # 根据命令行参数构建环境配置对象
    env_cfg = parse_env_cfg(
        args_cli.task, 
        device=args_cli.device, 
        num_envs=args_cli.num_envs, 
        use_fabric=not args_cli.disable_fabric
    )
    
    # 尝试加载实验配置
    # 优先尝试特定算法的配置，如果不存在则使用通用配置
    try:
        experiment_cfg = load_cfg_from_registry(args_cli.task, f"skrl_{algorithm}_cfg_entry_point")
    except ValueError:
        experiment_cfg = load_cfg_from_registry(args_cli.task, "skrl_cfg_entry_point")

    # 随机种子处理
    # 当种子设为 -1 时，随机生成一个种子
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # 设置代理和环境的随机种子
    # 注意：环境初始化过程中会发生一些随机化，所以在这里设置种子很重要
    experiment_cfg["seed"] = args_cli.seed if args_cli.seed is not None else experiment_cfg["seed"]
    env_cfg.seed = experiment_cfg["seed"]

    # 配置实验日志目录（用于加载检查点）
    log_root_path = os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] 正在从目录加载实验: {log_root_path}")
    
    # 获取检查点路径
    if args_cli.checkpoint:
        # 如果指定了具体检查点路径，直接使用
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        # 自动查找最新的检查点文件
        resume_path = get_checkpoint_path(
            log_root_path, 
            run_dir=f".*_{algorithm}_{args_cli.ml_framework}", 
            other_dirs=["checkpoints"]
        )
    log_dir = os.path.dirname(os.path.dirname(resume_path))

    # 创建 Isaac 环境实例
    # 使用 Gymnasium 接口创建环境
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    
    # 视频录制包装器配置
    if args_cli.video:
        # 配置视频录制参数
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),  # 视频保存目录
            "step_trigger": lambda step: step == 0,  # 在第一步触发录制
            "video_length": args_cli.video_length,  # 视频长度
            "disable_logger": True,  # 禁用日志记录以避免冲突
        }
        print("[INFO] 播放过程中将录制视频。")
        print_dict(video_kwargs, nesting=4)  # 打印视频配置详情
        env = gym.wrappers.RecordVideo(env, **video_kwargs)  # 包装环境以支持视频录制

    # 多智能体到单智能体转换
    # 某些算法（如 PPO）需要单智能体环境
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env, state_as_observation=True)

    # 获取环境（物理引擎）的时间步长用于实时评估
    try:
        dt = env.step_dt
    except AttributeError:
        dt = env.unwrapped.step_dt

    # 使用 skrl 环境包装器
    # 该包装器适配了 skrl 框架的要求
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)  # 等同于: `wrap_env(env, wrapper="auto")`

    # 配置并实例化 skrl 运行器
    # https://skrl.readthedocs.io/en/latest/api/utils/runner.html
    experiment_cfg["trainer"]["close_environment_at_exit"] = False  # 配置训练器在退出时不关闭环境
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0  # 不记录到 TensorBoard
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0  # 不生成检查点
    runner = Runner(env, experiment_cfg)  # 创建运行器实例

    print(f"[INFO] 正在从以下位置加载模型检查点: {resume_path}")
    runner.agent.load(resume_path)  # 加载模型检查点
    runner.agent.set_running_mode("eval")  # 将代理设置为评估模式
    
    # 创建数据绘图器实例
    plotter = DirectMARLPlotter(env, control_mode=args_cli.control_mode)

    # 重置环境并获取初始观测
    obs, _ = env.reset()
    timestep = 0

    # 开始环境仿真循环
    while simulation_app.is_running():
        # 记录开始时间用于实时控制
        start_time = time.time()
        
        # 在推理模式下运行所有操作
        with torch.inference_mode():
            # 代理执行动作
            outputs = runner.agent.act(obs, timestep=0, timesteps=0)
            
            # 处理多智能体动作（确定性动作）
            if hasattr(env, "possible_agents"):
                actions = {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in env.possible_agents}
            # 处理单智能体动作（确定性动作）
            else:
                actions = outputs[-1].get("mean_actions", outputs[0])
            
            # 环境执行一步
            if env.num_envs == 1:
                plotter.collect_data()  # 收集数据用于绘图
            obs, _, _, _, _ = env.step(actions)

        timestep += 1
        
        # 视频录制控制逻辑
        if args_cli.video:
            # 当达到指定视频长度时退出播放循环
            if timestep == args_cli.video_length:
                break

        # 实时运行控制
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)  # 睡眠以保持实时速度

    # 关闭环境
    env.close()

    # 数据可视化处理
    if env.num_envs == 1:
        if args_cli.save_plots:
            # 保存图表到文件
            plot_path = os.path.join(log_dir, "plots", "play")
            plotter.plot(save=True, save_dir=plot_path)
        else:
            # 显示图表（交互式查看）
            plotter.plot(save=False)


if __name__ == "__main__":
    # 执行主函数
    main()
    # 关闭模拟器应用程序
    simulation_app.close()
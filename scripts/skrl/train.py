# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
使用 skrl 训练强化学习代理的脚本。

该脚本提供了一个完整的训练流水线，支持多种 RL 算法和机器学习框架。
访问 skrl 文档 (https://skrl.readthedocs.io) 查看更多结构化的示例。
"""

"""首先启动 Isaac Sim 模拟器。"""

import argparse
import sys

from isaaclab.app import AppLauncher

# 添加命令行参数解析器
# 用于配置训练的各种参数和选项
parser = argparse.ArgumentParser(description="使用 skrl 训练 RL 代理。")

# 视频录制相关参数
parser.add_argument("--video", action="store_true", default=False, 
                   help="训练过程中录制视频。")
parser.add_argument("--video_length", type=int, default=200, 
                   help="录制视频的长度（以仿真步数为单位）。")
parser.add_argument("--video_interval", type=int, default=2000, 
                   help="视频录制间隔（以仿真步数为单位）。")

# 环境配置参数
parser.add_argument("--num_envs", type=int, default=None, 
                   help="要模拟的环境数量。")
parser.add_argument("--task", type=str, default=None, 
                   help="任务名称，用于指定要训练的具体任务。")

# 随机种子和分布式训练参数
parser.add_argument("--seed", type=int, default=None, 
                   help="用于环境的随机种子，确保实验可重现。")
parser.add_argument("--distributed", action="store_true", default=False, 
                   help="启用分布式训练，支持多 GPU 或多节点训练。")

# 训练控制参数
parser.add_argument("--max_iterations", type=int, default=None, 
                   help="RL 策略的最大训练迭代次数。")
parser.add_argument("--resume", action="store_true", default=False, 
                   help="从检查点恢复训练。")
parser.add_argument("--checkpoint", type=str, default=None, 
                   help="要恢复训练的检查点文件路径。")

# 机器学习框架和算法选择参数
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

# 添加 AppLauncher 的命令行参数
# 这些参数用于配置 Isaac Sim 模拟器
AppLauncher.add_app_launcher_args(parser)

# 解析已知参数，保留未知参数供 Hydra 处理
args_cli, hydra_args = parser.parse_known_args()

# 如果启用了视频录制，自动启用摄像头功能
if args_cli.video:
    args_cli.enable_cameras = True

# 清理 sys.argv 以便 Hydra 能正确处理剩余参数
sys.argv = [sys.argv[0]] + hydra_args

# 启动 Omniverse 应用程序
# AppLauncher 负责初始化 Isaac Sim 模拟器环境
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""以下为训练的主要逻辑部分。"""

import gymnasium as gym
import os
import random
from datetime import datetime

import skrl
from packaging import version

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
# Runner 是 skrl 提供的训练执行器
if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

# 导入自定义的环境包装器
from isaaclab_rl.skrl import SkrlVecEnvWrapper

# 导入项目相关的任务模块
import MARL_mav_carry_ext.tasks  # noqa: F401

# 导入 IsaacLab 任务模块
import isaaclab_tasks  # noqa: F401

# 导入环境配置和工具类
from isaaclab.envs import (
    DirectMARLEnv,           # 直接多智能体 RL 环境基类
    DirectMARLEnvCfg,        # 直接多智能体环境配置
    DirectRLEnvCfg,          # 直接 RL 环境配置
    ManagerBasedRLEnvCfg,    # 基于管理器的 RL 环境配置
    multi_agent_to_single_agent,  # 多智能体到单智能体的转换工具
)
from isaaclab.utils.dict import print_dict      # 字典打印工具
from isaaclab.utils.io.pkl import dump_pickle   # pickle 序列化工具
from isaaclab.utils.io.yaml import dump_yaml    # YAML 配置导出工具
from isaaclab_tasks.utils import get_checkpoint_path  # 检查点路径获取工具
from isaaclab_tasks.utils.hydra import hydra_task_config  # Hydra 配置装饰器

# 配置快捷变量
# 根据算法类型确定配置入口点
algorithm = args_cli.algorithm.lower()
agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """
    主训练函数，使用 skrl 代理进行训练。
    
    参数:
        env_cfg: 环境配置对象，包含场景、仿真等设置
        agent_cfg: 代理配置字典，包含网络结构、超参数等设置
    """
    
    # 使用命令行参数覆盖配置
    # 优先级：命令行参数 > 配置文件参数
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # 分布式训练配置
    # 为多 GPU 训练设置正确的设备标识
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
    
    # 设置最大训练步数
    # 总步数 = 迭代次数 × 每次 rollout 的步数
    if args_cli.max_iterations:
        agent_cfg["trainer"]["timesteps"] = args_cli.max_iterations * agent_cfg["agent"]["rollouts"]
    
    # 配置训练器在退出时不关闭环境
    # 这样可以确保环境资源被正确清理
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    
    # 配置机器学习框架到全局 skrl 变量
    # JAX 后端可以选择纯 JAX 或 NumPy 模式
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    # 随机种子处理
    # 当种子设为 -1 时，随机生成一个种子
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # 设置代理和环境的随机种子
    # 注意：环境初始化过程中会发生一些随机化，所以在这里设置种子很重要
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    env_cfg.seed = agent_cfg["seed"]

    # 配置实验日志目录
    # 构建日志存储路径结构
    log_root_path = os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] 实验日志将保存到目录: {log_root_path}")
    
    # 构造具体的运行日志目录名
    # 格式：时间戳_算法名_框架名[_实验名]
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{algorithm}_{args_cli.ml_framework}"
    if agent_cfg["agent"]["experiment"]["experiment_name"]:
        log_dir += f'_{agent_cfg["agent"]["experiment"]["experiment_name"]}'
    
    # 将目录信息更新到代理配置中
    agent_cfg["agent"]["experiment"]["directory"] = log_root_path
    agent_cfg["agent"]["experiment"]["experiment_name"] = log_dir
    
    # 构建完整的日志目录路径
    log_dir = os.path.join(log_root_path, log_dir)

    # 处理检查点加载逻辑
    if args_cli.resume:
        # 如果指定了具体检查点路径，直接使用
        if args_cli.checkpoint:
            resume_path = args_cli.checkpoint
        else:
            # 自动查找最新的检查点文件
            resume_path = get_checkpoint_path(
                log_root_path, 
                run_dir=f".*_{algorithm}_{args_cli.ml_framework}", 
                other_dirs=["checkpoints"]
            )

        print(f"[INFO] 正在从检查点加载模型: {resume_path}")

    # 导出配置文件到日志目录
    # 保存环境和代理配置以便复现实验
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # 创建 Isaac 环境实例
    # 使用 Gymnasium 接口创建环境
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    
    # 视频录制包装器配置
    if args_cli.video:
        # 配置视频录制参数
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),  # 视频保存目录
            "step_trigger": lambda step: step % args_cli.video_interval == 0,  # 录制触发条件
            "video_length": args_cli.video_length,  # 每个视频的长度
            "disable_logger": True,  # 禁用日志记录以避免冲突
        }
        print("[INFO] 训练过程中将录制视频。")
        print_dict(video_kwargs, nesting=4)  # 打印视频配置详情
        env = gym.wrappers.RecordVideo(env, **video_kwargs)  # 包装环境以支持视频录制

    # 多智能体到单智能体转换
    # 某些算法（如 PPO）需要单智能体环境
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env, state_as_observation=True)

    # 使用 skrl 环境包装器
    # 该包装器适配了 skrl 框架的要求
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)  # 等同于: `wrap_env(env, wrapper="auto")`

    # 配置并实例化 skrl 训练运行器
    # Runner 负责协调整个训练过程
    # 详细文档: https://skrl.readthedocs.io/en/latest/api/utils/runner.html
    runner = Runner(env, agent_cfg)

    # 如果启用了恢复训练，加载检查点
    if args_cli.resume:
        runner.agent.load(resume_path)

    # 开始训练过程
    runner.run()
    
    # 训练完成后关闭环境
    env.close()


if __name__ == "__main__":
    # 程序入口点
    print("正在运行主训练函数")
    main()
    # 关闭模拟器应用程序
    simulation_app.close()
"""使用 RSL-RL 强化学习代理播放检查点的脚本。"""

"""首先启动 Isaac Sim 模拟器。"""

import argparse

from isaaclab.app import AppLauncher

# 本地导入
import cli_args  # isort: skip

# 添加命令行参数解析器
# 用于配置播放过程的各种参数和选项
parser = argparse.ArgumentParser(description="使用 RSL-RL 训练 RL 代理。")

# 视频录制相关参数
parser.add_argument("--video", action="store_true", default=False, 
                   help="训练过程中录制视频。")
parser.add_argument("--video_length", type=int, default=200, 
                   help="录制视频的长度（以仿真步数为单位）。")

# 系统配置参数
parser.add_argument("--disable_fabric", action="store_true", default=False, 
                   help="禁用 fabric 并使用 USD I/O 操作。")
parser.add_argument("--num_envs", type=int, default=None, 
                   help="要模拟的环境数量。")
parser.add_argument("--task", type=str, default=None, 
                   help="任务名称，用于指定要播放的具体任务。")
parser.add_argument("--seed", type=int, default=None, 
                   help="用于环境的随机种子，确保实验可重现。")
parser.add_argument("--plot_data", type=bool, default=None, 
                   help="是否绘制当前运行的数据图表")

# 添加 RSL-RL 特定的命令行参数
cli_args.add_rsl_rl_args(parser)

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
import matplotlib.pyplot as plt
import numpy as np
import os
import torch

# 导入 RSL-RL 的训练运行器
from rsl_rl.runners import OnPolicyRunner

# 导入扩展模块以设置环境任务
import MARL_mav_carry_ext.tasks  # noqa: F401

# 导入工具函数
from isaaclab.utils.dict import print_dict      # 字典打印工具
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg  # 路径和配置解析工具
from isaaclab_tasks.utils.wrappers.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_onnx  # RSL-RL 包装器


def main():
    """
    使用 RSL-RL 代理进行播放的主要函数。
    
    该函数负责加载训练好的模型检查点，并在环境中执行推理，
    同时支持数据可视化和策略导出功能。
    """
    
    # 解析环境配置
    # 根据命令行参数构建环境配置对象
    env_cfg = parse_env_cfg(
        args_cli.task, 
        device=args_cli.device, 
        num_envs=args_cli.num_envs, 
        use_fabric=not args_cli.disable_fabric
    )
    
    # 解析代理配置
    # 获取 RSL-RL 训练运行器的配置
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # 配置实验日志目录
    # 构建日志存储路径结构
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] 正在从目录加载实验: {log_root_path}")
    
    # 获取检查点路径
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)

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
    
    # 使用 RSL-RL 环境包装器
    # 该包装器适配了 RSL-RL 框架的要求
    env = RslRlVecEnvWrapper(env)

    print(f"[INFO]: 正在从以下位置加载模型检查点: {resume_path}")

    # 加载预训练的模型
    # 创建 OnPolicyRunner 实例并加载检查点
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # 获取用于推理的训练策略
    # 将策略移动到环境所在的设备上
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # 将策略导出为 ONNX 格式
    # 便于部署和跨平台使用
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_onnx(ppo_runner.alg.actor_critic, export_model_dir, filename="policy.onnx")

    # 重置环境并获取初始观测
    obs, _ = env.get_observations()
    timestep = 0

    # 初始化数据收集列表（用于绘图）
    # 无人机推力数据
    drone_1_forces = []
    drone_2_forces = []
    drone_3_forces = []
    
    # 无人机扭矩数据
    drone_1_x_torque = []
    drone_1_y_torque = []
    drone_1_z_torque = []
    drone_2_x_torque = []
    drone_2_y_torque = []
    drone_2_z_torque = []
    drone_3_x_torque = []
    drone_3_y_torque = []
    drone_3_z_torque = []

    # 载荷位置数据
    payload_pos_x = []
    payload_pos_y = []
    payload_pos_z = []
    
    # 载荷姿态数据
    payload_quat_w = []
    payload_quat_x = []
    payload_quat_y = []
    payload_quat_z = []
    
    # 载荷线速度数据
    payload_lin_vel_x = []
    payload_lin_vel_y = []
    payload_lin_vel_z = []
    
    # 载荷角速度数据
    payload_ang_vel_x = []
    payload_ang_vel_y = []
    payload_ang_vel_z = []

    # 无人机1位置数据
    drone_1_pos_x = []
    drone_1_pos_y = []
    drone_1_pos_z = []
    
    # 无人机1姿态数据
    drone_1_quat_w = []
    drone_1_quat_x = []
    drone_1_quat_y = []
    drone_1_quat_z = []
    
    # 无人机1线速度数据
    drone_1_lin_vel_x = []
    drone_1_lin_vel_y = []
    drone_1_lin_vel_z = []
    
    # 无人机1角速度数据
    drone_1_ang_vel_x = []
    drone_1_ang_vel_y = []
    drone_1_ang_vel_z = []

    # 无人机2位置和姿态数据
    drone_2_pos_x = []
    drone_2_pos_y = []
    drone_2_pos_z = []
    drone_2_quat_w = []
    drone_2_quat_x = []
    drone_2_quat_y = []
    drone_2_quat_z = []
    drone_2_lin_vel_x = []
    drone_2_lin_vel_y = []
    drone_2_lin_vel_z = []
    drone_2_ang_vel_x = []
    drone_2_ang_vel_y = []
    drone_2_ang_vel_z = []

    # 无人机3位置和姿态数据
    drone_3_pos_x = []
    drone_3_pos_y = []
    drone_3_pos_z = []
    drone_3_quat_w = []
    drone_3_quat_x = []
    drone_3_quat_y = []
    drone_3_quat_z = []
    drone_3_lin_vel_x = []
    drone_3_lin_vel_y = []
    drone_3_lin_vel_z = []
    drone_3_ang_vel_x = []
    drone_3_ang_vel_y = []
    drone_3_ang_vel_z = []

    # 缆绳角度数据
    cable_angle_1_w = []
    cable_angle_1_x = []
    cable_angle_1_y = []
    cable_angle_1_z = []

    cable_angle_2_w = []
    cable_angle_2_x = []
    cable_angle_2_y = []
    cable_angle_2_z = []

    cable_angle_3_w = []
    cable_angle_3_x = []
    cable_angle_3_y = []
    cable_angle_3_z = []

    # 载荷误差数据
    payload_pos_error_x = []
    payload_pos_error_y = []
    payload_pos_error_z = []
    payload_quat_error_w = []
    payload_quat_error_x = []
    payload_quat_error_y = []
    payload_quat_error_z = []

    # 开始环境仿真循环
    while simulation_app.is_running():
        # 在推理模式下运行所有操作
        with torch.inference_mode():
            # 代理执行动作
            actions = policy(obs)
            
            # 环境执行一步
            obs, rewards, dones, _ = env.step(actions)
            timestep += 1
            
            # 如果启用了数据绘图功能
            if args_cli.plot_data:
                # 收集动作数据
                drone_1_forces.append(actions[:, 0].cpu().numpy())
                drone_2_forces.append(actions[:, 4].cpu().numpy())
                drone_3_forces.append(actions[:, 8].cpu().numpy())
                
                # 收集扭矩数据
                drone_1_x_torque.append(actions[:, 1].cpu().numpy())
                drone_1_y_torque.append(actions[:, 2].cpu().numpy())
                drone_1_z_torque.append(actions[:, 3].cpu().numpy())
                drone_2_x_torque.append(actions[:, 5].cpu().numpy())
                drone_2_y_torque.append(actions[:, 6].cpu().numpy())
                drone_2_z_torque.append(actions[:, 7].cpu().numpy())
                drone_3_x_torque.append(actions[:, 9].cpu().numpy())
                drone_3_y_torque.append(actions[:, 10].cpu().numpy())
                drone_3_z_torque.append(actions[:, 11].cpu().numpy())

                # 收集载荷观测数据
                payload_pos_x.append(obs[:, 0].cpu().numpy())
                payload_pos_y.append(obs[:, 1].cpu().numpy())
                payload_pos_z.append(obs[:, 2].cpu().numpy())
                payload_quat_w.append(obs[:, 3].cpu().numpy())
                payload_quat_x.append(obs[:, 4].cpu().numpy())
                payload_quat_y.append(obs[:, 5].cpu().numpy())
                payload_quat_z.append(obs[:, 6].cpu().numpy())
                payload_lin_vel_x.append(obs[:, 7].cpu().numpy())
                payload_lin_vel_y.append(obs[:, 8].cpu().numpy())
                payload_lin_vel_z.append(obs[:, 9].cpu().numpy())
                payload_ang_vel_x.append(obs[:, 10].cpu().numpy())
                payload_ang_vel_y.append(obs[:, 11].cpu().numpy())
                payload_ang_vel_z.append(obs[:, 12].cpu().numpy())

                # 收集无人机位置数据
                drone_1_pos_x.append(obs[:, 13].cpu().numpy())
                drone_1_pos_y.append(obs[:, 14].cpu().numpy())
                drone_1_pos_z.append(obs[:, 15].cpu().numpy())
                drone_2_pos_x.append(obs[:, 16].cpu().numpy())
                drone_2_pos_y.append(obs[:, 17].cpu().numpy())
                drone_2_pos_z.append(obs[:, 18].cpu().numpy())
                drone_3_pos_x.append(obs[:, 19].cpu().numpy())
                drone_3_pos_y.append(obs[:, 20].cpu().numpy())
                drone_3_pos_z.append(obs[:, 21].cpu().numpy())

                # 收集无人机姿态数据
                drone_1_quat_w.append(obs[:, 22].cpu().numpy())
                drone_1_quat_x.append(obs[:, 23].cpu().numpy())
                drone_1_quat_y.append(obs[:, 24].cpu().numpy())
                drone_1_quat_z.append(obs[:, 25].cpu().numpy())
                drone_2_quat_w.append(obs[:, 26].cpu().numpy())
                drone_2_quat_x.append(obs[:, 27].cpu().numpy())
                drone_2_quat_y.append(obs[:, 28].cpu().numpy())
                drone_2_quat_z.append(obs[:, 29].cpu().numpy())
                drone_3_quat_w.append(obs[:, 30].cpu().numpy())
                drone_3_quat_x.append(obs[:, 31].cpu().numpy())
                drone_3_quat_y.append(obs[:, 32].cpu().numpy())
                drone_3_quat_z.append(obs[:, 33].cpu().numpy())

                # 收集无人机线速度数据
                drone_1_lin_vel_x.append(obs[:, 34].cpu().numpy())
                drone_1_lin_vel_y.append(obs[:, 35].cpu().numpy())
                drone_1_lin_vel_z.append(obs[:, 36].cpu().numpy())
                drone_2_lin_vel_x.append(obs[:, 37].cpu().numpy())
                drone_2_lin_vel_y.append(obs[:, 38].cpu().numpy())
                drone_2_lin_vel_z.append(obs[:, 39].cpu().numpy())
                drone_3_lin_vel_x.append(obs[:, 40].cpu().numpy())
                drone_3_lin_vel_y.append(obs[:, 41].cpu().numpy())
                drone_3_lin_vel_z.append(obs[:, 42].cpu().numpy())

                # 收集无人机角速度数据
                drone_1_ang_vel_x.append(obs[:, 43].cpu().numpy())
                drone_1_ang_vel_y.append(obs[:, 44].cpu().numpy())
                drone_1_ang_vel_z.append(obs[:, 45].cpu().numpy())
                drone_2_ang_vel_x.append(obs[:, 46].cpu().numpy())
                drone_2_ang_vel_y.append(obs[:, 47].cpu().numpy())
                drone_2_ang_vel_z.append(obs[:, 48].cpu().numpy())
                drone_3_ang_vel_x.append(obs[:, 49].cpu().numpy())
                drone_3_ang_vel_y.append(obs[:, 50].cpu().numpy())
                drone_3_ang_vel_z.append(obs[:, 51].cpu().numpy())

                # 收集缆绳角度数据
                cable_angle_1_w.append(obs[:, 92].cpu().numpy())
                cable_angle_1_x.append(obs[:, 93].cpu().numpy())
                cable_angle_1_y.append(obs[:, 94].cpu().numpy())
                cable_angle_1_z.append(obs[:, 95].cpu().numpy())

                cable_angle_2_w.append(obs[:, 96].cpu().numpy())
                cable_angle_2_x.append(obs[:, 97].cpu().numpy())
                cable_angle_2_y.append(obs[:, 98].cpu().numpy())
                cable_angle_2_z.append(obs[:, 99].cpu().numpy())

                cable_angle_3_w.append(obs[:, 100].cpu().numpy())
                cable_angle_3_x.append(obs[:, 101].cpu().numpy())
                cable_angle_3_y.append(obs[:, 102].cpu().numpy())
                cable_angle_3_z.append(obs[:, 103].cpu().numpy())

                # 收集载荷误差数据
                payload_pos_error_x.append(obs[:, 52].cpu().numpy())
                payload_pos_error_y.append(obs[:, 53].cpu().numpy())
                payload_pos_error_z.append(obs[:, 54].cpu().numpy())
                payload_quat_error_w.append(obs[:, 55].cpu().numpy())
                payload_quat_error_x.append(obs[:, 56].cpu().numpy())
                payload_quat_error_y.append(obs[:, 57].cpu().numpy())
                payload_quat_error_z.append(obs[:, 58].cpu().numpy())

                # 如果达到终止条件或视频长度限制则退出
                if dones | timestep == args_cli.video_length:
                    break

        # 视频录制控制逻辑
        if args_cli.video:
            # 当达到指定视频长度时退出播放循环
            if timestep == args_cli.video_length:
                break

    # 如果启用了数据绘图功能
    if args_cli.plot_data:
        # 绘制动作数据图表
        plt.figure(figsize=(10, 8))
        plt.subplot(2, 2, 1)
        plt.plot(drone_1_forces, label="无人机 1")
        plt.plot(drone_2_forces, label="无人机 2")
        plt.plot(drone_3_forces, label="无人机 3")
        plt.title("无人机推力")
        plt.legend()

        plt.subplot(2, 2, 2)
        plt.plot(drone_1_x_torque, label="无人机 1 x 扭矩")
        plt.plot(drone_1_y_torque, label="无人机 1 y 扭矩")
        plt.plot(drone_1_z_torque, label="无人机 1 z 扭矩")
        plt.title("无人机 1 扭矩")
        plt.legend()

        plt.subplot(2, 2, 3)
        plt.plot(drone_2_x_torque, label="无人机 2 x 扭矩")
        plt.plot(drone_2_y_torque, label="无人机 2 y 扭矩")
        plt.plot(drone_2_z_torque, label="无人机 2 z 扭矩")
        plt.title("无人机 2 扭矩")
        plt.legend()

        plt.subplot(2, 2, 4)
        plt.plot(drone_3_x_torque, label="无人机 3 x 扭矩")
        plt.plot(drone_3_y_torque, label="无人机 3 y 扭矩")
        plt.plot(drone_3_z_torque, label="无人机 3 z 扭矩")
        plt.title("无人机 3 扭矩")
        plt.legend()

        # 绘制裁剪后的动作数据图表
        plt.figure(figsize=(10, 8))
        plt.subplot(2, 2, 1)
        plt.plot(np.clip(drone_1_forces, 0, 25), label="无人机 1")
        plt.plot(np.clip(drone_2_forces, 0, 25), label="无人机 2")
        plt.plot(np.clip(drone_3_forces, 0, 25), label="无人机 3")
        plt.title("无人机推力（裁剪后）")
        plt.legend()

        plt.subplot(2, 2, 2)
        plt.plot(np.clip(drone_1_x_torque, -0.05, 0.05), label="无人机 1 x 扭矩")
        plt.plot(np.clip(drone_1_y_torque, -0.05, 0.05), label="无人机 1 y 扭矩")
        plt.plot(np.clip(drone_1_z_torque, -0.05, 0.05), label="无人机 1 z 扭矩")
        plt.title("无人机 1 扭矩（裁剪后）")
        plt.legend()

        plt.subplot(2, 2, 3)
        plt.plot(np.clip(drone_2_x_torque, -0.05, 0.05), label="无人机 2 x 扭矩")
        plt.plot(np.clip(drone_2_y_torque, -0.05, 0.05), label="无人机 2 y 扭矩")
        plt.plot(np.clip(drone_2_z_torque, -0.05, 0.05), label="无人机 2 z 扭矩")
        plt.title("无人机 2 扭矩（裁剪后）")
        plt.legend()

        plt.subplot(2, 2, 4)
        plt.plot(np.clip(drone_3_x_torque, -0.05, 0.05), label="无人机 3 x 扭矩")
        plt.plot(np.clip(drone_3_y_torque, -0.05, 0.05), label="无人机 3 y 扭矩")
        plt.plot(np.clip(drone_3_z_torque, -0.05, 0.05), label="无人机 3 z 扭矩")
        plt.title("无人机 3 扭矩（裁剪后）")
        plt.legend()

        # 绘制载荷位置数据
        plt.figure(figsize=(10, 8))
        plt.subplot(2, 2, 1)
        plt.plot(payload_pos_x, label="x")
        plt.plot(payload_pos_y, label="y")
        plt.plot(payload_pos_z, label="z")
        plt.title("载荷位置")
        plt.legend()

        # 绘制载荷姿态数据
        plt.subplot(2, 2, 2)
        plt.plot(payload_quat_w, label="w")
        plt.plot(payload_quat_x, label="x")
        plt.plot(payload_quat_y, label="y")
        plt.plot(payload_quat_z, label="z")
        plt.title("载荷姿态")
        plt.legend()

        # 绘制载荷线速度数据
        plt.subplot(2, 2, 3)
        plt.plot(payload_lin_vel_x, label="x")
        plt.plot(payload_lin_vel_y, label="y")
        plt.plot(payload_lin_vel_z, label="z")
        plt.title("载荷线速度")
        plt.legend()

        # 绘制载荷角速度数据
        plt.subplot(2, 2, 4)
        plt.plot(payload_ang_vel_x, label="x")
        plt.plot(payload_ang_vel_y, label="y")
        plt.plot(payload_ang_vel_z, label="z")
        plt.title("载荷角速度")
        plt.legend()

        # 绘制无人机位置数据
        plt.figure(figsize=(10, 8))
        plt.subplot(2, 2, 1)
        plt.plot(drone_1_pos_x, label="无人机 1 x")
        plt.plot(drone_1_pos_y, label="无人机 1 y")
        plt.plot(drone_1_pos_z, label="无人机 1 z")
        plt.title("无人机 1 位置")
        plt.legend()

        plt.subplot(2, 2, 2)
        plt.plot(drone_2_pos_x, label="无人机 2 x")
        plt.plot(drone_2_pos_y, label="无人机 2 y")
        plt.plot(drone_2_pos_z, label="无人机 2 z")
        plt.title("无人机 2 位置")
        plt.legend()

        plt.subplot(2, 2, 3)
        plt.plot(drone_3_pos_x, label="无人机 3 x")
        plt.plot(drone_3_pos_y, label="无人机 3 y")
        plt.plot(drone_3_pos_z, label="无人机 3 z")
        plt.title("无人机 3 位置")
        plt.legend()

        # 绘制无人机姿态数据
        plt.figure(figsize=(10, 8))
        plt.subplot(2, 2, 1)
        plt.plot(drone_1_quat_w, label="无人机 1 w")
        plt.plot(drone_1_quat_x, label="无人机 1 x")
        plt.plot(drone_1_quat_y, label="无人机 1 y")
        plt.plot(drone_1_quat_z, label="无人机 1 z")
        plt.title("无人机 1 姿态")
        plt.legend()

        plt.subplot(2, 2, 2)
        plt.plot(drone_2_quat_w, label="无人机 2 w")
        plt.plot(drone_2_quat_x, label="无人机 2 x")
        plt.plot(drone_2_quat_y, label="无人机 2 y")
        plt.plot(drone_2_quat_z, label="无人机 2 z")
        plt.title("无人机 2 姿态")
        plt.legend()

        plt.subplot(2, 2, 3)
        plt.plot(drone_3_quat_w, label="无人机 3 w")
        plt.plot(drone_3_quat_x, label="无人机 3 x")
        plt.plot(drone_3_quat_y, label="无人机 3 y")
        plt.plot(drone_3_quat_z, label="无人机 3 z")
        plt.title("无人机 3 姿态")
        plt.legend()

        # 绘制无人机线速度数据
        plt.figure(figsize=(10, 8))
        plt.subplot(2, 2, 1)
        plt.plot(drone_1_lin_vel_x, label="无人机 1 x")
        plt.plot(drone_1_lin_vel_y, label="无人机 1 y")
        plt.plot(drone_1_lin_vel_z, label="无人机 1 z")
        plt.title("无人机 1 线速度")
        plt.legend()

        plt.subplot(2, 2, 2)
        plt.plot(drone_2_lin_vel_x, label="无人机 2 x")
        plt.plot(drone_2_lin_vel_y, label="无人机 2 y")
        plt.plot(drone_2_lin_vel_z, label="无人机 2 z")
        plt.title("无人机 2 线速度")
        plt.legend()

        plt.subplot(2, 2, 3)
        plt.plot(drone_3_lin_vel_x, label="无人机 3 x")
        plt.plot(drone_3_lin_vel_y, label="无人机 3 y")
        plt.plot(drone_3_lin_vel_z, label="无人机 3 z")
        plt.title("无人机 3 线速度")
        plt.legend()

        # 绘制无人机角速度数据
        plt.figure(figsize=(10, 8))
        plt.subplot(2, 2, 1)
        plt.plot(drone_1_ang_vel_x, label="无人机 1 x")
        plt.plot(drone_1_ang_vel_y, label="无人机 1 y")
        plt.plot(drone_1_ang_vel_z, label="无人机 1 z")
        plt.title("无人机 1 角速度")
        plt.legend()

        plt.subplot(2, 2, 2)
        plt.plot(drone_2_ang_vel_x, label="无人机 2 x")
        plt.plot(drone_2_ang_vel_y, label="无人机 2 y")
        plt.plot(drone_2_ang_vel_z, label="无人机 2 z")
        plt.title("无人机 2 角速度")
        plt.legend()

        plt.subplot(2, 2, 3)
        plt.plot(drone_3_ang_vel_x, label="无人机 3 x")
        plt.plot(drone_3_ang_vel_y, label="无人机 3 y")
        plt.plot(drone_3_ang_vel_z, label="无人机 3 z")
        plt.title("无人机 3 角速度")
        plt.legend()

        # 绘制缆绳角度数据
        plt.figure(figsize=(10, 8))
        plt.subplot(2, 2, 1)
        plt.plot(cable_angle_1_w, label="缆绳 1 w")
        plt.plot(cable_angle_1_x, label="缆绳 1 x")
        plt.plot(cable_angle_1_y, label="缆绳 1 y")
        plt.plot(cable_angle_1_z, label="缆绳 1 z")
        plt.title("缆绳 1 角度")
        plt.legend()

        plt.subplot(2, 2, 2)
        plt.plot(cable_angle_2_w, label="缆绳 2 w")
        plt.plot(cable_angle_2_x, label="缆绳 2 x")
        plt.plot(cable_angle_2_y, label="缆绳 2 y")
        plt.plot(cable_angle_2_z, label="缆绳 2 z")
        plt.title("缆绳 2 角度")
        plt.legend()

        plt.subplot(2, 2, 3)
        plt.plot(cable_angle_3_w, label="缆绳 3 w")
        plt.plot(cable_angle_3_x, label="缆绳 3 x")
        plt.plot(cable_angle_3_y, label="缆绳 3 y")
        plt.plot(cable_angle_3_z, label="缆绳 3 z")
        plt.title("缆绳 3 角度")
        plt.legend()

        # 绘制载荷误差数据
        plt.figure(figsize=(10, 8))
        plt.subplot(2, 2, 1)
        plt.plot(payload_pos_error_x, label="x")
        plt.plot(payload_pos_error_y, label="y")
        plt.plot(payload_pos_error_z, label="z")
        plt.title("载荷位置误差")
        plt.legend()

        plt.subplot(2, 2, 2)
        plt.plot(payload_quat_error_w, label="w")
        plt.plot(payload_quat_error_x, label="x")
        plt.plot(payload_quat_error_y, label="y")
        plt.plot(payload_quat_error_z, label="z")
        plt.title("载荷姿态误差")
        plt.legend()

        # 调整布局并显示所有图表
        plt.tight_layout()
        plt.show()

    # 关闭环境
    env.close()


if __name__ == "__main__":
    # 执行主函数
    main()
    # 关闭模拟器应用程序
    simulation_app.close()
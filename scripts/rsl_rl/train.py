"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse  # 用于解析命令行参数
from isaaclab.app import AppLauncher  # Isaac Sim 应用启动器

# 本地导入自定义模块
import cli_args  # isort: skip

# 添加命令行参数
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")  # 是否录制视频
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")  # 视频长度（步数）
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")  # 视频录制间隔（步数）
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."  # 禁用 Fabric 并使用 USD I/O 操作
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")  # 并行环境数量
parser.add_argument("--task", type=str, default=None, help="Name of the task.")  # 任务名称
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")  # 随机种子
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")  # 最大训练迭代次数

# 添加 RSL-RL 相关命令行参数
cli_args.add_rsl_rl_args(parser)
# 添加 AppLauncher 相关命令行参数
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()  # 解析所有命令行参数

# 如果启用了视频录制，则强制开启摄像头
if args_cli.video:
    args_cli.enable_cameras = True

# 启动 Isaac Sim 模拟器应用
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym  # 强化学习环境库
import os  # 操作系统接口
import torch  # PyTorch 深度学习框架
from datetime import datetime  # 时间处理模块

from rsl_rl.runners import OnPolicyRunner  # RSL-RL 的训练运行器

# 导入自定义任务扩展模块
import MARL_mav_carry_ext.tasks  # noqa: F401

from isaaclab.envs import ManagerBasedRLEnvCfg  # 环境配置类
from isaaclab.utils.dict import print_dict  # 字典打印工具
from isaaclab.utils.io import dump_pickle, dump_yaml  # 数据序列化工具
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg  # 工具函数
from isaaclab_tasks.utils.wrappers.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper  # RSL-RL 包装器

# 设置 PyTorch 的性能优化选项
torch.backends.cuda.matmul.allow_tf32 = True  # 允许 CUDA 使用 TF32 精度矩阵乘法
torch.backends.cudnn.allow_tf32 = True  # 允许 cuDNN 使用 TF32 精度卷积
torch.backends.cudnn.deterministic = False  # 不启用确定性算法
torch.backends.cudnn.benchmark = False  # 不启用自动寻找最优算法


def main():
    """Train with RSL-RL agent."""
    
    # 解析环境和代理配置
    env_cfg: ManagerBasedRLEnvCfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )  # 解析环境配置
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)  # 解析代理配置

    # 指定实验日志根目录
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)  # 日志根路径
    log_root_path = os.path.abspath(log_root_path)  # 获取绝对路径
    print(f"[INFO] Logging experiment in directory: {log_root_path}")  # 打印日志目录信息

    # 指定当前运行的日志子目录（带时间戳）
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # 当前时间戳
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"  # 添加运行名称（如果有）
    log_dir = os.path.join(log_root_path, log_dir)  # 组合完整路径

    # 设置最大训练迭代次数
    if args_cli.max_iterations:
        agent_cfg.max_iterations = args_cli.max_iterations  # 覆盖默认值

    # 创建 Isaac 环境实例
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)  # 渲染模式根据是否录制视频决定

    # 如果启用了视频录制，包装环境以支持视频记录
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),  # 视频保存目录
            "step_trigger": lambda step: step % args_cli.video_interval == 0,  # 触发条件：每隔一定步数录制一次
            "video_length": args_cli.video_length,  # 单个视频的长度
            "disable_logger": True,  # 禁用日志记录
        }
        print("[INFO] Recording videos during training.")  # 提示正在录制视频
        print_dict(video_kwargs, nesting=4)  # 打印视频录制参数
        env = gym.wrappers.RecordVideo(env, **video_kwargs)  # 包装环境以支持视频录制

    # 包装环境以适配 RSL-RL 框架
    env = RslRlVecEnvWrapper(env)

    # 创建 RSL-RL 训练运行器
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)  # 初始化运行器

    # 将 Git 仓库状态写入日志（用于版本追踪）
    runner.add_git_repo_to_log(__file__)

    # 如果启用了恢复训练，加载之前保存的模型检查点
    if agent_cfg.resume:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)  # 获取检查点路径
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")  # 打印加载路径
        runner.load(resume_path)  # 加载模型

    # 设置环境的随机种子以保证可重复性
    env.seed(agent_cfg.seed)

    # 将环境和代理配置保存到日志目录中
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)  # 保存环境配置为 YAML 文件
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)  # 保存代理配置为 YAML 文件
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)  # 保存环境配置为 Pickle 文件
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)  # 保存代理配置为 Pickle 文件

    # 开始训练
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)  # 执行训练循环

    # 关闭环境
    env.close()


if __name__ == "__main__":
    # 运行主函数
    main()
    # 关闭模拟器应用
    simulation_app.close()
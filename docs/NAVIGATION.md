# RLRoverLab 导航算法 → 本项目 Gazebo

本迁移以当前部署版本为基线，仅移植高度图导航的算法主链：状态、奖励、终止、目标／出生点
采样、CNN/MLP 网络及 skrl PPO 训练流程。仿真仍加载本项目 `worlds/moon.sdf` 或
`worlds/mars.sdf`，使用原机器人、地形、岩石、材质、关节、惯性、碰撞与物理引擎。
原部署脚本、Dockerfile 和世界文件保持不变。

参考源码固定于 [RLRoverLab 提交 578906a](https://github.com/abmoRobotics/RLRoverLab/tree/578906ad5c3383087aeec038be82b87e05d4ee2c)。
`third_party/rlroverlab` 只包含算法源码、配置、许可证和 SHA-256 清单，没有其仿真资产。
`space_robotics_nav/vendor` 保留算法函数／网络类主体，仅替换依赖导入。

## 安装

```bash
bash tools/navigation/setup.bash
```

使用 uv 管理 Python 3.12 和锁定依赖；不需要 Isaac Sim、USD、Conda 或 ROS。
优先以本机原部署镜像 `space_robotics_gz_envs:latest` 为基础，缺少时使用原项目预构建镜像。
可以用 `NAV_GAZEBO_IMAGE=<原部署镜像>` 明确指定。最终镜像 `space-robotics-rl:ionic`
继承原镜像的 Gazebo／物理／渲染库，仅添加同步数据接口程序。

资产直接复用 `assets/cache`；缺少时从原项目镜像提取生成资产，并按原 SDF 的 Fuel URL
缓存原机器人。不会导出或使用 AAU Rover、RLRoverLab 地形或 PhysX。

## 训练

```bash
# 月面：原 EXPLORER R2；3000 向量步 × 4 环境 = 12000 条 transition
uv run --frozen python -m space_robotics_nav.train \
  --world moon --num-envs 4 --device cuda:0 --steps 3000 \
  --mini-batches 8 --output logs/navigation/moon_run01

# 火星：原 Perseverance
uv run --frozen python -m space_robotics_nav.train \
  --world mars --num-envs 4 --device cuda:0 --steps 3000 \
  --mini-batches 8 --output logs/navigation/mars_run01

# 同时启用固定留出种子的训练前／后随机策略对比
uv run --frozen python -m space_robotics_nav.train \
  --world moon --num-envs 4 --device cuda:0 --steps 3000 \
  --mini-batches 8 --eval-episodes 8 --stochastic-eval \
  --output logs/navigation/moon_paired01

# 从新回合恢复网络、Adam 和随机数状态，额外训练 30000 步
uv run --frozen python -m space_robotics_nav.train \
  --world moon --num-envs 4 --device cuda:0 --steps 30000 \
  --mini-batches 8 --checkpoint logs/navigation/moon_run01/final.pt \
  --output logs/navigation/moon_resume01
```

未指定 `--steps` 时沿用原 YAML 的 1000000 步。默认直接训练；添加 `--eval-episodes`
才会先等待完整初始评估。`--stochastic-eval` 只控制评估方式，训练本身始终使用高斯采样。
使用新输出目录以保留实验。Ctrl+C 会保存 `interrupted.pt`。恢复从新回合／新 rollout
开始，不承诺逐物理步续跑。

输出包含 `run.json`、`initial.pt`、`final.pt`、优化器／随机状态侧文件、逐回合日志和
TensorBoard；启用前后评估时另生成 `comparison.json`。定期 checkpoint 位于 `ppo/checkpoints`。

```bash
uv run --frozen tensorboard --logdir logs/navigation --port 6006
```

## 在 Gazebo 中观看

在桌面终端运行，播放程序自行启动实际 Gazebo 物理进程与 GUI：

```bash
uv run --frozen python -m space_robotics_nav.play \
  logs/navigation/moon_run01/final.pt --seed 10000
```

绿色标记表示目标，橙色条表示要求的最终朝向。默认使用确定性策略、1 倍速度，
自动切换回合；关闭窗口或 Ctrl+C 结束。可用 `--stochastic`、`--episodes 3`、
`--speed 0.5`，无 NVIDIA 容器图形支持时可用 `--software-rendering`。
界面沿用原场景光照和材质；目标注记没有碰撞体。
不需要先运行 `.docker/local.bash`；原手动部署入口仍可单独使用。

## 单独评估

```bash
uv run --frozen python -m space_robotics_nav.evaluate \
  logs/navigation/moon_run01/final.pt --episodes 12 --seed 30000 \
  --output logs/navigation/moon_run01/deterministic_eval.json

uv run --frozen python -m space_robotics_nav.evaluate \
  logs/navigation/moon_run01/final.pt --episodes 12 --seed 30000 --stochastic \
  --output logs/navigation/moon_run01/stochastic_eval.json
```

同时看成功率、碰撞率、最终距离、总回报与每步回报。配对评估固定环境种子，随机动作
使用每回合独立的随机数生成器；不会因其他回合先结束而改变该回合的噪声序列。
只有 95% 配对 bootstrap 区间下界大于零时，奖励提升标记才为真。
短期奖励改善不代表已经可靠导航。

## 接口和保留的参数

| 项目 | 实现 |
| --- | --- |
| 观测 | float32 `(N, 10206)`：2 个上次动作、距离 ×0.11、方位角 ÷π、目标朝向误差 ÷π、10201 个高度值 |
| 扫描 | 原网格 5×5 m、0.05 m、101×101、xy 顺序；高度公式 `sensor_z-hit_z-0.26878` |
| 动作 | `(N, 2)`，依次为线速度 m/s、角速度 rad/s，送入原 `/cmd_vel` DiffDrive |
| 任务 | 9 m 圆周目标，独立随机目标朝向，150 s／750 控制步 |
| 成功 | 距离 <0.18 m 且朝向误差 <0.1 rad |
| 采样 | 原 0.05 m 高度投影、岩石／坡度形态学掩码、2 m 安全核、4096 出生点、出生边距 10 m、目标边距 7 m |
| 网络 | 原卷积编码器 `[8,16,32,64]`、编码 MLP `[80,60]`、主 MLP `[256,160,128]`、独立高斯 policy／value |
| PPO | rollouts 60、epochs 4、原 mini_batches 60、lr 1e-4、γ 0.99、GAE 0.95、clip 0.2、grad clip 0.5、KL 0.008 |

八项奖励的函数、权重和内部归一化均保留：距离 +5、到达 +5、振荡 −0.05、目标方位 −1.5、
倒车 −0.5、碰撞 −3、远离 −2、目标朝向 +5；继续乘控制周期 0.2 s。
保留源环境“先奖励／终止，后更新目标命令”的顺序、网络 `[4:-1]` 切片及单侧四次振荡惩罚。
没有为提高验收奖励而修改这些行为。

## 必要的 Gazebo 适配

- 原月面重力 −1.622 m/s²，原火星重力 −3.73 m/s²；物理步长均为 0.025 s。
  每 8 步执行一次策略，保持原算法的 5 Hz。训练只取消墙钟等待，不改 SDF 物理配置。
- 原 DiffDrive 自行限制速度／加速度；不套用参考车的 Ackermann 参数，也不额外缩放原动作。
  原速度限制为线速度 ±0.5 m/s、角速度 ±1 rad/s。
- 原 Perseverance 的轮轴沿 +X，实际前向为 −Y；目标坐标和扫描方向采用 −π/2 航向偏移，
  机器人资产、轮轴、轮径和轮距不变。
- 高度图从原碰撞网格求射线交点，动态岩石位姿来自 Gazebo；不是用其他项目的地图代替。
  射线未命中使用 100 m 量程；越过地图有效扫描边界会终止并记录 `out_of_bounds`。
- 接触力只统计原世界独立岩石模型，地面支撑力被过滤；原地形网格内烘焙的地形起伏仍归地面。
- 每个子环境独立 Gazebo 进程，重置原机器人、关节、速度及动态物体后稳定 100 个物理步。
  默认 4 个 CPU 物理环境；上游 128 个 GPU 物理环境不是本机吞吐量假设。
- 配置 YAML 默认值保持原样。文档小规模命令显式覆盖 `mini_batches=8`；所有覆盖写入 `run.json`。

当前交付验证高度图 + PPO 主链；没有将相机/RGB-D、SAC/TD3 等其他分支冒充为已验收功能。

## 检查

```bash
uv run --frozen --extra test pytest tests/navigation -q
RUN_GAZEBO_TESTS=1 uv run --frozen --extra test pytest tests/navigation/test_gazebo.py -q
```

数值测试与固定源码对照；集成测试使用本项目真实月面／火星机器人，检查物理参数、控制、
状态、奖励时序、重置和原岩石接触力。实测记录见 [验证报告](NAVIGATION_VALIDATION.md)。

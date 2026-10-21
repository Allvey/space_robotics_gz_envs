# RLRoverLab 导航算法 → 本项目 Gazebo

本迁移以当前部署版本为基线，仅移植高度图导航的算法主链：状态、奖励、终止、目标／出生点
采样、CNN/MLP 网络及 skrl PPO 训练流程。仿真仍加载本项目 `worlds/moon.sdf` 或
`worlds/mars.sdf`，使用原机器人、地形、岩石、材质、关节、惯性、碰撞与物理引擎。
原部署脚本、Dockerfile 和世界文件保持不变。

参考源码固定于 [RLRoverLab 提交 578906a](https://github.com/abmoRobotics/RLRoverLab/tree/578906ad5c3383087aeec038be82b87e05d4ee2c)。
`third_party/rlroverlab` 只包含算法源码、配置、许可证和 SHA-256 清单，没有其仿真资产。
`space_robotics_nav/vendor` 保留算法函数／网络类主体，仅替换依赖导入。
默认训练使用 v2 修正适配层；原始实现作为显式基线保留。修正项见下文。

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
# 本机已实测的月面配置：16 环境、48000 条 transition，每小批约 120 条
# 显式使用原控制器探索尺度实验，mini_batches 自动适配为 8
uv run --frozen python -m space_robotics_nav.train \
  --world moon --num-envs 16 --device cuda:0 --steps 3000 \
  --config configs/navigation/ppo_gazebo_native_std.yaml \
  --output logs/navigation/moon_v2_run01

# 4 环境小规模对照：原 EXPLORER R2；12000 条 transition，每小批 30 条
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

# 从上述 v2 运行恢复网络、Adam 和随机数状态，额外训练 30000 步
uv run --frozen python -m space_robotics_nav.train \
  --world moon --num-envs 16 --device cuda:0 --steps 30000 \
  --config configs/navigation/ppo_gazebo_native_std.yaml \
  --checkpoint logs/navigation/moon_v2_run01/final.pt \
  --output logs/navigation/moon_v2_resume01
```

未指定 `--steps` 时沿用原 YAML 的 1000000 步。默认直接训练；添加 `--eval-episodes`
才会先等待完整初始评估。`--stochastic-eval` 只控制评估方式，训练本身始终使用高斯采样。
使用新输出目录以保留实验。Ctrl+C 会保存 `interrupted.pt`。恢复从新回合／新 rollout
开始，不承诺逐物理步续跑。

默认配置为 `configs/navigation/ppo_gazebo.yaml`，继承原始 `ppo.yaml` 并启用 v2。
旧 `moon_ppo` 等 checkpoint 仍按其 `run.json` 中的原始版本评估／播放；v2 应从头训练，
不接续旧版本的模型和 Adam 状态。继续旧基线时显式传入 `--config configs/navigation/ppo.yaml`。

未传 `--mini-batches` 时，v2 按上游每批约 128 条样本自动适配：4／8／16 个环境分别采用
2／4／8 个小批次，每批约 120 条；128 个环境时恢复原始 60 个小批次。实际数量写入 `run.json`。
上面显式指定 8 的 4 环境命令用于复现小规模基线，只有每批 30 条；正式训练可去掉该覆盖。

另提供显式的原控制器探索尺度实验：`--config configs/navigation/ppo_gazebo_native_std.yaml`。
该配置仅将从头训练的初始线速度／角速度标准差设置为 `[0.25, 0.5]`，对应原速度上限幅值的一半；
策略均值结构、动作物理单位及奖励均不变，标准差之后仍由 PPO 学习。恢复 checkpoint 时沿用其已学习的标准差。
该实验应与自身未训练模型比较，不能把降低初始噪声带来的奖励差异当作学习增益。

输出包含 `run.json`、`initial.pt`、`final.pt`、优化器／随机状态侧文件、逐回合日志和
TensorBoard；启用前后评估时另生成 `comparison.json`。定期 checkpoint 位于 `ppo/checkpoints`。
日志新增采样似然的更新前／后 KL、实际梯度步数、零更新 rollout、逐项奖励、线速度／角速度
请求均值和超限比例。先看成功率与距离进展，再结合奖励判断学习效果。

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

同时看成功率、碰撞率、最终距离、轨迹最小距离、总回报与每步回报。
`position_reached` 表示曾进入位置容差，可与严格成功率对照判断最终朝向是否仍有问题。
配对评估固定环境种子，随机动作
使用每回合独立的随机数生成器；不会因其他回合先结束而改变该回合的噪声序列。
只有 95% 配对 bootstrap 区间下界大于零时，奖励提升标记才为真。
评估同时给出距离进展增益及其配对区间；总回报提高、但更早远离目标而终止时，不视为导航改善。
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
| PPO | rollouts 60、epochs 4、lr 1e-4、γ 0.99、GAE 0.95、clip 0.2、grad clip 0.5、KL 0.008；v2 按约 128 样本／批适配 mini_batches，原基线为 60 |

八项奖励的函数、权重和内部归一化均保留：距离 +5、到达 +5、振荡 −0.05、目标方位 −1.5、
倒车 −0.5、碰撞 −3、远离 −2、目标朝向 +5；继续乘控制周期 0.2 s。
保留源环境“先奖励／终止，后更新目标命令”的顺序及单侧四次振荡惩罚。
v2 不改变奖励公式、权重、网络层尺寸和任务难度，单独修正以下训练行为：

- 高度图编码输入使用 `[5:]`：5 个标量之后的完整 10201 像素；基线仍使用原 `[4:-1]`。
- 采样、优化和播放都冻结 BatchNorm 的运行统计；从头训练时保持初始均值 0、方差 1，
  卷积和 BatchNorm 仿射参数继续接受梯度更新。这样 PPO 的概率比不会因模式切换改变。
- 超时处切断 GAE 轨迹，使用自动重置前的真实末尾观测作价值 bootstrap；碰撞等真实终止
  不 bootstrap，即使同一步也达到了时间上限。适配依赖锁定的 skrl 2.1.0 语义。

实现位于 `space_robotics_nav/adaptation.py`；`third_party` 和 `vendor` 的源快照均保留。

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
- 原始 YAML 保持原样，默认 v2 通过继承配置启用修正。文档小规模命令显式覆盖
  `mini_batches=8`；所有覆盖和适配版本写入 `run.json`。

当前交付验证高度图 + PPO 主链；没有将相机/RGB-D、SAC/TD3 等其他分支冒充为已验收功能。

## 检查

```bash
uv run --frozen --extra test pytest tests/navigation -q
RUN_GAZEBO_TESTS=1 uv run --frozen --extra test pytest tests/navigation/test_gazebo.py -q
```

数值测试与固定源码对照；集成测试使用本项目真实月面／火星机器人，检查物理参数、控制、
状态、奖励时序、重置和原岩石接触力。训练正确性测试另外覆盖 BatchNorm 模式一致性、
高度图像素边界、真实 Adam 更新和超时回报隔离。
旧基线实测记录见 [验证报告](NAVIGATION_VALIDATION.md)，修正版本见 [v2 验证](NAVIGATION_TRAINING_FIXES.md)。

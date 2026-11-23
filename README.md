# Space Robotics Gazebo Environments

<p align="left">
  <a href="https://github.com/AndrejOrsula/space_robotics_gz_envs/actions/workflows/docker.yml"> <img alt="Docker" src="https://github.com/AndrejOrsula/space_robotics_gz_envs/actions/workflows/docker.yml/badge.svg"></a>
</p>

A collection of Gazebo environments for simulating space robotics scenarios. All assets are procedurally generated using [Blender](https://blender.org) and automatically exported to the [SDF](http://sdformat.org) format for simulations inside Gazebo. Below are examples of the included [worlds](worlds):

<p align="center" float="middle">
  <a href="worlds/mars.sdf">
    <img width="49.0%" src="https://github.com/user-attachments/assets/7172e090-7e3e-402d-83a9-96954c1a9af4"/>
  </a><a href="worlds/moon.sdf">
    <img width="49.0%" src="https://github.com/user-attachments/assets/451af69e-70bd-4b3c-9f1c-8bb4120b612b"/>
  </a>
<br>
  <a href="worlds/mars_array.sdf">
    <img width="49.0%" src="https://github.com/user-attachments/assets/ca97f067-c4b0-4dd9-85ca-efbc2026fcd9"/>
  </a><a href="worlds/moon_array.sdf">
    <img width="49.0%" src="https://github.com/user-attachments/assets/e5984191-08da-49bb-9d0d-326ddf4d26cd"/>
  </a>
</p>

The assets are based on the work in [Space Robotics Bench](https://andrejorsula.github.io/space_robotics_bench) that targets Isaac Sim. However, all procedural pipelines for generating the assets are available in a separate [`srb_assets` repository](https://github.com/AndrejOrsula/srb_assets) that is included as a [submodule](assets). Below is a video showcasing the procedural generation directly inside Blender:

https://github.com/user-attachments/assets/5345ebe2-5692-4df4-8bea-1967cc4a2aa1

## Instructions

Navigation reinforcement learning on the original Gazebo robots and worlds is available through the
[quick start below](#单车导航强化学习新增) and the [RLRoverLab algorithm migration guide](docs/NAVIGATION.md).

基于原有 EXPLORER R2 改装的采矿车与运输车，见 [模型预览、启动与控制说明](docs/MINING_VEHICLES.md)。
展示场景为 [`worlds/mining_vehicles.sdf`](worlds/mining_vehicles.sdf)。

<details><summary><h3>Installation (Docker)</h3></summary>

This section provides instructions for running the simulation within a Docker container. If you are using a different operating system than Ubuntu, you may need to adjust the following steps accordingly or refer to the official documentation for each step.

#### 1. Install [Docker Engine](https://docs.docker.com/engine)

First, install Docker Engine by following the [official installation instructions](https://docs.docker.com/engine/install). For example:

```bash
curl -fsSL https://get.docker.com | sh
sudo systemctl enable --now docker

sudo groupadd docker
sudo usermod -aG docker $USER
newgrp docker
```

#### 2. Install [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit)

Next, install the NVIDIA Container Toolkit, which is required to enable GPU support for Docker containers. Follow the [official installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) or use the following commands:

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

#### 3. Clone the Repository

Next, clone the `space_robotics_gz_envs` repository locally. Make sure to include the `--recurse-submodules` flag to clone also the submodule containing the procedural pipelines for the simulation assets.

```bash
git clone --recurse-submodules https://github.com/AndrejOrsula/space_robotics_gz_envs.git
```

<details><summary><h4>4. (OPTIONAL) Build the Docker Image</h4></summary>

> [!NOTE]
> This step can be skipped if you prefer to use the pre-built Docker image from the [Docker Hub](https://hub.docker.com/r/andrejorsula/space_robotics_gz_envs). This results in a much faster setup process (just continue with the next step).

You can build the Docker image for `space_robotics_gz_envs` by running the provided [`.docker/build.bash`](https://github.com/AndrejOrsula/space_robotics_gz_envs/blob/main/.docker/build.bash) script.

```bash
space_robotics_gz_envs/.docker/build.bash
```

To ensure that the image was built successfully, run the following command. You should see the `space_robotics_gz_envs` image listed among recently created Docker images.

```bash
docker images
```

</details>

</details>

### Running the Simulation

#### Verify the Functionality of Gazebo

Let's start by verifying that Gazebo is functioning correctly on your system:

```bash
.docker/run.bash gz sim
```

#### (Optional) Generate Procedural Assets

> [!NOTE]
> This step can be skipped if you are using the pre-built Docker image from the [Docker Hub](https://hub.docker.com/r/andrejorsula/space_robotics_gz_envs).

To generate procedural assets, you can run the following command:

```bash
.docker/run.bash scripts/procgen_assets.bash
```

#### Example Worlds

You can launch Gazebo with one of the included worlds by passing its SDF basename as the argument. For example:

```bash
# Lunar sufarce with a rover that cam be controlled via the Teleop plugin
.docker/run.bash gz sim moon.sdf
```

https://github.com/user-attachments/assets/019b57b3-17ee-4ca7-aef8-03bcf408268f

```bash
# Martian surface with a rover (robot requires tuning)
.docker/run.bash gz sim mars.sdf
```

https://github.com/user-attachments/assets/7a33dcea-5f39-4340-a71d-284fe89fd74c

```bash
# Grid (2x2) of lunar surfaces with some solar panels
.docker/run.bash gz sim moon_array.sdf
```

```bash
# Grid (2x2) of martian surfaces
.docker/run.bash gz sim mars_array.sdf
```

```bash
# Empty world with enabled Resource Spawner plugin
.docker/run.bash gz sim _empty.sdf
```

## 单车导航强化学习（新增）

本项目移植了 [RLRoverLab](https://github.com/abmoRobotics/RLRoverLab) 的导航强化学习算法，
包括状态、奖励、CNN/MLP 网络及 skrl PPO 训练流程。仿真继续使用本项目原有的 Gazebo、
月球／火星世界、机器人与物理资产。原有手动启动、遥控和资产生成方式继续可用。

**当前导航功能为单车模式：每个仿真世界只有一辆车。** 训练时可以并行多个独立世界，
共同训练一个策略；Gazebo 可视化始终只播放一辆车。

新增功能包括：

- **高度图导航**：10206 维观测由 5 个导航标量和 101×101 局部高度图组成。
  标量包含上次线速度／角速度动作、目标距离、目标方位和最终朝向误差；策略输出线速度与角速度。
- **并行单车 PPO 训练**：各个独立世界分别采样，车辆之间没有物理交互。
- **模型保存与训练曲线**：保存训练配置和模型，通过 TensorBoard 查看奖励、成功率和碰撞率。
- **连续随机目标播放**：车辆连续执行多个随机目标，支持跟随视角、目标位置／朝向标记、
  从当前位置继续导航以及碰撞、越界或翻倒后的重置。
- **可选的末端规划与停靠**：接近目标后使用 RS／差速路径规划和反馈控制完成位置、朝向调整。

### 安装导航扩展

完成上方 Docker 安装；使用 NVIDIA GPU 时同时完成 NVIDIA Container Toolkit 配置。
在项目根目录、已安装 `uv` 的终端中执行：

```bash
bash tools/navigation/setup.bash
```

该脚本准备 Python 3.12 锁定依赖与原项目资产，构建导航接口镜像
`space-robotics-rl:ionic`，并运行相关检查。训练与播放会自行启动 Gazebo 容器，
无需先用 `.docker/run.bash` 启动另一个世界。

### 从头训练

下面运行 16 个独立的单车世界，共同训练一个 PPO 策略：

```bash
uv run --frozen python -m space_robotics_nav.train \
  --world moon \
  --num-envs 16 \
  --device cuda:0 \
  --iterations 834 \
  --seed 42 \
  --config configs/navigation/ppo_gazebo_native_std.yaml \
  --output logs/navigation/moon_single_run01
```

`--num-envs` 表示独立世界数，默认值为 4。只需要一个训练世界时设为 `1`；
增加该参数只增加并行采样，不会在同一世界生成多辆车。
`--iterations` 表示完整的 PPO 训练轮次，每轮先采样再更新策略。默认每轮采样 60 个
向量步，因此 834 轮对应 50040 个向量步；16 环境共约 80 万条 transition，
4 环境则约 20 万条。小批次数自动适配环境数量。
兼容旧参数 `--steps`，但不能与 `--iterations` 同时使用；步数预算会向上补齐为完整轮次，
例如 `--steps 50000` 实际执行 834 轮／50040 步，并在启动时说明。两者都不指定时，
从配置文件的步数预算换算轮次。

训练沿用原目标距离 9 m、随机目标朝向和 150 s 回合时限。
成功条件为**距离 <0.18 m 且最终朝向误差 <0.1 rad**；RS 不参与训练。
`--world mars` 可使用原火星环境和机器人。

从头训练不传 `--checkpoint`，每次使用新的输出目录。`run.json` 保存训练配置，
`initial.pt`、`final.pt` 保存初始和最终模型，定期模型位于 `ppo/checkpoints/agent_*.pt`。
`best_agent.pt` 按训练回报选取，需结合成功率、距离进展和碰撞率判断导航能力。

每轮 PPO 更新结束后，终端用分隔线逐行输出以下指标。采样指标统计本轮全部环境的
全部采样，不依赖 episode 是否已经结束；每轮重新累计。

| 指标 | 含义 |
| --- | --- |
| `iteration` | 已完成轮次／固定总轮次 |
| `progress` | 按已完成轮次计算的进度 |
| `elapsed_time` | 训练已用时间，时:分:秒 |
| `estimated_remaining_time` | 按已完成轮次平均耗时估算的剩余时间，包含 PPO 更新耗时，不含最后保存与可选评估 |
| `reward_mean` | 平均每步环境奖励，使用超时价值 bootstrap 前的原始奖励 |
| `distance_to_goal_mean_m` | 本轮所有采样时刻的平均目标距离，米 |
| `approach_speed_mean_mps` | 同一目标下距离减少量／控制周期的均值，米/秒；正值接近、负值远离，排除重置跳变 |
| `linear_speed_mean_mps` | Gazebo 实测速度在车辆导航前向上的投影均值，米/秒，非动作请求值 |
| `collision_step_rate` | 检测到岩石碰撞的采样数／本轮采样总数，显示百分比和计数 |
| `success_count` | 本轮成功到达目标的次数 |

例如 4 环境每轮 60 步，共 240 条采样。碰撞采样率与按已完成回合计算的碰撞率口径不同。
每轮结果立即追加到 `iterations.jsonl`；每个已完成 episode 的累计奖励、最终／最近距离、
成功／碰撞／越界与超时结果立即追加到 `episodes.jsonl`，训练正常结束时仍保存
`training_episodes.json`。训练早期预计剩余时间可能波动；恢复训练时轮次和统计从本次运行重新计数。

### 查看训练曲线

```bash
uv run --frozen tensorboard --logdir logs/navigation --port 6006
```

浏览器打开 `http://localhost:6006`。新训练在 `ppo` 中以 **iteration** 为横轴，每轮更新后
写入并刷新指标；优先查看 `Train/reward_mean`、`Train/distance_to_goal_mean_m`、
`Train/approach_speed_mean_mps`、`Train/collision_step_rate` 和 `Train/success_count`。
`Progress/` 显示进度与预计剩余秒数，策略／价值损失、KL、奖励分项和动作诊断也按 iteration 记录。

完整回合数据保存在独立的 `episode_metrics` 中，以已完成 **episode 编号**为横轴，包含
`Episode/return`、`Episode/distance`、`Episode/distance_progress`、`Episode/min_distance`、
`Episode/success`、`Episode/collision`，以及累计的 `Episode/success_rate` 和
`Episode/collision_rate`。尚无完整回合时不写回合指标，不用零值代替缺失数据。
旧训练日志保留原有向量步横轴，与新训练直接比较时应注意单位差异。
平均奖励上升需要结合目标距离、成功率和碰撞率解读，不能单独作为导航性能提升的结论。

### 在 Gazebo 中观看模型

在桌面终端执行，程序会启动一个 Gazebo 世界和一辆车，默认由策略直接控制，**末端规划默认关闭**：

```bash
uv run --frozen python -m space_robotics_nav.play \
  logs/navigation/moon_single_run01/final.pt \
  --device cuda:0 \
  --goals 10
```

训练尚未结束时，可将 `final.pt` 替换为已保存的 `ppo/checkpoints/best_agent.pt`、
`ppo/checkpoints/agent_XXXX.pt` 或 `interrupted.pt`。加载模型需要其训练目录中的 `run.json`。

`--goals 10` 表示连续执行 10 个目标，`--goals 0` 持续导航。不传 `--seed` 时每次生成新任务；
`--seed 30002` 可固定运行种子。默认跟随车辆，`--fixed-camera` 固定视角，
`--reset-between-goals` 让每个任务从重新采样的出生点开始。
绿色信标表示目标位置，橙色横条表示目标朝向。关闭窗口或 Ctrl+C 结束播放。

### RS 末端规划的含义与使用

**RS 是 Reeds–Shepp 路径的缩写**，是一种连接车辆当前位置、朝向与目标位置、朝向的
几何路径规划方法。它将直线和圆弧组合成允许前进、倒车的路径，并满足设定的转弯半径约束。

在本项目中，强化学习策略负责接近目标；启用末端模块后，规划器检查原场景碰撞几何、
选择有效路径，再由反馈控制器跟踪，最后结合差速停靠控制完成位置和朝向细调。
RS 属于播放时的规划与控制模块，不是神经网络的一部分，也不参与 PPO 训练。

观看启用 RS 的单车导航：

```bash
uv run --frozen python -m space_robotics_nav.play \
  logs/navigation/moon_single_run01/final.pt \
  --device cuda:0 --goals 10 \
  --terminal-planner --terminal-family rs
```

该版本默认在距离目标 **2 m 内**尝试接管；无有效路径时继续策略导航。
最后 0.6 m 内使用差速停靠反馈，成功标准仍为 **0.18 m／0.1 rad**。
`--terminal-distance` 可设置接管距离；不传 `--terminal-family rs` 时，默认同时比较
RS 和差速候选路径。关闭末端规划时，省略 `--terminal-planner` 即可。
终端的 `mode=rl`、`terminal_rs` 或 `terminal_diff-drive` 表示当前控制来源。
开启规划后的表现包含控制模块的贡献，不应全部归因于强化学习模型。

完整接口、参数和已有验证记录见：

- [导航算法迁移与训练说明](docs/NAVIGATION.md)
- [近目标规划与停靠](docs/NAVIGATION_TERMINAL.md)
- [PPO 训练适配与验证记录](docs/NAVIGATION_TRAINING_FIXES.md)

## License

This project is dual-licensed under either the [MIT](LICENSE-MIT) or [Apache 2.0](LICENSE-APACHE) licenses.

All assets created by contributors of this repository and those generated from the included procedural pipelines are licensed under the [CC0 1.0 Universal](https://github.com/AndrejOrsula/srb_assets/blob/main/LICENSE-CC0) license. Some assets are based on modified third-party resources, which might require you to give appropriate credit to the original author. Please review [`srb_assets` repository](https://github.com/AndrejOrsula/srb_assets) for more information.

<a href="https://creativecommons.org/publicdomain/zero/1.0"><img src="https://licensebuttons.net/l/zero/1.0/88x31.png" width="88" height="31"></a>

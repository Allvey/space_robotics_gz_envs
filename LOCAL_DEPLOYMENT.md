# 本机部署

使用 Docker 中的 Ubuntu 24.04、Gazebo Ionic 和已有场景资源，无需 Conda。
启动脚本强制使用 NVIDIA 渲染，适配本机 AMD 核显＋NVIDIA 独显。

在项目根目录执行：

```bash
# 启动月面场景，打开 Gazebo 窗口，容器在后台运行
bash .docker/local.bash start

# 查看状态／日志／停止
bash .docker/local.bash status
bash .docker/local.bash logs
bash .docker/local.bash stop

# 不显示窗口
bash .docker/local.bash start moon --headless

# 切换场景前先停止；可选 moon、mars、moon_array、mars_array、_empty
bash .docker/local.bash stop
bash .docker/local.bash start mars
```

图形启动需在已登录的桌面终端中执行。关闭 Gazebo 窗口会结束本次容器。
月面场景可以通过右侧 Teleoperation 面板发布 `/cmd_vel` 控制车辆。
这里使用 Gazebo Transport，并非 ROS 2 话题；本次部署的是原版仿真环境。
上游模型会输出部分 SDF／材质兼容和三角网格碰撞警告；本机已验证场景运行和速度指令。
容器标准输出日志限制为每份 10 MB、保留三份。

本机复用镜像 `space_robotics_gz_envs:latest`，可通过 `GAZEBO_IMAGE` 指定其他镜像。
迁移到新机器时可使用上游预构建镜像：

```bash
docker pull andrejorsula/space_robotics_gz_envs:latest
GAZEBO_IMAGE=andrejorsula/space_robotics_gz_envs:latest bash .docker/local.bash start
```

场景和脚本从当前仓库只读挂载。生成的地形、石块及 Gazebo 下载缓存分别保存在
`space_robotics_gz_base_assets` 和 `space_robotics_gz_base_home` Docker 卷中，停止容器不会删除。
首次启动需要下载 Fuel 机器人模型；本次部署已缓存月面和火星场景的机器人。

如需修改程序化资产生成管线，先确保子模块已初始化：

```bash
git submodule update --init --recursive
```

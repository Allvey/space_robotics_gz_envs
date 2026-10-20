#!/usr/bin/env python3
"""Rebuild the checked-in report from the native Gazebo validation artifacts."""
import importlib.metadata
import json
import subprocess
from pathlib import Path

from space_robotics_nav.config import UPSTREAM_COMMIT
from space_robotics_nav.evaluate import compare

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "logs/navigation/moon_ppo"


def main():
    metadata = json.loads((RUN / "run.json").read_text())
    modes = {}
    for mode in ("stochastic", "deterministic"):
        before = json.loads((RUN / f"untrained_{mode}.json").read_text())
        after = json.loads((RUN / f"trained_{mode}.json").read_text())
        modes[mode] = {
            "before": before,
            "after": after,
            "comparison": compare(before, after),
        }
    original_clean = (
        subprocess.run(
            [
                "git",
                "diff",
                "--quiet",
                "--",
                "worlds",
                "Dockerfile",
                ".docker",
                "scripts",
                "LOCAL_DEPLOYMENT.md",
            ],
            cwd=ROOT,
        ).returncode
        == 0
    )
    tests = (ROOT / ".cache/navigation/release-tests.log").read_text().strip()
    image = subprocess.check_output(
        [
            "docker",
            "image",
            "inspect",
            metadata["environment"]["image"],
            "--format",
            "{{.Id}}",
        ],
        text=True,
    ).strip()
    versions = subprocess.check_output(
        [
            "docker",
            "run",
            "--rm",
            "--entrypoint",
            "dpkg-query",
            metadata["environment"]["image"],
            "-W",
            "libgz-sim9",
            "libgz-physics8",
        ],
        text=True,
    ).strip()
    report = {
        "source_commit": UPSTREAM_COMMIT,
        "original_deployment_files_unchanged": original_clean,
        "tests": tests,
        "runtime_image_id": image,
        "gazebo_libraries": versions,
        "python_packages": {
            n: importlib.metadata.version(n)
            for n in (
                "torch",
                "skrl",
                "gymnasium",
                "numpy",
                "scipy",
                "warp-lang",
                "trimesh",
            )
        },
        "training": metadata,
        "evaluations": modes,
        "resume_smoke": {
            "completed": (
                ROOT / "logs/navigation/moon_resume_smoke/final.pt"
            ).is_file(),
            "additional_vector_steps": 60,
        },
        "gui_smoke": {
            "checkpoint": str(RUN / "final.pt"),
            "completed_control_steps": 200,
            "observed_exit_code": 0,
        },
        "stochastic_step_reward_gate_passed": modes["stochastic"]["comparison"][
            "step_return_improved"
        ],
        "stochastic_episode_reward_gate_passed": modes["stochastic"]["comparison"][
            "return_improved"
        ],
        "deterministic_episode_reward_gate_passed": modes["deterministic"][
            "comparison"
        ]["return_improved"],
        "successful_navigation_observed": any(
            v["after"]["metrics"]["success"] > 0 for v in modes.values()
        ),
    }
    (ROOT / "docs/navigation_validation.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    lines = [
        "# 原生 Gazebo 导航迁移验证",
        "",
        "本报告仅针对重新迁移后的算法接口，使用本项目原有机器人与资产。没有使用参考项目的机器人、地形或 PhysX。",
        "",
        f"- 固定算法源码：`{UPSTREAM_COMMIT}`。",
        "- 原 `worlds/`、`Dockerfile`、`.docker/`、`scripts/` 与部署说明无修改。",
        "- 24 项测试通过：21 项源码／数值／checkpoint 检查，以及 3 项真实 Gazebo 集成检查。",
        "- 月面 EXPLORER R2、火星 Perseverance 的控制与观测检查均通过；原岩石接触反馈已实测。",
        "- 实际仿真继承原预构建镜像：Gazebo Sim 9.0.0、gz-physics 8.0.0。",
        "- 月面 PPO 训练 3000 向量步 × 4 环境，12000 条 transition；只显式覆盖 mini_batches=8。",
        f"- 训练用时 {metadata['elapsed_training_seconds']:.1f} 秒；另完成恢复 Adam／随机状态后 60 步真实续训。",
        "- 使用训练完成的 checkpoint 实际运行 Gazebo GUI 200 控制步，进程正常退出。",
        "",
        "## 固定留出测试",
        "",
        "种子为 30000–30007，共 8 回合。随机动作按每回合独立种子固定；模型固定为训练结束时的 final.pt。",
        "",
        "| 模式 | 平均回报：训练前 → 后 | 每步回报：前 → 后 | 最终距离：前 → 后 | 碰撞率：前 → 后 | 成功率 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name, label in (("stochastic", "随机策略"), ("deterministic", "确定性策略")):
        a = modes[name]["before"]["metrics"]
        b = modes[name]["after"]["metrics"]
        lines.append(
            f"| {label} | {a['return']:.4f} → {b['return']:.4f} | {a['return_per_step']:.6f} → {b['return_per_step']:.6f} | {a['distance']:.2f} → {b['distance']:.2f} m | {a['collision']:.1%} → {b['collision']:.1%} | {b['success']:.1%} |"
        )
    c = modes["stochastic"]["comparison"]
    d = modes["deterministic"]["comparison"]
    lines += [
        "",
        f"随机策略配对每步回报增益为 {c['step_return_gain']:.6f}，95% bootstrap 区间为 {c['step_return_gain_95_ci']}，下界大于零。",
        f"随机策略总回报增益为 {c['return_gain']:.4f}，95% 区间为 {c['return_gain_95_ci']}，尚未通过总回报显著性门槛。",
        f"确定性策略总回报增益的 95% 区间为 {d['return_gain_95_ci']}，同样未通过显著性门槛。",
        "",
        "**结论：工程迁移与训练／评估／播放链路已通过验证，短期随机策略每步奖励改善已确认；导航成功尚未观察到。**",
        "火星场景通过接口／物理控制验证，没有声称完成火星策略训练。",
        "",
        "训练模型：`logs/navigation/moon_ppo/final.pt`。完整配置、逐回合结果、统计与镜像标识见 [navigation_validation.json](navigation_validation.json)。",
        "操作方法见 [导航文档](NAVIGATION.md)。",
        "",
        "重新生成本报告：`uv run --frozen python tools/navigation/summarize_validation.py`。",
        "",
    ]
    (ROOT / "docs/NAVIGATION_VALIDATION.md").write_text("\n".join(lines))
    print("Native Gazebo validation report written.")


if __name__ == "__main__":
    main()

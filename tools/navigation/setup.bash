#!/usr/bin/env bash
# Build only the task adapter; retain the original Gazebo runtime and assets.
set -euo pipefail
NAV_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$NAV_ROOT"
uv sync --python 3.12 --frozen --extra test
NAV_BASE_IMAGE="${NAV_GAZEBO_IMAGE:-space_robotics_gz_envs:latest}"
if ! docker image inspect "$NAV_BASE_IMAGE" >/dev/null 2>&1; then
    NAV_BASE_IMAGE="andrejorsula/space_robotics_gz_envs:latest"
    if ! docker image inspect "$NAV_BASE_IMAGE" >/dev/null 2>&1; then
        docker pull "$NAV_BASE_IMAGE"
    fi
fi
uv run --frozen python tools/navigation/prepare_assets.py --image "$NAV_BASE_IMAGE"
NAV_COMPILER="${NAV_COMPILER_IMAGE:-ubuntu:24.04}"
if [[ -z "${NAV_COMPILER_IMAGE:-}" ]] && docker image inspect space-robotics-navigation:ionic >/dev/null 2>&1; then
    NAV_COMPILER=space-robotics-navigation:ionic
fi
# A tiny build context also works with classic Docker, which ignores per-file dockerignore.
NAV_CONTEXT="$(mktemp -d)"
trap 'rm -rf "$NAV_CONTEXT"' EXIT
mkdir -p "$NAV_CONTEXT/native/navigation"
cp Dockerfile.navigation "$NAV_CONTEXT/Dockerfile"
cp native/navigation/worker.cc "$NAV_CONTEXT/native/navigation/worker.cc"
docker build --network host --build-arg "PROJECT_IMAGE=$NAV_BASE_IMAGE" \
    --build-arg "COMPILER_IMAGE=$NAV_COMPILER" -t space-robotics-rl:ionic "$NAV_CONTEXT"
uv run --frozen --extra test pytest tests/navigation -q
RUN_GAZEBO_TESTS=1 uv run --frozen --extra test pytest tests/navigation/test_gazebo.py -q

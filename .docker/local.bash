#!/usr/bin/env bash
# Local Gazebo deployment: start [WORLD] [--headless], stop, status, logs, shell.
set -euo pipefail

REPOSITORY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${GAZEBO_IMAGE:-space_robotics_gz_envs:latest}"
CONTAINER="${GAZEBO_CONTAINER:-space-robotics-gazebo}"
ASSETS_VOLUME="space_robotics_gz_base_assets"
HOME_VOLUME="space_robotics_gz_base_home"

case "${1:-start}" in
    stop) docker stop --time 20 "$CONTAINER"; exit ;;
    status) docker ps -a --filter "name=^/${CONTAINER}$"; exit ;;
    logs) docker logs --tail 100 -f "$CONTAINER"; exit ;;
    shell) docker exec -it "$CONTAINER" bash; exit ;;
    start) [[ $# -eq 0 ]] || shift ;;
    *) echo "Usage: bash .docker/local.bash {start [WORLD] [--headless]|stop|status|logs|shell}" >&2; exit 2 ;;
esac

WORLD=moon
HEADLESS=false
for arg in "$@"; do
    case "$arg" in
        --headless) HEADLESS=true ;;
        moon|mars|moon_array|mars_array|_empty) WORLD="$arg" ;;
        *) echo "Unknown argument: $arg" >&2; exit 2 ;;
    esac
done

if docker container inspect "$CONTAINER" >/dev/null 2>&1; then
    echo "Container $CONTAINER already exists. Stop it before changing worlds."
    docker ps -a --filter "name=^/${CONTAINER}$"
    exit 0
fi
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "Missing image: $IMAGE. Pull the upstream image and set GAZEBO_IMAGE to its name." >&2
    exit 1
fi

OPTIONS=(
    --detach --rm --init --name "$CONTAINER"
    --log-opt max-size=10m --log-opt max-file=3
    --network host --ipc host --gpus all
    --label space_robotics_gz_envs.local=true
    --env "GZ_PARTITION=$CONTAINER"
    --env NVIDIA_DRIVER_CAPABILITIES=all
    --env __NV_PRIME_RENDER_OFFLOAD=1
    --env __GLX_VENDOR_LIBRARY_NAME=nvidia
    --env QT_X11_NO_MITSHM=1
    --volume "$REPOSITORY_DIR/worlds:/root/ws/worlds:ro"
    --volume "$REPOSITORY_DIR/scripts:/root/ws/scripts:ro"
    --volume "$ASSETS_VOLUME:/root/ws/assets/cache"
    --volume "$HOME_VOLUME:/root/.gz"
)
if [[ -f "$REPOSITORY_DIR/assets/srb_assets/scripts/blender/procgen_assets.py" ]]; then
    OPTIONS+=(--volume "$REPOSITORY_DIR/assets/srb_assets:/root/ws/assets/srb_assets:ro")
fi
if [[ -d /dev/dri ]]; then
    OPTIONS+=(--device /dev/dri:/dev/dri)
fi

COMMAND=(gz sim -r -v 3)
if "$HEADLESS"; then
    COMMAND+=(-s --headless-rendering)
else
    : "${DISPLAY:?No graphical session found; use start --headless.}"
    XAUTH_FILE="${XDG_RUNTIME_DIR:-/tmp}/${CONTAINER}.xauth"
    umask 077
    touch "$XAUTH_FILE"
    AUTH_DATA="$(xauth nlist "$DISPLAY")"
    if [[ -z "$AUTH_DATA" ]]; then
        echo "No X11 authentication found for $DISPLAY. Run from the desktop terminal or use --headless." >&2
        exit 1
    fi
    printf '%s\n' "$AUTH_DATA" | sed 's/^..../ffff/' | xauth -f "$XAUTH_FILE" nmerge -
    OPTIONS+=(
        --env "DISPLAY=$DISPLAY"
        --env "XAUTHORITY=$XAUTH_FILE"
        --env QT_QPA_PLATFORM=xcb
        --volume "$XAUTH_FILE:$XAUTH_FILE:ro"
        --volume /tmp/.X11-unix:/tmp/.X11-unix:ro
    )
fi
COMMAND+=("/root/ws/worlds/$WORLD.sdf")

docker run "${OPTIONS[@]}" "$IMAGE" "${COMMAND[@]}"
echo "Started $WORLD in $CONTAINER. Logs: bash .docker/local.bash logs"

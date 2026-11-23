#!/usr/bin/env bash
set -euo pipefail
project="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
mode="${1:?Expected train, view, or play}"; shift
watch=0
hours=3
args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --watch) watch=1; shift ;;
    --hours) hours="${2:?Provide training hours}"; shift 2 ;;
    *) args+=("$1"); shift ;;
  esac
done
case "$mode" in
  train) entry="$project/scripts/rsl_rl/train.py"; task=Oarbot-Iteration1-Mobile-v0; count=101 ;;
  view) entry="$project/scripts/view.py"; task=Oarbot-Iteration1-Mobile-v0; count=1; watch=1 ;;
  play) entry="$project/scripts/rsl_rl/play.py"; task=Oarbot-Iteration1-Mobile-Eval-v0; count=8; watch=1 ;;
  check) entry="$project/scripts/check_setup.py"; task=Oarbot-Iteration1-Mobile-v0; count=4 ;;
  *) echo "Unknown mode: $mode"; exit 2 ;;
esac
cd "$project"
python3 scripts/processes.py --guard
if [[ "$watch" == 1 ]] && ss -H -ltn 'sport = :49100' | grep -q .; then
  echo "Streaming port 49100 is occupied. Close the current Isaac Sim/Lab streaming session first."
  exit 2
fi
runtime="$HOME/.cache/oarbot_iteration_1"
mkdir -p "$runtime" "$project/logs"
kit_args="--portable-root $runtime --reset-user --/exts/isaacsim.core.throttling/enable_async=false --/app/asyncRendering=false --/app/asyncRenderingLowLatency=false"
# Use the rendering experience for offscreen camera runs too. The smaller
# headless experience can hang during native app startup on this installation.
visual=(--viz none --experience /opt/isaaclab/apps/isaaclab.python.rendering.kit)
if [[ "$watch" == 1 ]]; then
  visual=(--viz kit --livestream 1)
  kit_args+=" --/exts/omni.kit.livestream.app/primaryStream/publicIp=128.113.120.75 --/exts/omni.kit.livestream.app/primaryStream/signalPort=49100 --/exts/omni.kit.livestream.app/primaryStream/streamPort=47998 --/exts/omni.kit.livestream.app/primaryStream/streamType=webrtc --/exts/omni.kit.livestream.app/primaryStream/allowDynamicResize=false --/exts/omni.kit.livestream.app/primaryStream/targetFps=30 --/app/window/width=1280 --/app/window/height=720 --/app/renderer/resolution/width=1280 --/app/renderer/resolution/height=720"
fi
echo "Iteration 1: $mode. Camera rendering is ON, even without the WebRTC viewer."
exec env -u VIRTUAL_ENV -u CONDA_PREFIX PYTHONUNBUFFERED=1 PUBLIC_IP=128.113.120.75 \
  OARBOT_MAX_TRAIN_HOURS="$hours" PYTHONPATH="$project/source/iteration_1${PYTHONPATH:+:$PYTHONPATH}" \
  python3 -c 'import os,signal,sys; signal.signal(signal.SIGINT, signal.SIG_DFL); os.execv(sys.argv[1], sys.argv[1:])' \
  /opt/isaaclab/isaaclab.sh -p "$entry" --task "$task" --num_envs "$count" \
  --enable_cameras "${visual[@]}" --kit_args "$kit_args" "${args[@]}"

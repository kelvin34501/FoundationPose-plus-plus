#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p ./tmp/matplotlib ./tmp/object_pose_server

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate foundationposepp

export MPLCONFIGDIR="$SCRIPT_DIR/tmp/matplotlib"
export TMPDIR="$SCRIPT_DIR/tmp"

SYNC_CHANNEL="${SYNC_CHANNEL:-tcp://localhost:9600}"
PUB_CHANNEL="${PUB_CHANNEL:-tcp://*:9670}"
OBJECT_CONFIG="${OBJECT_CONFIG:-$SCRIPT_DIR/tmp/object_config.yml}"
CALIB_FILEDIR="${CALIB_FILEDIR:-$SCRIPT_DIR/../../common/calib_camera_w_mocap/calib_main/calib/calib__2026_0428_2011_55}"
CAMERA_INFO="${CAMERA_INFO:-103422070997=camera_top,818312071299=camera_side_1,011422072489=camera_side_2}"
VIDEO_SHAPE="${VIDEO_SHAPE:-1280x720}"
DISPLAY_SCALE="${DISPLAY_SCALE:-0.75}"
SAM_API_ENDPOINT="${SAM_API_ENDPOINT:-http://localhost:9002/hq_sam}"
SAM_API_AUTOSTART="${SAM_API_AUTOSTART:-1}"
SAM_API_SCRIPT="${SAM_API_SCRIPT:-$SCRIPT_DIR/src/WebAPI/hq_sam_api_alt.py}"
SAM_API_CHECKPOINT_PATH="${SAM_API_CHECKPOINT_PATH:-$SCRIPT_DIR/sam-hq/pretrained_checkpoints/sam_hq_vit_l.pth}"
SAM_API_MODEL_TYPE="${SAM_API_MODEL_TYPE:-vit_l}"
SAM_API_STARTUP_TIMEOUT="${SAM_API_STARTUP_TIMEOUT:-120}"

SAM_API_AUTOSTART_ARGS=()
if [[ "$SAM_API_AUTOSTART" == "1" || "$SAM_API_AUTOSTART" == "true" || "$SAM_API_AUTOSTART" == "yes" ]]; then
  SAM_API_AUTOSTART_ARGS=(--sam_api_autostart)
fi

exec python server.py \
  --server.video_shape "$VIDEO_SHAPE" \
  --server.sync_channel "$SYNC_CHANNEL" \
  --server.pub_channel "$PUB_CHANNEL" \
  --camera_info "$CAMERA_INFO" \
  --calib_filedir "$CALIB_FILEDIR" \
  --object_config "$OBJECT_CONFIG" \
  --activate_2d_tracker \
  --activate_kalman_filter \
  --sam_api_endpoint "$SAM_API_ENDPOINT" \
  "${SAM_API_AUTOSTART_ARGS[@]}" \
  --sam_api_script "$SAM_API_SCRIPT" \
  --sam_api_checkpoint_path "$SAM_API_CHECKPOINT_PATH" \
  --sam_api_model_type "$SAM_API_MODEL_TYPE" \
  --sam_api_startup_timeout "$SAM_API_STARTUP_TIMEOUT" \
  --display_scale "$DISPLAY_SCALE"

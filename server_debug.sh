#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p ./tmp/matplotlib ./tmp/object_pose_server

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate foundationposepp

export MPLCONFIGDIR="$SCRIPT_DIR/tmp/matplotlib"
export TMPDIR="$SCRIPT_DIR/tmp"

SYNC_CHANNEL="${SYNC_CHANNEL:-tcp://localhost:9602}"
PUB_CHANNEL="${PUB_CHANNEL:-tcp://*:9670}"
OBJECT_CONFIG="${OBJECT_CONFIG:-$SCRIPT_DIR/tmp/object_config.yml}"
CALIB_FILEDIR="${CALIB_FILEDIR:-$SCRIPT_DIR/../../common/calib_camera_w_mocap/calib_main/calib/calib__2026_0428_2011_55}"
CAMERA_INFO="${CAMERA_INFO:-103422070997=camera_top,818312071299=camera_side_1,011422072489=camera_side_2}"
VIDEO_SHAPE="${VIDEO_SHAPE:-1280x720}"
DISPLAY_SCALE="${DISPLAY_SCALE:-0.75}"

exec python server.py \
  --server.video_shape "$VIDEO_SHAPE" \
  --server.sync_channel "$SYNC_CHANNEL" \
  --server.pub_channel "$PUB_CHANNEL" \
  --camera_info "$CAMERA_INFO" \
  --calib_filedir "$CALIB_FILEDIR" \
  --object_config "$OBJECT_CONFIG" \
  --activate_2d_tracker \
  --activate_kalman_filter \
  --display_scale "$DISPLAY_SCALE"

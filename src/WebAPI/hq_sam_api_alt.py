import argparse
import os
from typing import List

import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
from PIL import Image

from segment_anything import SamPredictor, sam_model_registry


class SAMModel:

    def __init__(self, model_type: str, checkpoint_path: str, device: str):
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.sam.to(device=device)
        self.predictor = SamPredictor(self.sam)
        self.device = device

    def segment(self, image: np.ndarray, bbox: List[float]) -> np.ndarray:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"The input image should be RGB HWC, got shape {image.shape}")

        with torch.no_grad():
            self.predictor.set_image(image, image_format="RGB")
            masks, _, _ = self.predictor.predict(
                box=np.asarray(bbox, dtype=np.float32),
                multimask_output=False,
                hq_token_only=False,
            )

        mask = np.asarray(masks)
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask[0]
        if mask.ndim != 2:
            raise ValueError(f"SAM-HQ returned an unexpected mask shape: {mask.shape}")
        return mask.astype(bool)


app = FastAPI(title="SAM-HQ Alternate Web API")

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    else "cpu"
)


class Message(BaseModel):
    frame_path: str
    bbox_xywh: List[int]
    output_mask_path: str


@app.post("/hq_sam")
async def segment_image(message: Message):
    try:
        frame = Image.open(message.frame_path).convert("RGB")
        image_rgb = np.array(frame)
        height, width = image_rgb.shape[:2]

        x1 = max(0, min(width - 1, int(message.bbox_xywh[0])))
        y1 = max(0, min(height - 1, int(message.bbox_xywh[1])))
        x2 = max(x1, min(width, int(message.bbox_xywh[0] + message.bbox_xywh[2])))
        y2 = max(y1, min(height, int(message.bbox_xywh[1] + message.bbox_xywh[3])))
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid bbox_xywh for image shape {image_rgb.shape}: {message.bbox_xywh}")

        mask = sam_model.segment(image_rgb, [x1, y1, x2, y2])
        mask_uint8 = (mask * 255).astype(np.uint8)

        output_dir = os.path.dirname(message.output_mask_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        Image.fromarray(mask_uint8).save(message.output_mask_path)

        return Response(content="Mask generated successfully", media_type="text/plain", status_code=200)
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)})


@app.post("/hq_sam/binary")
async def segment_image_binary(request: Request, x: int, y: int, w: int, h: int):
    try:
        frame_png = await request.body()
        frame_bgr = cv2.imdecode(np.frombuffer(frame_png, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame_bgr is None:
            raise ValueError("Request body is not a valid PNG image")

        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        height, width = image_rgb.shape[:2]
        x1 = max(0, min(width - 1, x))
        y1 = max(0, min(height - 1, y))
        x2 = max(x1, min(width, x + w))
        y2 = max(y1, min(height, y + h))
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid bbox_xywh for image shape {image_rgb.shape}: {[x, y, w, h]}")

        mask = sam_model.segment(image_rgb, [x1, y1, x2, y2])
        mask_uint8 = (mask * 255).astype(np.uint8)
        encoded, mask_png = cv2.imencode(".png", mask_uint8)
        if not encoded:
            raise RuntimeError("Could not encode SAM-HQ mask as PNG")

        return Response(content=mask_png.tobytes(), media_type="image/png", status_code=200)
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the alternate SAM-HQ Web API")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="./sam-hq/pretrained_checkpoints/sam_hq_vit_l.pth",
    )
    parser.add_argument("--model_type", type=str, default=os.getenv("HQ_SAM_MODEL_TYPE", "vit_l"))
    parser.add_argument("--port", type=int, default=9002)
    args = parser.parse_args()

    sam_model = SAMModel(
        model_type=args.model_type,
        checkpoint_path=args.checkpoint_path,
        device=device,
    )

    uvicorn.run(app, host="127.0.0.1", port=args.port)

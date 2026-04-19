# XAI506 중간 프로젝트 - Vision Foundation Models

Vision foundation model 세 가지 (Depth Anything 3, OWLv2, Grounding DINO + SAM) 를 하나의 입력에 순차적으로 적용해보는 데모 프로젝트입니다.

## 어떤 걸 실행하면 되나요?

프로젝트의 메인은 두 개의 notebook 입니다. 아래 설치만 마치면 그대로 셀을 위에서부터 실행하면 됩니다. 두 notebook 모두 실행 결과가 이미 저장되어 있어서, 굳이 돌려보지 않아도 어떤 출력이 나오는지 확인할 수 있습니다.

### `foundation_models_demo.ipynb` (메인)

이미지 또는 영상 한 장을 입력으로 받아서 세 모델을 순차적으로 돌려보는 통합 데모. 상단의 `INPUT_PATH` 변수만 원하는 파일로 바꿔주세요.

### `depth_anything_3d_demo.ipynb`

영상에서 depth map GIF 와 3D point cloud 회전 GIF 를 생성하는 데모. DA3 의 3D 기능만 따로 보고 싶을 때 사용합니다.

## 설치

CUDA GPU 필요 (VRAM 8GB 이상 권장).

```bash
conda create -y -n deep-learning python=3.10
conda activate deep-learning

pip install "torch>=2" torchvision xformers
pip install transformers pillow scipy imageio opencv-python matplotlib numpy trimesh huggingface_hub

# Depth Anything 3 는 source install 이 필요합니다
cd depth_anything
git clone https://github.com/ByteDance-Seed/depth-anything-3.git repo
cd repo && pip install -e ".[app]"
```

## 사용한 모델들

### 1. Depth Anything 3 (ByteDance)

영상이나 이미지에서 픽셀별 깊이를 추정하고, 그 결과로 3D point cloud 까지 복원하는 모델.

- 기본: [`depth-anything/DA3-LARGE-1.1`](https://huggingface.co/depth-anything/DA3-LARGE-1.1) — 0.35B, VRAM 8GB 이상
- 가벼운 버전: [`depth-anything/DA3-BASE`](https://huggingface.co/depth-anything/DA3-BASE) — ~0.1B, VRAM 6GB 정도면 돌아감

VRAM 이 부족하면 notebook 상단의 `MODEL_ID` 변수를 `depth-anything/DA3-BASE` 로 바꿔주세요. CLI 로 쓸 때는 아래처럼 `MODEL` 환경변수로 지정할 수 있습니다.

```bash
MODEL=depth-anything/DA3-BASE ./depth_anything/scripts/run_video.sh inputs/video.mp4
```

### 2. OWLv2 (Google)

"monitor", "사람" 같은 자유로운 텍스트 쿼리로 객체를 탐지하는 zero-shot detector. 사전 정의된 카테고리에 얽매이지 않습니다.

- [`google/owlv2-base-patch16-ensemble`](https://huggingface.co/google/owlv2-base-patch16-ensemble) (기본)
- [`google/owlv2-large-patch14-ensemble`](https://huggingface.co/google/owlv2-large-patch14-ensemble) (large)

### 3. Grounding DINO + SAM (IDEA-Research + Meta)

텍스트 → bounding box → 픽셀 단위 마스크까지 이어지는 segmentation pipeline.

- Grounding DINO: [`IDEA-Research/grounding-dino-tiny`](https://huggingface.co/IDEA-Research/grounding-dino-tiny) — 텍스트에서 박스를 뽑음
- SAM: [`facebook/sam-vit-base`](https://huggingface.co/facebook/sam-vit-base) — 박스를 정밀한 마스크로 변환

프롬프트 없이 이미지의 모든 객체를 자동으로 세그먼트하는 SAM auto-mask 모드도 함께 제공됩니다.

## CLI 스크립트 (선택)

Notebook 없이 한 번만 돌려보고 싶을 때 사용하세요.

```bash
# 객체 탐지
python detection/scripts/detect.py <image> "monitor, keyboard, mouse"

# 텍스트 기반 세그멘테이션
python detection/scripts/grounded_sam.py <image> "monitor. keyboard. chair."

# 자동 세그멘테이션 (프롬프트 X)
python detection/scripts/auto_segment.py <image>

# 영상 depth + 3D point cloud
./depth_anything/scripts/run_video.sh <video.mp4>
```

각 스크립트 상단 docstring 에 더 자세한 옵션 설명이 있습니다.

---

## English

Three vision foundation models wired into a single demo. Start with **`foundation_models_demo.ipynb`** — run the cells top to bottom, and change `INPUT_PATH` near the top to point at your image/video. For 3D point cloud generation specifically, use **`depth_anything_3d_demo.ipynb`**.

Models used:
- **Depth Anything 3** — monocular depth estimation: [DA3-LARGE-1.1](https://huggingface.co/depth-anything/DA3-LARGE-1.1) (default) or [DA3-BASE](https://huggingface.co/depth-anything/DA3-BASE) for lower VRAM.
- **OWLv2** — zero-shot object detection: [owlv2-base-patch16-ensemble](https://huggingface.co/google/owlv2-base-patch16-ensemble).
- **Grounding DINO + SAM** — text-prompted segmentation: [grounding-dino-tiny](https://huggingface.co/IDEA-Research/grounding-dino-tiny) + [sam-vit-base](https://huggingface.co/facebook/sam-vit-base).

A CUDA GPU is required. See the 설치 section above for the full install command.

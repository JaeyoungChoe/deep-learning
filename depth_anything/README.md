# Depth Anything 3

[ByteDance-Seed/depth-anything-3](https://github.com/ByteDance-Seed/depth-anything-3) 를
이용해 입력 video 에서 depth / 3D 결과를 추출한다.

기본 모델 variant: **`depth-anything/DA3-LARGE-1.1`** (0.35B params).

## 폴더 구조

```
depth_anything/
├── README.md          # (이 파일)
├── repo/              # depth-anything-3 git clone (editable install)
├── inputs/            # 입력 video 들 (사용자가 직접 채움, .gitignore)
├── outputs/           # 실행 결과 (.gitignore)
└── scripts/
    └── run_video.sh   # da3 video CLI wrapper
```

## 환경 세팅 (최초 1회)

전체 프로젝트는 `deep-learning` conda env 를 공유한다.

```bash
# 1. conda env 생성
conda create -y -n deep-learning python=3.10
conda activate deep-learning

# 2. depth-anything-3 clone
cd /home/jaeyoung/workspace/deep-learning/depth_anything
git clone https://github.com/ByteDance-Seed/depth-anything-3.git repo
cd repo

# 3. PyTorch + xformers (CUDA 환경 가정)
pip install "torch>=2" torchvision xformers

# 4. editable install ([app] extras: gradio 포함, gsplat 제외)
pip install -e ".[app]"
```

`[all]` extras 는 `[app] + [gs]` 인데, `[gs]` 의 `gsplat` 은 build isolation
환경에서 `torch` 를 못 찾아 빌드가 실패한다 (검증됨). video depth 추출만 할
때는 필요 없으므로 `[app]` 만 설치한다. 3D Gaussian rendering 이 필요해지면
다음과 같이 별도 설치:

```bash
pip install --no-build-isolation \
    git+https://github.com/nerfstudio-project/gsplat.git@0b4dddf04cb687367602c01196913cde6a743d70
```

> 참고: `da3 --help` 실행 시 `gsplat` 미설치 경고가 뜨지만, video depth 파이프라인에는
> 영향 없다.

## 환경 검증

```bash
conda activate deep-learning
python -c "import torch, xformers, depth_anything_3; print(torch.__version__, torch.cuda.is_available())"
da3 --help
da3 video --help
```

`torch.cuda.is_available()` 가 `True` 여야 한다.

## 실행

1. 입력 video 를 `inputs/` 에 복사
2. 실행:

```bash
conda activate deep-learning
cd /home/jaeyoung/workspace/deep-learning/depth_anything
./scripts/run_video.sh inputs/my_video.mp4
# 결과: outputs/my_video/ 에 .glb 등 생성
```

3. feature 시각화까지 원하면 `glb-feat_vis` 포맷 사용:

```bash
./scripts/run_video.sh inputs/my_video.mp4 glb-feat_vis
```

첫 실행 시 Hugging Face Hub 에서 `depth-anything/DA3-LARGE-1.1` 가중치를 자동
다운로드한다 (수 GB).

## 모델 variant 변경

`scripts/run_video.sh` 의 `--model-dir` 인자를 다른 HF repo id 로 바꾸면 된다.

| Variant | HF repo id | Params | 권장 VRAM |
|---|---|---|---|
| BASE | `depth-anything/DA3-BASE` | ~0.1B | 6GB+ |
| LARGE (default) | `depth-anything/DA3-LARGE-1.1` | 0.35B | 8GB+ |
| GIANT | `depth-anything/DA3-GIANT` | ~1B | 12GB+ |

(정확한 repo id / 사이즈는 [공식 README](https://github.com/ByteDance-Seed/depth-anything-3) 참고)

## 트러블슈팅

- **xformers / torch CUDA 버전 mismatch**
  → 시스템 CUDA 에 맞는 torch wheel 로 재설치
  (`pip install torch --index-url https://download.pytorch.org/whl/cuXXX`).
- **VRAM 부족**
  → `run_video.sh` 의 `--model-dir` 을 `depth-anything/DA3-BASE` 로 교체.
- **HF 다운로드 실패**
  → `huggingface-cli login` 또는 `HF_HOME` 캐시 경로 확인.

# DeepLabV3+ with CBAM for Road Surface Segmentation

## 1. 프로젝트 개요

본 프로젝트는 DeepLabV3+ 아키텍처에 **CBAM (Convolutional Block Attention Module)**을 적용하여 도로 표면과 같은 객체를 분할(Segmentation)하는 모델을 구현합니다.

학습된 모델은 TensorFlow Lite(.tflite) 형식으로 변환 및 양자화(INT8)하여, 모바일이나 임베디드 환경과 같은 리소스가 제한된 장치에서도 효율적으로 동작할 수 있도록 최적화하는 전체 파이프라인을 포함합니다.

---

## 2. 주요 기능

- **DeepLabV3+ & CBAM**: 강력한 분할 성능을 자랑하는 DeepLabV3+ 모델에 어텐션 메커니즘(CBAM)을 추가하여 정확도를 향상시켰습니다.
- **전이 학습 (Transfer Learning)**: ImageNet으로 사전 학습된 MobileNetV2 백본을 사용하여 더 빠르고 안정적으로 모델을 학습합니다.
- **2단계 학습 전략**:
  1.  **특징 추출**: 백본을 동결하고 새로 추가된 분할 헤드만 학습합니다.
  2.  **미세 조정**: 전체 모델을 낮은 학습률로 추가 학습하여 성능을 극대화합니다.
- **TFLite 변환 및 양자화**: 학습된 모델을 Float32 및 INT8 정수형 TFLite 모델로 변환하여, 모델의 크기를 줄이고 추론 속도를 높입니다.
- **다양한 추론 결과 저장**: 오버레이, 비교 이미지 등 다양한 시각화 옵션을 제공합니다.

---

## 3. 파일 구조

```bash
.
├── checkpoints/                  # 학습된 모델(.keras)이 저장되는 폴더
│   └── best_model.keras
├── data/                         # 데이터셋 폴더
│   └── tfrecords/                # TFRecord 파일들이 위치하는 곳
├── exported_models/              # 내보내기된 모델이 저장되는 폴더
│   ├── Model/                    # SavedModel 형식
│   └── tflite_models/            # TFLite 형식 (.tflite)
├── test_images/                  # 추론에 사용할 테스트 이미지 폴더
├── tflite_results/               # TFLite 추론 결과가 저장되는 폴더
├── main_cbam.py                  # 학습, 평가, 모델 내보내기 메인 스크립트
├── inference_tflite.py           # TFLite 모델 추론 스크립트
├── create_tfrecords_from_coco.py # COCO 데이터셋을 TFRecord로 변환하는 스크립트
├── deeplab_v3_plus_cbam.py       # 모델 아키텍처 정의
├── segmentation_dataset.py       # 데이터셋 로더 정의
├── requirements.txt              # 필요 라이브러리 목록
└── README.md                     # 프로젝트 설명 파일 (현재 문서)
```

---

## 4. 환경 설정

`requirements.txt` 파일을 사용하여 필요한 라이브러리를 설치합니다. `pycocotools`도 함께 설치합니다.

```bash
pip install -r requirements.txt
pip install pycocotools
```

---

## 5. 전체 워크플로우

### 1단계: 데이터 다운로드

1.  [Roboflow](https://roboflow.com/)와 같은 플랫폼을 사용하여 이미지에 대한 분할(Segmentation) 주석(Annotation) 작업을 수행합니다.
2.  주석 작업이 완료되면, 데이터를 **COCO Segmentation** 형식으로 내보내기(Export)하여 다운로드합니다.
3.  다운로드한 데이터의 압축을 풀고, 아래와 같이 프로젝트 내에 폴더를 구성합니다.

```bash
./data/coco/
├── train/
│   ├── _annotations.coco.json
│   └── (학습용 이미지들).jpg
└── valid/
    ├── _annotations.coco.json
    └── (검증용 이미지들).jpg
```

### 2단계: 데이터 변환 (COCO to TFRecord)

다운로드한 COCO 형식의 데이터셋을 모델 학습에 사용하기 위해 TFRecord 형식으로 변환합니다.

1.  **`create_tfrecords_from_coco.py` 스크립트 수정**:
    제공된 `create_tfrecords_from_coco.py` 파일을 열어 상단의 `DATA_SPLITS` 딕셔너리에 있는 경로들을 자신의 환경에 맞게 수정합니다.

    ```python
    # 예시
    DATA_SPLITS = {
        'train': {
            'json_path': './data/coco/train/_annotations.coco.json',
            'image_dir': './data/coco/train'
        },
        'valid': {
            'json_path': './data/coco/valid/_annotations.coco.json',
            'image_dir': './data/coco/valid'
        },
    }
    OUTPUT_DIR = './data/tfrecords'
    ```

2.  **변환 스크립트 실행**:
    수정이 완료되면, 터미널에서 아래 명령어를 실행하여 변환을 시작합니다.

    ```bash
    python create_tfrecords_from_coco.py
    ```

3.  **변환 결과**:
    변환이 완료되면 `OUTPUT_DIR`로 지정한 폴더(예: `./data/tfrecords/`) 안에 `images-train-*.tfrecord`와 `images-valid-*.tfrecord` 파일들이 생성됩니다.

### 3단계: 모델 학습 (`train` 모드)

TFRecord로 변환된 데이터셋을 사용하여 모델을 학습합니다.

- **실행 명령어**:
  ```bash
  python main_cbam.py --mode train --epoch 50 --initial_epochs 10 --batch 8
  ```

- **주요 인자**:
  - `--epoch`: 전체 학습 에포크 수 (기본값: 50)
  - `--initial_epochs`: 전체 에포크 중, 1단계(특징 추출)에 사용할 에포크 수 (기본값: 10)
  - `--batch`: 배치 크기 (기본값: 8)
  - `--learning_rate`: 초기 학습률 (기본값: 1e-3)
  - `--resume_training`: `./checkpoints/best_model.keras`에서 학습을 이어서 진행합니다.
  - `--no_transfer`: 전이 학습을 사용하지 않고 처음부터 학습합니다.

### 4단계: 모델 내보내기 (`export` 모드)

학습된 `.keras` 모델을 SavedModel, Float32 TFLite, INT8 TFLite 형식으로 모두 변환합니다.

- **실행 명령어**:
  ```bash
  python main_cbam.py --mode export
  ```

### 5단계: 모델 추론 (`inference_tflite.py`)

`export` 모드로 생성된 `.tflite` 파일을 사용하여 이미지 분할을 수행하고 성능을 확인합니다.

- **실행 명령어**:
  ```bash
  # 예시: INT8 양자화 모델 사용
  python inference_tflite.py \
    --model_path ./exported_models/tflite_models/Model_quant.tflite \
    --input_path ./test_images \
    --output_dir ./tflite_results \
    --save_mode comparison
  ```

- **주요 인자**:
  - `--model_path`: 사용할 `.tflite` 모델 파일의 경로 (**필수**)
  - `--input_path`: 추론할 이미지 또는 폴더 경로 (**필수**)
  - `--output_dir`: 결과 이미지를 저장할 폴더 경로 (**필수**)
  - `--save_mode`: 결과 저장 방식 (기본값: `overlay`)
    - `overlay`: 원본 이미지에 마스크를 겹쳐서 저장
    - `comparison`: 원본, 마스크, 오버레이 이미지를 나란히 붙여서 저장
    - `all`: `overlay`와 `comparison` 결과를 모두 저장

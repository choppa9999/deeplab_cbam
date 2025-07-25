import os
import argparse
import tensorflow as tf
from tensorflow import keras
import numpy as np

# --- 의존성 ---
# 이 스크립트를 실행하려면, 동일한 폴더에 아래 파일들이 필요합니다.
# 1. deeplab_v3_plus_cbam.py (모델 아키텍처를 불러오기 위함)
# 2. segmentation_dataset.py (데이터 로더를 사용하기 위함)
from deeplab_v3_plus_cbam import DeeplabV3PlusWithCBAM, CBAM
from segmentation_dataset import SegmentationDataset


# --- 사용자 정의 평가 지표 (Metric) ---
class MeanIoUWithIgnore(tf.keras.metrics.Metric):
    """
    특정 레이블(ignore_label)을 무시하고 mIoU를 계산하는 사용자 정의 클래스입니다.
    """

    def __init__(self, num_classes, ignore_label=255, name='mean_iou_with_ignore', dtype=None):
        super().__init__(name=name, dtype=dtype)
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        # TensorFlow의 기본 MeanIoU를 내부적으로 사용합니다.
        self.iou_metric = keras.metrics.MeanIoU(num_classes=num_classes)

    def update_state(self, y_true, y_pred, sample_weight=None):
        """매 스텝마다 라벨과 예측값을 받아와 상태를 업데이트합니다."""
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.argmax(y_pred, axis=-1, output_type=tf.int32)

        # y_true의 마지막 차원(채널)을 제거하여 y_pred와 shape를 맞춥니다.
        # (B, H, W, 1) -> (B, H, W)
        y_true_squeezed = tf.squeeze(y_true, axis=-1)

        # ignore_label에 해당하는 픽셀을 찾습니다. mask의 shape는 (B, H, W)가 됩니다.
        mask = tf.not_equal(y_true_squeezed, self.ignore_label)

        # 마스크를 적용하여 ignore_label 픽셀을 계산에서 제외합니다.
        y_true_masked = tf.boolean_mask(y_true_squeezed, mask)
        y_pred_masked = tf.boolean_mask(y_pred, mask)

        # 제외된 데이터로 내부 IoU 메트릭을 업데이트합니다.
        self.iou_metric.update_state(y_true_masked, y_pred_masked)

    def result(self):
        """최종 mIoU 결과를 반환합니다."""
        return self.iou_metric.result()

    def reset_state(self):
        """상태를 초기화합니다."""
        self.iou_metric.reset_state()

    def get_config(self):
        """클래스 설정을 저장하고 불러올 때 사용됩니다."""
        config = super().get_config()
        config.update({
            'num_classes': self.num_classes,
            'ignore_label': self.ignore_label
        })
        return config


def run_evaluation(args):
    """
    테스트 데이터셋에 대한 모델 평가를 수행합니다.
    """
    print("\n--- 테스트셋 평가 모드 시작 ---")

    # 1. 모델 불러오기
    if not os.path.exists(args.model_path):
        print(f"오류: '{args.model_path}'에서 모델 파일을 찾을 수 없습니다.")
        return

    print(f"학습된 모델 로드 중: {args.model_path}")
    try:
        # 모델 로드 시, 사용자 정의 클래스들을 알려주어야 합니다.
        custom_objects = {
            'MeanIoUWithIgnore': MeanIoUWithIgnore,
            'DeeplabV3PlusWithCBAM': DeeplabV3PlusWithCBAM,
            'CBAM': CBAM
        }
        model = keras.models.load_model(args.model_path, custom_objects=custom_objects, compile=False)
        print("모델 로드 완료.")
    except Exception as e:
        print(f"\n오류: 모델을 로드하는 중 문제가 발생했습니다: {e}")
        return

    # 2. 테스트 데이터셋 준비
    print("테스트 데이터셋 로드 중...")
    input_shape = (args.image_height, args.image_width)
    try:
        # 'test' 서브셋을 사용하여 데이터 로더를 생성합니다.
        test_loader = SegmentationDataset(
            dataset_name='images',  # TFRecord 생성 시 사용한 이름
            dataset_dir=args.dataset_dir,
            subset='test',
            image_height=input_shape[0],
            image_width=input_shape[1],
            is_training=False  # 평가는 항상 False
        )

        total_test_samples = test_loader.get_num_data()
        if total_test_samples == 0:
            print(f"오류: 테스트 데이터셋에 샘플이 없습니다. 경로를 확인하세요: {args.dataset_dir}")
            return

        test_ds = test_loader.make_batch(args.batch_size)
        print(f"총 {total_test_samples}개의 테스트 샘플을 찾았습니다.")
    except FileNotFoundError as e:
        print(f"오류: 테스트 데이터셋 파일을 찾을 수 없습니다. {e}")
        return

    # 3. 모델 컴파일 및 평가
    num_classes = model.layers[-1].filters
    print(f"모델의 클래스 수: {num_classes}")

    # 평가에 사용할 지표들을 정의합니다.
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    accuracy_metric = keras.metrics.SparseCategoricalAccuracy(name='pixel_accuracy')
    mean_iou_metric = MeanIoUWithIgnore(num_classes=num_classes)

    model.compile(loss=loss_fn, metrics=[accuracy_metric, mean_iou_metric])

    print("\n모델 평가를 시작합니다...")
    results = model.evaluate(test_ds,
                             steps=total_test_samples // args.batch_size if args.batch_size > 0 else total_test_samples)

    # 4. 결과 출력
    print("\n" + "=" * 40)
    print("      테스트 데이터셋 최종 평가 결과")
    print("=" * 40)
    for name, value in zip(model.metrics_names, results):
        if name == 'loss':
            print(f"  - 손실 (Test Loss)            : {value:.4f}")
        elif name == 'pixel_accuracy':
            print(f"  - 픽셀 정확도 (Test Accuracy)  : {value:.4f}")
        elif name == 'mean_iou_with_ignore':
            print(f"  - 최종 mIoU (Test mIoU)        : {value:.4f}")
    print("=" * 40)
    print("\n--- 평가 완료 ---")


def main():
    parser = argparse.ArgumentParser(description="학습된 모델을 테스트 데이터셋으로 평가하는 스크립트")

    parser.add_argument('--model_path', type=str, default='./checkpoints/best_model.keras',
                        help="평가할 .keras 모델 파일 경로")
    parser.add_argument('--dataset_dir', type=str, default='./data/tfrecords',
                        help="TFRecord 파일들이 저장된 폴더 경로")
    parser.add_argument('--image_height', type=int, default=512,
                        help="모델 입력 이미지 높이")
    parser.add_argument('--image_width', type=int, default=512,
                        help="모델 입력 이미지 너비")

    # --- [수정] ---
    # GPU 메모리 부족(OOM) 오류를 방지하기 위해 기본 배치 크기를 4로 줄입니다.
    # 사용자의 GPU VRAM에 따라 2 또는 1로 더 줄여야 할 수 있습니다.
    parser.add_argument('--batch_size', type=int, default=4,
                        help="평가에 사용할 배치 크기")

    args = parser.parse_args()
    run_evaluation(args)


if __name__ == '__main__':
    main()

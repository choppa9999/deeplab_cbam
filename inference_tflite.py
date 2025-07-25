import os
import argparse
import numpy as np
import tensorflow as tf
from PIL import Image
from tqdm import tqdm


def create_pascal_label_colormap():
    """
    PASCAL VOC 데이터셋의 분할 마스크에 사용되는 것과 유사한 컬러맵을 생성합니다.
    이 컬러맵은 각 클래스 ID에 고유한 색상을 매핑하여 시각화를 돕습니다.
    """
    colormap = np.zeros((256, 3), dtype=int)
    ind = np.arange(256, dtype=int)
    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((ind >> channel) & 1) << shift
        ind >>= 3
    return colormap


def label_to_color_image(label):
    """
    세그멘테이션 라벨(클래스 ID 배열)을 사람이 볼 수 있는 컬러 이미지로 변환합니다.

    Args:
        label (np.ndarray): 2D 배열 형태의 클래스 ID 맵.

    Returns:
        np.ndarray: RGB 색상으로 변환된 이미지 배열.
    """
    if label.ndim != 2:
        raise ValueError(f'label은 2D 배열이어야 합니다. 현재 shape: {label.shape}')
    colormap = create_pascal_label_colormap()
    return colormap[label].astype(np.uint8)


def run_inference(args):
    """
    TFLite 모델을 사용하여 이미지에 대한 세그멘테이션 추론을 수행하고 결과를 저장합니다.

    Args:
        args (argparse.Namespace): 스크립트 실행 시 전달된 인자.
    """
    model_path = args.model_path
    input_path = args.input_path
    output_dir = args.output_dir
    save_mode = args.save_mode

    print(f"TFLite 모델 로드 중: {model_path}")
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
    except Exception as e:
        print(f"오류: TFLite 모델 파일을 로드할 수 없습니다. 경로를 확인하세요: {e}")
        return

    # 모델의 입력 및 출력 세부 정보 가져오기
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    # 모델이 요구하는 입력 이미지의 높이와 너비
    _, height, width, _ = input_details['shape']
    print(f"모델 입력 크기: ({height}, {width})")

    # 처리할 이미지 경로 목록 생성
    image_paths = []
    supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
    if os.path.isdir(input_path):
        for filename in os.listdir(input_path):
            if filename.lower().endswith(supported_extensions):
                image_paths.append(os.path.join(input_path, filename))
    elif os.path.isfile(input_path) and input_path.lower().endswith(supported_extensions):
        image_paths.append(input_path)

    if not image_paths:
        print(f"오류: '{input_path}' 경로에서 처리할 이미지를 찾을 수 없습니다.")
        return

    os.makedirs(output_dir, exist_ok=True)
    print(f"총 {len(image_paths)}개의 이미지를 처리합니다. 결과는 '{output_dir}'에 저장됩니다.")

    # 모델이 요구하는 입력 데이터 타입 정보 한 번만 출력
    input_type = input_details['dtype']
    print(f"\n💡 정보: 모델이 예상하는 입력 타입은 '{np.dtype(input_type).name}' 입니다.")

    for image_path in tqdm(image_paths, desc="TFLite 추론 진행률"):
        try:
            original_image_pil = Image.open(image_path).convert('RGB')
            original_size = original_image_pil.size

            # --- [핵심 수정] ---
            # 학습/양자화 과정과 동일한 전처리 방식을 적용합니다.
            # 1. 이미지를 TensorFlow 텐서로 변환합니다.
            image_tensor = tf.convert_to_tensor(original_image_pil, dtype=tf.float32)

            # 2. 학습 때와 동일한 'BILINEAR' 방식으로 이미지 크기를 조정합니다.
            #    이것이 양자화 모델의 성능을 결정하는 핵심 요소입니다.
            resized_image_tensor = tf.image.resize(
                image_tensor, [height, width], method=tf.image.ResizeMethod.BILINEAR
            )

            # 3. 모델의 입력 타입에 맞게 데이터 타입을 변환하고 배치 차원을 추가합니다.
            if input_type == np.uint8:
                # UINT8 양자화 모델의 경우
                input_data = tf.cast(resized_image_tensor, tf.uint8)
                input_data = tf.expand_dims(input_data, axis=0)
            else:
                # FLOAT32 모델의 경우 (필요 시 정규화)
                # MobileNet 계열 모델은 보통 [-1, 1] 범위로 정규화합니다.
                # 학습 시 사용한 정규화 방식을 그대로 적용해야 합니다.
                input_data = tf.expand_dims(resized_image_tensor, axis=0)
                # 예: input_data = (input_data / 127.5) - 1.0

            # 준비된 데이터를 TFLite 모델에 입력
            interpreter.set_tensor(input_details['index'], input_data)
            interpreter.invoke()

            # 추론 결과 가져오기
            output_data = interpreter.get_tensor(output_details['index'])

            # 후처리: 배치 차원 제거 및 원본 크기로 복원
            seg_map = np.squeeze(output_data).astype(np.uint8)
            seg_map_pil = Image.fromarray(seg_map)
            resized_seg_map = seg_map_pil.resize(original_size, Image.NEAREST)

            # 시각화를 위한 컬러 마스크 생성 및 오버레이
            color_mask_image = label_to_color_image(np.array(resized_seg_map))
            original_image_np = np.array(original_image_pil)
            overlayed_image = (original_image_np * 0.6 + color_mask_image * 0.4).astype(np.uint8)

            # 결과 저장
            base_filename = os.path.basename(image_path)
            filename_no_ext, _ = os.path.splitext(base_filename)

            if save_mode in ['comparison', 'all']:
                comparison_image = np.hstack((original_image_np, color_mask_image, overlayed_image))
                comparison_path = os.path.join(output_dir, f"{filename_no_ext}_comparison.png")
                Image.fromarray(comparison_image).save(comparison_path)

            if save_mode in ['overlay', 'all']:
                output_path = os.path.join(output_dir, f"{filename_no_ext}_overlay.png")
                Image.fromarray(overlayed_image).save(output_path)

        except Exception as e:
            print(f"\n오류: '{image_path}' 처리 중 문제가 발생했습니다: {e}")
            import traceback
            traceback.print_exc()

    print("\n--- 모든 추론 완료 ---")


def main():
    parser = argparse.ArgumentParser(description="TFLite DeepLabV3+ CBAM 모델 추론 스크립트")

    # required=True를 제거하고 default 값을 설정합니다.
    parser.add_argument('--model_path',
                        type=str,
                        default='./exported_models/tflite_models/Model_quant.tflite',
                        help="추론에 사용할 .tflite 모델 파일 경로. 지정하지 않으면 기본값이 사용됩니다.")

    # 나머지 인자들은 required=True를 유지하거나 필요에 따라 default를 설정할 수 있습니다.
    parser.add_argument('-i', '--input_path', type=str, required=True,
                        help="추론할 입력 이미지 또는 폴더 경로")

    # 출력 경로도 짧은 별명을 추가해 줍니다.
    parser.add_argument('-o', '--output_dir', type=str, required=True,
                        help="추론 결과를 저장할 폴더 경로")


    parser.add_argument(
        '--save_mode',
        type=str,
        default='overlay',
        choices=['overlay', 'comparison', 'all'],
        help="추론 결과 저장 방식: 'overlay' (오버레이 결과만), 'comparison' (원본, 마스크, 오버레이 비교), 'all' (모두 저장)"
    )
    args = parser.parse_args()

    run_inference(args)


if __name__ == '__main__':
    main()

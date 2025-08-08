# main_cbam.py

import os
import datetime
import argparse
import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
import io
import re

from matplotlib import pyplot as plt
from PIL import Image
from tqdm import tqdm 

import psutil

from deeplab_v3_plus_cbam import DeeplabV3PlusWithCBAM, CBAM
from segmentation_dataset import SegmentationDataset

class EpochMetricsLogger(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        train_acc = logs.get('pixel_accuracy')
        val_acc = logs.get('val_pixel_accuracy')
        train_iou = logs.get('mean_iou_with_ignore')
        val_iou = logs.get('val_mean_iou_with_ignore')

        print(f"\n\nEpoch {epoch + 1} 완료" + "="*40)
        if train_acc is not None:
            print(f"  - 학습 정확도 (Train Accuracy)   : {train_acc:.4f}")
        if train_iou is not None:
            print(f"  - 학습 IoU (Train IoU)           : {train_iou:.4f}")
        if val_acc is not None:
            print(f"  - 검증 정확도 (Validation Accuracy): {val_acc:.4f}")
        if val_iou is not None:
            print(f"  - 검증 IoU (Validation IoU)      : {val_iou:.4f}")
        print("="*55 + "\n")


def setup_environment(args):
    tf.config.optimizer.set_jit(False)
    print("[정보] XLA JIT 컴파일러가 비활성화되었습니다.")

    if args.mixed_precision:
        policy = keras.mixed_precision.Policy('mixed_float16')
        keras.mixed_precision.set_global_policy(policy)
        print("혼합 정밀도(Mixed Precision) 학습이 활성화되었습니다.")

    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            for gpu in physical_devices:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("[정보] GPU 메모리 동적 할당이 활성화되었습니다.")
        except RuntimeError as e:
            print(f"GPU 메모리 설정 중 오류 발생: {e}")
    print(f"TensorFlow 버전: {tf.__version__}")
    print(f"사용 가능한 GPU 수: {len(physical_devices)}")

INPUT_SHAPE = (512, 512, 3)
IGNORE_LABEL = 255
DATASET_NAME = 'images'
DATASET_DIR = './data/tfrecords'
OUTPUT_STRIDE = 16
MODEL_DIR = './checkpoints'
EXPORT_SAVEDMODEL_DIR = './exported_models/Model'
TFLITE_MODEL_PATH = './exported_models/tflite_models/Model.tflite'
ALLOWED_CLASSES = ['dry', 'humid', 'slush', 'snow', 'wet']

def create_pascal_label_colormap():
    colormap = np.zeros((256, 3), dtype=np.uint8)
    ind = np.arange(256, dtype=np.uint8)
    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((ind >> channel) & 1) << shift
        ind >>= 3
    return colormap

def label_to_color_image(label):
    if label.ndim != 2:
        raise ValueError('label은 2D 배열이어야 합니다. 현재 shape: ', label.shape)
    colormap = create_pascal_label_colormap()
    return colormap[label]

def create_legend_image():
    """하드코딩된 클래스 목록을 사용하여 범례 이미지를 생성합니다."""
    print("범례(legend) 생성 중...")
    
    new_id_to_name = {i + 1: name for i, name in enumerate(ALLOWED_CLASSES)}
    legend_labels = {0: 'background', **new_id_to_name}
    colormap = create_pascal_label_colormap()

    fig, ax = plt.subplots(figsize=(3, len(legend_labels) * 0.35), dpi=120)
    ax.set_title("Legend", fontweight='bold')
    
    for i, (class_id, class_name) in enumerate(sorted(legend_labels.items())):
        color = colormap[class_id] / 255.0
        ax.add_patch(plt.Rectangle((0, i), 1, 1, color=color))
        ax.text(1.2, i + 0.5, f"{class_id}: {class_name}", va='center', fontsize=10)

    ax.set_ylim(len(legend_labels), -0.5)
    ax.set_xlim(0, 4)
    ax.axis('off')

    try:
        io_buf = io.BytesIO()
        fig.savefig(io_buf, format='png', bbox_inches='tight', dpi=150)
        io_buf.seek(0)
        legend_img = np.array(Image.open(io_buf))
        plt.close(fig)
        print("범례 이미지 생성 완료.")
        if legend_img.shape[2] == 4:
            legend_img = legend_img[:, :, :3]
        return legend_img
    except Exception as e:
        print(f"오류: 범례 이미지를 생성하는 중 문제가 발생했습니다: {e}")
        plt.close(fig)
        return None

def find_best_checkpoint(checkpoint_dir):
    """지정된 디렉토리에서 가장 높은 IoU 점수를 가진 체크포인트 파일을 찾습니다."""
    best_file = None
    best_iou = -1.0
    
    if not os.path.isdir(checkpoint_dir):
        return None

    pattern = re.compile(r"model-ep\d+-iou([\d\.]+)\.keras")
    
    for f in os.listdir(checkpoint_dir):
        match = pattern.match(f)
        if match:
            iou = float(match.group(1))
            if iou > best_iou:
                best_iou = iou
                best_file = f
                
    if best_file:
        return os.path.join(checkpoint_dir, best_file)
    elif os.path.exists(os.path.join(checkpoint_dir, "best_model.keras")):
        return os.path.join(checkpoint_dir, "best_model.keras")
    return None

class MeanIoUWithIgnore(tf.keras.metrics.Metric):
    def __init__(self, num_classes, name='mean_iou_with_ignore', dtype=None):
        super().__init__(name=name, dtype=dtype)
        self.num_classes = num_classes
        self.iou_metric = keras.metrics.MeanIoU(num_classes=num_classes)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, tf.float32)
            mask = tf.not_equal(sample_weight, 0)
            y_true_masked = tf.boolean_mask(y_true, mask)
            y_pred_masked = tf.boolean_mask(y_pred, mask)
            self.iou_metric.update_state(y_true_masked, y_pred_masked)
        else:
            self.iou_metric.update_state(y_true, y_pred)

    def result(self):
        return self.iou_metric.result()

    def reset_state(self):
        self.iou_metric.reset_state()
    
    def get_config(self):
        config = super().get_config()
        config.update({'num_classes': self.num_classes})
        return config

def get_loss_and_metrics(num_classes):
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
    accuracy_metric = keras.metrics.SparseCategoricalAccuracy(name='pixel_accuracy')
    mean_iou_metric = MeanIoUWithIgnore(num_classes=num_classes)
    return loss_fn, accuracy_metric, mean_iou_metric

def train(args):
    print(f"\n--- 학습 모드 시작 (Epochs: {args.epochs}, Batch Size: {args.batch_size}) ---")
    
    checkpoint_path = os.path.join(MODEL_DIR, "best_model.keras")

    if os.path.exists(checkpoint_path) and args.resume_training:
        print("\n" + "="*60)
        print(f"이전 학습에서 저장된 모델을 찾았습니다: {checkpoint_path}")
        model = keras.models.load_model(checkpoint_path, custom_objects={'MeanIoUWithIgnore': MeanIoUWithIgnore, 'DeeplabV3PlusWithCBAM': DeeplabV3PlusWithCBAM, 'CBAM': CBAM})
        print("모델과 옵티마이저 상태를 모두 로드하여 학습을 이어서 시작합니다.")
        print("="*60 + "\n")
        
        num_classes = model.layers[-1].filters
    else:
        print("\n" + "="*60)
        if args.resume_training:
            print("경고: --resume_training이 지정되었지만, 저장된 모델 파일이 없습니다.")
        print("처음부터 새로운 학습을 시작합니다.")
        
        train_loader_info = SegmentationDataset(DATASET_NAME, DATASET_DIR, 'trainval', INPUT_SHAPE[0], INPUT_SHAPE[1], is_training=True)
        num_classes = train_loader_info.num_classes
        if num_classes <= 1:
            print(f"오류: 데이터셋에서 유효한 클래스를 찾지 못했습니다.")
            return

        print(f"\n[정보] Roboflow에서 가져온 {num_classes - 1}개의 클래스와 '라벨링되지 않은 배경(ID: 0)'을 포함하여,")
        print(f"       모델은 총 {num_classes}개의 대상을 구분하도록 학습됩니다.")
        
        model = DeeplabV3PlusWithCBAM(num_classes=num_classes, input_shape=INPUT_SHAPE, output_stride=OUTPUT_STRIDE)
        
        model.build(input_shape=(None, *INPUT_SHAPE))
        
        lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(args.learning_rate, 1, 1e-6, 0.9)
        optimizer = tf.keras.optimizers.AdamW(lr_schedule, args.weight_decay)
        loss_fn, acc_metric, iou_metric = get_loss_and_metrics(num_classes)
        model.compile(optimizer=optimizer, loss=loss_fn, metrics=[acc_metric, iou_metric])
        print("="*60 + "\n")
    
    train_loader = SegmentationDataset(DATASET_NAME, DATASET_DIR, 'trainval', INPUT_SHAPE[0], INPUT_SHAPE[1], is_training=True)
    val_loader = SegmentationDataset(DATASET_NAME, DATASET_DIR, 'valid', INPUT_SHAPE[0], INPUT_SHAPE[1], is_training=False)
    
    total_train_samples = train_loader.get_num_data()
    total_val_samples = val_loader.get_num_data()
    
    steps_per_epoch = total_train_samples // args.batch_size
    validation_steps = total_val_samples // args.batch_size
    if steps_per_epoch == 0:
        raise ValueError(f"steps_per_epoch이 0입니다.")
    if validation_steps == 0 and total_val_samples > 0:
        validation_steps = 1

    train_ds = train_loader.make_batch(args.batch_size)
    val_ds = val_loader.make_batch(args.batch_size)
    
    model.summary()
    
    lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=args.learning_rate,
        decay_steps=steps_per_epoch * args.epochs,
        end_learning_rate=1e-6,
        power=0.9
    )
    model.optimizer.learning_rate = lr_schedule
    
    log_dir = os.path.join(MODEL_DIR, "logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    
    callbacks = [
        keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_weights_only=False,
            monitor='val_mean_iou_with_ignore', 
            mode='max', 
            save_best_only=True, 
            verbose=1
        ),
        EpochMetricsLogger()
    ]
    
    print("\n" + "="*60)
    print("학습 진행 상황을 보려면, 새 터미널을 열고 아래 명령어를 실행하세요:")
    print(f"python3 -m tensorboard.main --logdir {os.path.abspath(log_dir)}")
    print("="*60 + "\n")

    print(f"학습 시작! (Steps per epoch: {steps_per_epoch}, Validation steps: {validation_steps})")
    model.fit(
        train_ds,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        validation_steps=validation_steps,
        callbacks=callbacks
    )
    print(f"--- 학습 완료 ---")

def evaluate(args):
    print(f"\n--- 평가 모드 시작 ---")
    
    checkpoint_path = os.path.join(MODEL_DIR, "best_model.keras")
    if not os.path.exists(checkpoint_path):
        print(f"오류: '{checkpoint_path}'에서 학습된 모델 파일을 찾을 수 없습니다.")
        return
    
    print(f"학습된 모델 로드 시도: {checkpoint_path}")
    try:
        model = keras.models.load_model(
            checkpoint_path, 
            custom_objects={'MeanIoUWithIgnore': MeanIoUWithIgnore, 'DeeplabV3PlusWithCBAM': DeeplabV3PlusWithCBAM, 'CBAM': CBAM},
            compile=False 
        )
        print("학습된 모델 로드 완료. 평가를 위해 다시 컴파일합니다.")
    except Exception as e:
        print(f"\n오류: 모델을 로드하는 중 문제가 발생했습니다: {e}")
        return

    val_loader = SegmentationDataset(DATASET_NAME, DATASET_DIR, 'valid', INPUT_SHAPE[0], INPUT_SHAPE[1], is_training=False)
    
    num_classes = model.layers[-1].filters
    total_val_samples = val_loader.get_num_data()
    if total_val_samples == 0:
        print(f"오류: 검증 데이터셋에 샘플이 없습니다.")
        return
    eval_steps = math.ceil(total_val_samples / 1)
    
    val_ds = val_loader.make_batch(1)

    loss_fn, acc_metric, iou_metric = get_loss_and_metrics(num_classes)
    model.compile(loss=loss_fn, metrics=[acc_metric, iou_metric])

    print(f"모델 평가 중... (Total steps: {eval_steps})")
    results = model.evaluate(val_ds, steps=eval_steps)
    print(f"평가 결과: {dict(zip(model.metrics_names, results))}")
    print(f"--- 평가 완료 ---")

def inference(args):
    process = psutil.Process(os.getpid())
    initial_ram_mb = process.memory_info().rss / (1024 * 1024)
    print(f"\n--- 추론 모드 시작 (초기 RAM: {initial_ram_mb:.2f} MB) ---")
    if not args.input_path or not args.output_dir:
        print("오류: --input_path와 --output_dir 인자를 모두 지정해야 합니다.")
        return

    checkpoint_path = os.path.join(MODEL_DIR, "best_model.keras")
    if not os.path.exists(checkpoint_path):
        print(f"오류: '{MODEL_DIR}'에서 학습된 모델 파일(*.keras)을 찾을 수 없습니다.")
        return
    
    print(f"학습된 최고 성능 모델 로드 시도: {checkpoint_path}")
    try:
        model = keras.models.load_model(
            checkpoint_path, 
            custom_objects={'MeanIoUWithIgnore': MeanIoUWithIgnore, 'DeeplabV3PlusWithCBAM': DeeplabV3PlusWithCBAM, 'CBAM': CBAM},
            compile=False
        )
        print("학습된 모델 로드 완료 (추론 모드).")
    except Exception as e:
        print(f"\n오류: 모델을 로드하는 중 문제가 발생했습니다: {e}")
        return

    # --- [수정] roboflow_dir 없이 범례 생성 ---
    legend_image = create_legend_image()
    
    if legend_image is None:
        print("경고: 범례를 생성하지 못했지만, 추론은 계속 진행합니다.")

    input_path = args.input_path.strip()
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

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"총 {len(image_paths)}개의 이미지를 처리합니다.")

    for image_path in tqdm(image_paths, desc="추론 진행률"):
        try:
            image_raw = tf.io.read_file(image_path)
            original_image_tensor = tf.image.decode_image(image_raw, channels=3)
            original_image_np = original_image_tensor.numpy()
            
            image_float = tf.cast(original_image_tensor, tf.float32)
            resized_image = tf.image.resize(image_float, (INPUT_SHAPE[0], INPUT_SHAPE[1]))
            image_batch = tf.expand_dims(resized_image, 0)
            
            predictions = model.predict(image_batch, verbose=0)
            
            current_ram_mb = process.memory_info().rss / (1024 * 1024)
            tqdm.write(f"  - '{os.path.basename(image_path)}' 추론 후 RAM: {current_ram_mb:.2f} MB")
            
            original_height, original_width, _ = original_image_np.shape
            seg_map = tf.argmax(predictions[0], axis=-1)
            seg_map = tf.cast(seg_map, tf.uint8)
            seg_map_resized = tf.image.resize(tf.expand_dims(seg_map, -1), (original_height, original_width), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            seg_map_resized = tf.squeeze(seg_map_resized, axis=-1)
            
            color_mask_image = label_to_color_image(seg_map_resized.numpy())
            overlayed_image = (original_image_np * 0.6 + color_mask_image * 0.4).astype(np.uint8)
            
            base_filename = os.path.basename(image_path)
            filename_no_ext, _ = os.path.splitext(base_filename)

            resized_legend = None
            if legend_image is not None:
                resized_legend = tf.image.resize(legend_image, (original_height, legend_image.shape[1])).numpy().astype(np.uint8)

            if args.save_mode in ['comparison', 'all']:
                comparison_image = np.hstack((original_image_np, color_mask_image, overlayed_image))
                if resized_legend is not None:
                    final_comparison_image = np.hstack((comparison_image, resized_legend))
                else:
                    final_comparison_image = comparison_image
                
                comparison_path = os.path.join(args.output_dir, f"{filename_no_ext}_comparison.png")
                tf.keras.utils.save_img(comparison_path, final_comparison_image)

            if args.save_mode in ['overlay', 'all']:
                final_overlay_image = overlayed_image
                
                output_path = os.path.join(args.output_dir, f"{filename_no_ext}_result.png")
                tf.keras.utils.save_img(output_path, final_overlay_image)

        except Exception as e:
            print(f"오류: '{image_path}' 처리 중 문제가 발생했습니다: {e}")

    final_ram_mb = process.memory_info().rss / (1024 * 1024)
    print(f"\n[정보] 모든 추론 완료 후 RAM 사용량: {final_ram_mb:.2f} MB")
    print("\n--- 모든 추론 완료 ---")

def export_savedmodel(args):
    print(f"\n--- SavedModel 내보내기 모드 시작 ---")
    
    checkpoint_path = os.path.join(MODEL_DIR, "best_model.keras")
    if not os.path.exists(checkpoint_path):
        print(f"오류: '{MODEL_DIR}'에서 학습된 모델 파일을 찾을 수 없습니다.")
        return None
        
    print(f"학습된 최고 성능 모델 로드: {checkpoint_path}")
    model = keras.models.load_model(checkpoint_path, custom_objects={'MeanIoUWithIgnore': MeanIoUWithIgnore, 'DeeplabV3PlusWithCBAM': DeeplabV3PlusWithCBAM, 'CBAM': CBAM})

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, *INPUT_SHAPE), dtype=tf.float32, name='input_image')
    ])
    def serving_fn(input_image):
        logits = model(input_image, training=False)
        predictions = tf.expand_dims(
            tf.argmax(logits, axis=3, output_type=tf.int32), axis=3, name='segmentation_mask')
        return {'output_mask': predictions}

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    export_path = os.path.join(EXPORT_SAVEDMODEL_DIR, timestamp)
    os.makedirs(export_path, exist_ok=True)
    tf.saved_model.save(model, export_path, signatures={'serving_default': serving_fn})

    print(f"SavedModel이 다음 경로에 내보내졌습니다: {export_path}")
    return export_path

def export_tflite_model(saved_model_path, quantize=True):
    print(f"\n--- TensorFlow Lite 모델 변환 모드 시작 ---")

    if not saved_model_path or not os.path.exists(saved_model_path):
        print(f"오류: 유효한 SavedModel 경로가 없습니다: {saved_model_path}")
        return

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
    
    if quantize:
        print("[정보] INT8 양자화를 진행합니다.")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        def representative_data_gen():
            dataset_loader = SegmentationDataset(DATASET_NAME, DATASET_DIR, 'valid', INPUT_SHAPE[0], INPUT_SHAPE[1], is_training=False)
            for images, _, _ in dataset_loader.make_batch(1).take(100):
                yield [images]
        converter.representative_dataset = representative_data_gen
    else:
        print("[정보] 양자화를 진행하지 않고 Float32 TFLite 모델을 생성합니다.")
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]

    try:
        tflite_model = converter.convert()
        
        if quantize:
            tflite_path = TFLITE_MODEL_PATH.replace('.tflite', '_quant.tflite')
        else:
            tflite_path = TFLITE_MODEL_PATH.replace('_quant.tflite', '_float32.tflite')
            
        os.makedirs(os.path.dirname(tflite_path), exist_ok=True)
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)
        print(f"TFLite 모델이 성공적으로 저장되었습니다: {tflite_path}")
    except Exception as e:
        print(f"TFLite 모델 변환 중 오류 발생: {e}")

def main():
    parser = argparse.ArgumentParser(description="DeepLabV3+ CBAM 모델 학습 및 배포 스크립트")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'eval', 'inference', 'export', 'export_tflite'])
    parser.add_argument('--input_path', type=str, help="추론에 사용할 입력 이미지 또는 폴더 경로")
    parser.add_argument('--output_dir', type=str, help="추론 결과를 저장할 폴더 경로")
    # --- [수정] roboflow_dir를 더 이상 사용하지 않으므로 제거 ---
    # parser.add_argument('--roboflow_dir', type=str, help="클래스 정보 및 범례 생성을 위한 원본 Roboflow 데이터셋 폴더 경로")
    parser.add_argument(
        '--save_mode', 
        type=str, 
        default='overlay', 
        choices=['overlay', 'comparison', 'all'], 
        help="추론 결과 저장 방식"
    )
    
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=4e-5)
    parser.add_argument('--mixed_precision', action='store_true', help="혼합 정밀도 학습 활성화")
    parser.add_argument('--resume_training', action='store_true', help="저장된 best_model.keras에서 학습을 이어합니다.")
    parser.add_argument('--no_quant', action='store_true', help="TFLite 변환 시 양자화를 비활성화합니다.")
    
    args = parser.parse_args()

    setup_environment(args)

    if args.mode == 'train':
        train(args)
    elif args.mode == 'eval':
        evaluate(args)
    elif args.mode == 'inference':
        inference(args)
    elif args.mode == 'export':
        exported_path = export_savedmodel(args)
        if exported_path:
            export_tflite_model(exported_path, quantize=not args.no_quant)
    elif args.mode == 'export_tflite':
        try:
            export_dirs = [d for d in os.listdir(EXPORT_SAVEDMODEL_DIR) if os.path.isdir(os.path.join(EXPORT_SAVEDMODEL_DIR, d))]
            if not export_dirs:
                raise FileNotFoundError
            latest_export = sorted(export_dirs)[-1]
            saved_model_to_convert = os.path.join(EXPORT_SAVEDMODEL_DIR, latest_export)
            print(f"가장 최근 SavedModel을 변환합니다: {saved_model_to_convert}")
            export_tflite_model(saved_model_to_convert, quantize=not args.no_quant)
        except (FileNotFoundError, IndexError):
            print(f"오류: '{EXPORT_SAVEDMODEL_DIR}'에서 SavedModel을 찾을 수 없습니다.")

if __name__ == '__main__':
    main()


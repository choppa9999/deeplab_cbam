#!/usr/bin/env python3
"""
evaluate_tflite_coco.py
-----------------------
TFLite 모델(float32)을 사용하여 COCO JSON + 이미지로 세그멘테이션 성능을 평가합니다.
배경 포함 Pixel Accuracy, Mean IoU, Mean Dice 계산.
라즈베리파이4 등 ARM CPU 환경에서도 동작할 수 있도록 설계되었습니다.
"""

import os
import json
import argparse
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw
from tqdm import tqdm

# -----------------------------
# 보조 함수: GT 마스크 생성
# -----------------------------

def create_mask_from_polygons(polygons, height, width):
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for polygon in polygons:
        if len(polygon) >= 6:
            coords = [(polygon[i], polygon[i+1]) for i in range(0, len(polygon), 2)]
            draw.polygon(coords, fill=1)
    return np.array(mask)


def load_coco(json_path, image_dir):
    with open(json_path, 'r') as f:
        coco = json.load(f)

    target_classes = ['dry', 'humid', 'slush', 'snow', 'wet']
    categories = {c['id']: c for c in coco['categories'] if c['name'] in target_classes}
    cat_map = {cid: idx+1 for idx, cid in enumerate(sorted(categories))}

    images = {img['id']: img for img in coco['images']}
    anns_by_img = {}
    for ann in coco['annotations']:
        anns_by_img.setdefault(ann['image_id'], []).append(ann)

    data = []
    for img_id, info in images.items():
        img_path = os.path.join(image_dir, info['file_name'])
        if os.path.exists(img_path):
            data.append({
                'image_path': img_path,
                'info': info,
                'anns': anns_by_img.get(img_id, [])
            })
    print(f"총 {len(data)}개 이미지 로드 완료")
    return data, cat_map


def gt_mask_from_anns(anns, cat_map, h, w):
    gt = np.zeros((h, w), dtype=np.uint8)
    for ann in sorted(anns, key=lambda x: x.get('area', 0), reverse=True):
        cid = ann['category_id']
        if cid not in cat_map:
            continue
        nid = cat_map[cid]
        for poly in ann['segmentation'] if isinstance(ann['segmentation'], list) else []:
            if len(poly) >= 6:
                mask = create_mask_from_polygons([poly], h, w)
                gt[mask == 1] = nid
    return gt


def preprocess_image(img_path, in_h, in_w):
    img = Image.open(img_path).convert('RGB')
    original_size = img.size
    img_tensor = tf.convert_to_tensor(img, dtype=tf.float32)
    img_tensor = tf.image.resize(img_tensor, (in_h, in_w), method='bilinear')
    input_data = tf.expand_dims(img_tensor, 0)  # (1, H, W, 3)
    return input_data, original_size


def run_inference(interpreter, input_data, original_size):
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    # 모델 타입 (float32)
    if input_details['dtype'] == np.float32:
        input_data = tf.cast(input_data, tf.float32)
    else:
        input_data = tf.cast(input_data, input_details['dtype'])
    interpreter.set_tensor(input_details['index'], input_data)
    interpreter.invoke()
    pred = interpreter.get_tensor(output_details['index'])  # (1, H, W, C) 또는 (1, H, W)
    pred = np.squeeze(pred)  # -> (H, W, C) or (H, W)

    # 로그it 맵인 경우 C 차원을 argmax로 클래스 맵(2D)으로 변환
    if pred.ndim == 3:
        pred = np.argmax(pred, axis=-1)

    pred = pred.astype(np.uint8)  # PIL 호환

    pred_img = Image.fromarray(pred)
    pred_resized = pred_img.resize(original_size, Image.NEAREST)
    return np.array(pred_resized)


def iou_score(pred, gt, cls):
    inter = np.sum((pred == cls) & (gt == cls))
    union = np.sum((pred == cls) | (gt == cls))
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return inter / union


def dice_score(pred, gt, cls):
    inter = np.sum((pred == cls) & (gt == cls))
    total = np.sum(pred == cls) + np.sum(gt == cls)
    if total == 0:
        return 1.0 if inter == 0 else 0.0
    return 2 * inter / total


def main():
    ap = argparse.ArgumentParser(description="TFLite 모델 COCO 평가 (배경 포함)")
    ap.add_argument('--tflite_model', type=str, default='exported_models/tflite_models/Model_quant.tflite')
    ap.add_argument('--json_path', type=str, default='data/COCO/test_without_street.json')
    ap.add_argument('--image_dir', type=str, default='data/images')
    ap.add_argument('--input_height', type=int, default=512)
    ap.add_argument('--input_width', type=int, default=512)
    args = ap.parse_args()

    if not os.path.exists(args.tflite_model):
        print(f"TFLite 모델을 찾을 수 없습니다: {args.tflite_model}")
        return

    # Interpreter 준비
    print(f"TFLite 모델 로드: {args.tflite_model}")
    interpreter = tf.lite.Interpreter(model_path=args.tflite_model, num_threads=4)
    interpreter.allocate_tensors()

    # 데이터 로드
    data_list, cat_map = load_coco(args.json_path, args.image_dir)
    num_classes = len(cat_map) + 1  # background 포함
    metrics = {
        'acc': [], 'miou': [], 'mdice': [],
        'ious': [[] for _ in range(num_classes)],
        'dices': [[] for _ in range(num_classes)]
    }

    for idx, item in enumerate(tqdm(data_list, desc='Evaluate')):
        img_path = item['image_path']
        h, w = item['info']['height'], item['info']['width']
        gt = gt_mask_from_anns(item['anns'], cat_map, h, w)

        inp, orig_size = preprocess_image(img_path, args.input_height, args.input_width)
        pred = run_inference(interpreter, inp, orig_size)

        # metrics
        mask = (gt < num_classes) & (pred < num_classes)
        if not np.any(mask):
            continue
        metrics['acc'].append(np.mean(pred[mask] == gt[mask]))
        ious = []
        dices = []
        for cls in range(num_classes):
            iou = iou_score(pred, gt, cls)
            dice = dice_score(pred, gt, cls)
            metrics['ious'][cls].append(iou)
            metrics['dices'][cls].append(dice)
            ious.append(iou)
            dices.append(dice)
        metrics['miou'].append(np.mean(ious))
        metrics['mdice'].append(np.mean(dices))

        if (idx+1) % 100 == 0:
            print(f"\n[Progress {idx+1}/{len(data_list)}] PixelAcc: {np.mean(metrics['acc']):.4f}, MeanIoU: {np.mean(metrics['miou']):.4f}, MeanDice: {np.mean(metrics['mdice']):.4f}")

    # 최종 결과
    print("\n" + "="*60)
    print("TFLite 모델 최종 평가 결과 (배경 포함)")
    print("="*60)
    print(f"Pixel Accuracy: {np.mean(metrics['acc']):.4f}")
    print(f"Mean IoU: {np.mean(metrics['miou']):.4f}")
    print(f"Mean Dice: {np.mean(metrics['mdice']):.4f}")

    class_names = ['background', 'dry', 'humid', 'slush', 'snow', 'wet']
    print("\n클래스별 IoU / Dice:")
    for cls in range(num_classes):
        cls_iou = np.mean(metrics['ious'][cls]) if metrics['ious'][cls] else 0
        cls_dice = np.mean(metrics['dices'][cls]) if metrics['dices'][cls] else 0
        name = class_names[cls] if cls < len(class_names) else f"class_{cls}"
        print(f"  {name:10} | IoU: {cls_iou:.4f} | Dice: {cls_dice:.4f}")

    print("="*60)


if __name__ == '__main__':
    main() 
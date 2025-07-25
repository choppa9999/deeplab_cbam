# create_tfrecords_from_coco.py
# 사용법: 스크립트 상단의 DATA_SPLITS와 OUTPUT_DIR을 수정한 후 'python3 create_tfrecords_from_coco.py' 실행

import os
import json
import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask
import tensorflow as tf
import io
from tqdm import tqdm

# --- [수정 필요] 경로 설정 ---
# 변환할 데이터셋 정보를 여기에 직접 입력합니다.
# 필요한 만큼 항목을 추가하거나 수정하여 사용할 수 있습니다.
DATA_SPLITS = {

    'train1': {
        'json_path': './data/robo/train/_annotations.coco.json',
        'image_dir': './data/robo/train'
    },
    'valid1': {
        'json_path': './data/robo/valid/_annotations.coco.json',
        'image_dir': './data/robo/valid'
    },
    'test1': {
        'json_path': './data/robo/test/_annotations.coco.json',
        'image_dir': './data/robo/test'
    },
    'test': {
        'json_path': './data/COCO/test_without_street.json',
        'image_dir': './data/images'
    },
    'train': {
        'json_path': './data/COCO/train_without_street.json',
        'image_dir': './data/images'
    },
    'valid': {
        'json_path': './data/COCO/valid_without_street.json',
        'image_dir': './data/images'
    }
}

# 생성될 TFRecord 파일들을 저장할 최종 폴더 경로
OUTPUT_DIR = './data/tfrecords'
# --- 경로 설정 끝 ---


# --- 사용할 클래스 이름 목록 ---
ALLOWED_CLASSES = ['dry', 'humid', 'slush', 'snow', 'wet']


def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def create_tf_example(image_path, annotations, cat_id_map):
    try:
        with tf.io.gfile.GFile(image_path, 'rb') as fid:
            encoded_image = fid.read()

        pil_image = Image.open(io.BytesIO(encoded_image))
        width, height = pil_image.size
        image_format = pil_image.format.lower().encode('utf-8')


        final_mask = np.zeros((height, width), dtype=np.uint8)
        for ann in sorted(annotations, key=lambda x: x['area'], reverse=True):
            if ann['category_id'] not in cat_id_map:
                continue

            category_id = cat_id_map[ann['category_id']]

            if isinstance(ann['segmentation'], list):
                rles = coco_mask.frPyObjects(ann['segmentation'], height, width)
                rle = coco_mask.merge(rles)
            elif isinstance(ann['segmentation']['counts'], list):
                rle = coco_mask.frPyObjects([ann['segmentation']], height, width)
            else: # RLE 형식일 경우
                rle = ann['segmentation']
            
            binary_mask = coco_mask.decode(rle)
            if len(binary_mask.shape) == 3: # 다차원 배열일 경우 2D로 변환
                binary_mask = np.any(binary_mask, axis=2)

            final_mask[binary_mask == 1] = category_id

        mask_image = Image.fromarray(final_mask)
        with io.BytesIO() as output:
            mask_image.save(output, format="PNG")
            encoded_mask = output.getvalue()

        feature = {
            'image/encoded': _bytes_feature(encoded_image),
            'image/format': _bytes_feature(image_format),
            'image/height': _int64_feature(height),
            'image/width': _int64_feature(width),
            'label/encoded': _bytes_feature(encoded_mask),
            'label/format': _bytes_feature(b'png'),
        }
        return tf.train.Example(features=tf.train.Features(feature=feature))
    except Exception as e:
        print(f"오류 발생 (이미지: {image_path}): {e}")
        return None


def process_split(set_name, info, output_dir):
    json_path = info['json_path']
    image_dir = info['image_dir']

    if not os.path.exists(json_path):
        print(f"경고: '{json_path}'를 찾을 수 없어 '{set_name}' 스플릿을 건너뜁니다.")
        return

    print(f"\n--- '{set_name}' 스플릿 TFRecord 생성 시작 ---")

    with open(json_path, 'r') as f:
        coco_data = json.load(f)

    images = {img['id']: img for img in coco_data['images']}

    categories_original = {cat['id']: cat for cat in coco_data['categories']}
    categories = {k: v for k, v in categories_original.items() if v['name'] in ALLOWED_CLASSES}
    print(f"'{set_name}' 스플릿에서 사용할 클래스: {[cat['name'] for cat in categories.values()]}")

    sorted_cat_ids = sorted(categories.keys())
    cat_id_map = {cat_id: i + 1 for i, cat_id in enumerate(sorted_cat_ids)}

    print("카테고리 매핑 정보 (0은 배경):")
    for cat_id, new_id in cat_id_map.items():
        print(f"  - COCO ID {cat_id} ('{categories[cat_id]['name']}') -> New ID {new_id}")

    img_to_anns = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in img_to_anns:
            img_to_anns[img_id] = []
        img_to_anns[img_id].append(ann)

    num_shards = 10
    num_total_images = len(images)
    if num_total_images == 0:
        print(f"'{set_name}' 스플릿에 이미지가 없습니다.")
        return

    output_file_prefix = os.path.join(output_dir, f'images-{set_name}')

    with tqdm(total=num_total_images, desc=f"Processing {set_name}") as pbar:
        shard_id = 0
        num_images_in_shard = 0
        writer = tf.io.TFRecordWriter(f"{output_file_prefix}-{shard_id:05d}-of-{num_shards:05d}.tfrecord")

        for img_id, img_info in images.items():
            image_path = os.path.join(image_dir, img_info['file_name'])
            current_annotations = img_to_anns.get(img_id, [])

            tf_example = create_tf_example(image_path, current_annotations, cat_id_map)

            if tf_example:
                writer.write(tf_example.SerializeToString())
                num_images_in_shard += 1

                if num_images_in_shard >= (num_total_images // num_shards) and shard_id < num_shards - 1:
                    writer.close()
                    shard_id += 1
                    writer = tf.io.TFRecordWriter(f"{output_file_prefix}-{shard_id:05d}-of-{num_shards:05d}.tfrecord")
                    num_images_in_shard = 0
            pbar.update(1)

        if writer:
            writer.close()

    print(f"'{set_name}' 스플릿에 대한 TFRecord 생성 완료! 총 {shard_id}개의 파일이 생성되었습니다.")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for set_name, info in DATA_SPLITS.items():
        process_split(set_name, info, OUTPUT_DIR)

    print("\n모든 TFRecord 변환 작업이 완료되었습니다.")


if __name__ == '__main__':
    main()


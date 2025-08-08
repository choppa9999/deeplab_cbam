# segmentation_dataset.py

import os
import tensorflow as tf


class SegmentationDataset:
    """
    모든 종류의 데이터셋을 위한 TFRecord 로더 및 전처리 파이프라인.
    샘플 수와 클래스 수를 자동으로 계산하도록 수정되었습니다.
    """

    def __init__(self,
                 dataset_name,
                 dataset_dir,
                 subset,
                 image_height,
                 image_width,
                 min_scale_factor=0.5,
                 max_scale_factor=2.0,
                 is_training=True):
        """
        데이터셋 객체를 초기화합니다.
        """
        self.dataset_name = dataset_name
        self.dataset_dir = dataset_dir
        self.subset = subset
        self.image_height = image_height
        self.image_width = image_width
        self.min_scale_factor = min_scale_factor
        self.max_scale_factor = max_scale_factor
        self.is_training = is_training
        self.ignore_label = 255

        # TFRecord 파일 경로 설정
        self.data_files = self._get_data_files()
        
        print(f"'{self.subset}' 데이터셋의 샘플 수를 세는 중입니다. 잠시 기다려 주세요...")
        self.num_samples = self._count_records(self.data_files)
        print(f"완료. 총 {self.num_samples}개의 샘플을 찾았습니다.")
        
        print(f"'{self.subset}' 데이터셋의 클래스 수를 확인하는 중입니다...")
        self.num_classes = self._determine_num_classes(self.data_files)
        print(f"완료. 총 {self.num_classes}개의 클래스(배경 포함)를 찾았습니다.")

    def _determine_num_classes(self, file_list, num_samples_to_check=200):
        """주어진 TFRecord 파일의 일부를 스캔하여 클래스 수를 결정합니다."""
        if len(file_list) == 0: return 0
        
        dataset = tf.data.TFRecordDataset(file_list).take(num_samples_to_check)
        dataset = dataset.map(self._parse_record, num_parallel_calls=tf.data.AUTOTUNE)

        unique_labels = set([0])
        for _, label in dataset:
            unique_in_label = tf.unique(tf.reshape(label, [-1])).y
            for l in unique_in_label.numpy():
                if l != self.ignore_label:
                    unique_labels.add(l)

        if not unique_labels:
            print("경고: 데이터셋 샘플에서 유효한 레이블을 찾을 수 없습니다. 클래스 수를 1로 설정합니다.")
            return 1

        max_label = max(unique_labels)
        return max_label + 1

    def _count_records(self, file_list):
        """주어진 TFRecord 파일 목록의 총 레코드 수를 계산합니다."""
        count = 0
        for fn in file_list:
            for _ in tf.data.TFRecordDataset(fn):
                count += 1
        return count

    def _get_data_files(self):
        """TFRecord 파일 목록을 반환합니다."""
        # --- [수정] 파일 이름 패턴을 실제 생성된 파일에 맞게 변경 ---
        if self.subset == 'trainval':
            # 'images-train-*'과 'images-valid-*' 패턴을 모두 찾습니다.
            train_pattern = os.path.join(self.dataset_dir, f'images-train*-of-*.tfrecord')
            val_pattern = os.path.join(self.dataset_dir, f'images-valid*-of-*.tfrecord')
            data_files = tf.io.gfile.glob(train_pattern) + tf.io.gfile.glob(val_pattern)
        elif self.subset == 'valid':
             val_pattern = os.path.join(self.dataset_dir, f'images-valid*-of-*.tfrecord')
             data_files = tf.io.gfile.glob(val_pattern)
        else: # 'train' 등 다른 서브셋
            file_pattern = os.path.join(self.dataset_dir, f'images-{self.subset}*-of-*.tfrecord')
            data_files = tf.io.gfile.glob(file_pattern)
        # --- 수정 완료 ---

        if not data_files:
            raise FileNotFoundError(f"TFRecord 파일을 찾을 수 없습니다. 경로와 subset 이름을 확인하세요: {self.dataset_dir}, subset: {self.subset}")

        return sorted(data_files)

    def get_num_data(self):
        """데이터셋의 총 샘플 수를 반환합니다."""
        return self.num_samples

    def _parse_record(self, raw_record):
        """TFRecord에서 이미지와 레이블을 파싱합니다."""
        keys_to_features = {
            'image/encoded': tf.io.FixedLenFeature((), tf.string),
            'label/encoded': tf.io.FixedLenFeature((), tf.string),
        }
        parsed = tf.io.parse_single_example(raw_record, keys_to_features)
        image = tf.image.decode_image(parsed['image/encoded'], channels=3)
        image = tf.cast(image, tf.float32)
        image.set_shape([None, None, 3])
        label = tf.image.decode_image(parsed['label/encoded'], channels=1)
        label = tf.cast(label, tf.int32)
        label.set_shape([None, None, 1])
        return image, label

    def _preprocess_image_and_label(self, image, label):
        """
        이미지 및 레이블에 전처리/데이터 증강을 적용하고 sample_weight를 생성합니다.
        """
        if self.is_training:
            scale = tf.random.uniform([], self.min_scale_factor, self.max_scale_factor, dtype=tf.float32)
            image_shape = tf.shape(image)
            new_height = tf.cast(tf.cast(image_shape[0], tf.float32) * scale, tf.int32)
            new_width = tf.cast(tf.cast(image_shape[1], tf.float32) * scale, tf.int32)
            image = tf.image.resize(image, [new_height, new_width], method=tf.image.ResizeMethod.BILINEAR)
            label = tf.image.resize(label, [new_height, new_width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            label = tf.cast(label, tf.int32)

        combined = tf.concat([image, tf.cast(label, tf.float32)], axis=-1)
        image_shape = tf.shape(combined)
        padded_height = tf.maximum(self.image_height, image_shape[0])
        padded_width = tf.maximum(self.image_width, image_shape[1])
        combined_padded = tf.image.pad_to_bounding_box(combined, 0, 0, padded_height, padded_width)
        combined_cropped = tf.image.random_crop(combined_padded, size=[self.image_height, self.image_width, 4])
        image = combined_cropped[:, :, :3]
        label = tf.cast(combined_cropped[:, :, 3:], tf.int32)

        if self.is_training and tf.random.uniform(()) > 0.5:
            image = tf.image.flip_left_right(image)
            label = tf.image.flip_left_right(label)

        sample_weight = tf.cast(tf.not_equal(label, self.ignore_label), dtype=tf.float32)
        return image, label, sample_weight

    def make_batch(self, batch_size):
        """
        TensorFlow 데이터셋을 생성하고 배치 단위로 준비합니다.
        """
        dataset = tf.data.Dataset.from_tensor_slices(self.data_files)
        dataset = dataset.interleave(
            tf.data.TFRecordDataset,
            cycle_length=tf.data.AUTOTUNE,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        if self.is_training:
            buffer_size = self.get_num_data() if self.get_num_data() > 0 else 10000
            dataset = dataset.shuffle(buffer_size=buffer_size)

        dataset = dataset.map(self._parse_record, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(self._preprocess_image_and_label, num_parallel_calls=tf.data.AUTOTUNE)
        
        dataset = dataset.repeat()
        dataset = dataset.batch(batch_size, drop_remainder=self.is_training)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset


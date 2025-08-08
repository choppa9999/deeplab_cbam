import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2


class CBAM(layers.Layer):
    """Convolutional Block Attention Module (CBAM)"""

    def __init__(self, reduction_ratio=8, kernel_size=7, **kwargs):
        super(CBAM, self).__init__(**kwargs)
        self.reduction_ratio = reduction_ratio
        self.kernel_size = kernel_size

    def build(self, input_shape):
        channel = input_shape[-1]
        self.mlp_1 = layers.Dense(channel // self.reduction_ratio, activation='relu', name='cbam_mlp1')
        self.mlp_2 = layers.Dense(channel, name='cbam_mlp2')
        self.spatial_conv = layers.Conv2D(1, self.kernel_size, padding='same', activation='sigmoid',
                                          name='cbam_spatial_conv')
        super(CBAM, self).build(input_shape)

    def call(self, inputs):
        # Channel Attention
        input_channel = inputs.shape[-1]
        avg_pool = layers.GlobalAveragePooling2D()(inputs)
        max_pool = layers.GlobalMaxPooling2D()(inputs)

        # Shared MLP
        avg_mlp = self.mlp_2(self.mlp_1(avg_pool))
        max_mlp = self.mlp_2(self.mlp_1(max_pool))
        
        channel_attention = tf.nn.sigmoid(avg_mlp + max_mlp)
        channel_attention_reshaped = layers.Reshape((1, 1, input_channel))(channel_attention)
        channel_refined = inputs * channel_attention_reshaped

        # Spatial Attention
        avg_pool_s = tf.reduce_mean(channel_refined, axis=-1, keepdims=True)
        max_pool_s = tf.reduce_max(channel_refined, axis=-1, keepdims=True)
        spatial_concat = tf.concat([avg_pool_s, max_pool_s], axis=-1)
        spatial_attention = self.spatial_conv(spatial_concat)

        refined_feature = channel_refined * spatial_attention
        return refined_feature

    def get_config(self):
        config = super().get_config()
        config.update({
            'reduction_ratio': self.reduction_ratio,
            'kernel_size': self.kernel_size,
        })
        return config


class DeeplabV3PlusWithCBAM(keras.Model):
    """
    DeepLabV3+ 모델 아키텍처 (CBAM 및 Dropout 적용 버전).
    """

    # 1. __init__ 시그니처에 backbone_weights 추가
    def __init__(self, num_classes, input_shape=(512, 512, 3), output_stride=16, upsample_logits=True, backbone_weights='imagenet', **kwargs):
        super().__init__(name='DeeplabV3Plus_with_CBAM', **kwargs)
        self.num_classes = num_classes
        self.input_shape_ = input_shape
        self.output_stride = output_stride
        self.upsample_logits = upsample_logits
        self.backbone_weights = backbone_weights # get_config를 위해 저장

        # MobileNetV2 백본 모델 생성
        base_backbone = MobileNetV2(
            input_shape=self.input_shape_, 
            include_top=False, 
            weights=self.backbone_weights
        )

        # 특징 추출을 위한 중간 레이어 출력 정의
        high_level_feature_layer_name = 'block_16_project_BN'
        low_level_feature_layer_name = 'block_3_expand_relu'
        layer_outputs = [
            base_backbone.get_layer(high_level_feature_layer_name).output,
            base_backbone.get_layer(low_level_feature_layer_name).output,
        ]
        
        # 2. 특징 추출기 모델의 이름을 'backbone'으로 지정
        # 이 이름을 통해 main.py에서 이 레이어를 찾아 동결/해제할 수 있습니다.
        self.backbone = keras.Model(
            inputs=base_backbone.input, 
            outputs=layer_outputs,
            name='backbone'
        )

        # ASPP (Atrous Spatial Pyramid Pooling)
        atrous_rates = [6, 12, 18] if output_stride == 16 else [12, 24, 36]
        self.aspp_conv_1x1 = self._conv_bn_relu(256, 1, 'aspp_conv_1x1')
        self.aspp_conv_atrous_1 = self._conv_bn_relu(256, 3, 'aspp_conv_atrous_1', dilation_rate=atrous_rates[0])
        self.aspp_conv_atrous_2 = self._conv_bn_relu(256, 3, 'aspp_conv_atrous_2', dilation_rate=atrous_rates[1])
        self.aspp_conv_atrous_3 = self._conv_bn_relu(256, 3, 'aspp_conv_atrous_3', dilation_rate=atrous_rates[2])
        self.image_pooling_pool = layers.GlobalAveragePooling2D(keepdims=True, name='aspp_image_pooling_pool')
        self.image_pooling_conv = self._conv_bn_relu(256, 1, 'aspp_image_pooling_conv')
        self.aspp_concat_conv = self._conv_bn_relu(256, 1, 'aspp_concat_conv')
        self.aspp_cbam = CBAM(name='aspp_cbam')

        # Decoder
        self.decoder_low_level_conv = self._conv_bn_relu(48, 1, 'decoder_low_level_conv')
        self.decoder_conv1 = self._conv_bn_relu(256, 3, 'decoder_conv1')
        self.decoder_conv2 = self._conv_bn_relu(256, 3, 'decoder_conv2')
        self.decoder_dropout = layers.Dropout(0.5)
        self.decoder_cbam = CBAM(name='decoder_cbam')
        self.final_conv = layers.Conv2D(num_classes, 1, name='final_logits')

    def _conv_bn_relu(self, filters, kernel_size, name_prefix, strides=1, dilation_rate=1):
        return keras.Sequential([
            layers.Conv2D(filters, kernel_size, strides=strides, padding='same', dilation_rate=dilation_rate,
                          use_bias=False, name=f"{name_prefix}_conv"),
            layers.BatchNormalization(name=f"{name_prefix}_bn"),
            layers.ReLU(name=f"{name_prefix}_relu")
        ], name=name_prefix)

    def call(self, inputs, training=False):
        # 3. 모델 내부의 전처리 로직을 제거하여 이중 전처리를 방지합니다.
        # 전처리는 main.py의 데이터 파이프라인에서 한 번만 수행됩니다.
        encoder_output, low_level_features = self.backbone(inputs, training=training)

        # ASPP
        aspp_features = [
            self.aspp_conv_1x1(encoder_output, training=training),
            self.aspp_conv_atrous_1(encoder_output, training=training),
            self.aspp_conv_atrous_2(encoder_output, training=training),
            self.aspp_conv_atrous_3(encoder_output, training=training),
        ]
        image_features = self.image_pooling_pool(encoder_output)
        image_features = self.image_pooling_conv(image_features, training=training)
        image_features = tf.image.resize(image_features, tf.shape(encoder_output)[1:3], method='bilinear')
        aspp_features.append(image_features)

        aspp_output = layers.Concatenate(axis=-1)(aspp_features)
        aspp_output = self.aspp_concat_conv(aspp_output, training=training)
        aspp_output = self.aspp_cbam(aspp_output)

        # Decoder
        low_level_features = self.decoder_low_level_conv(low_level_features, training=training)
        aspp_output_upsampled = tf.image.resize(aspp_output, tf.shape(low_level_features)[1:3], method='bilinear')
        decoder_input = layers.Concatenate(axis=-1)([aspp_output_upsampled, low_level_features])

        decoder_output = self.decoder_conv1(decoder_input, training=training)
        decoder_output = self.decoder_conv2(decoder_output, training=training)
        decoder_output = self.decoder_dropout(decoder_output, training=training)
        decoder_output = self.decoder_cbam(decoder_output)

        if self.upsample_logits:
            decoder_output = tf.image.resize(decoder_output, self.input_shape_[0:2], method='bilinear')

        logits = self.final_conv(decoder_output)
        return logits

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_classes": self.num_classes,
            "input_shape": self.input_shape_,
            "output_stride": self.output_stride,
            "upsample_logits": self.upsample_logits,
            "backbone_weights": self.backbone_weights, # config에 추가
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


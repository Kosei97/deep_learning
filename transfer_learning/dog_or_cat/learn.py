import tensorflow as tf
from transfer_learning.dog_or_cat.constant import TRAIN_DIR, VALIDATION_DIR
from transfer_learning.dog_or_cat.config import (
    IMG_SIZE,
    BATCH_SIZE,
    IMG_SHAPE,
    LEARNING_RATE,
    EPOCHS,
)


def create_model():
    # 水増しレイヤー(回転させて学習画像を増やす)
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.2),
        ]
    )
    # tf.keras.applications.MobileNetV2の画像データは-1~1が推奨されているため、画像データをリスケール
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

    # 転移学習の基盤を用意
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SHAPE, include_top=False, weights="imagenet"
    )

    # 畳み込みベースを凍結
    base_model.trainable = False

    # 5x5 空間の空間位置を平均化し、特徴を画像ごとに単一の1280要素ベクトルに変換
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

    # 正の数はクラス 1 を予測し、負の数はクラス 0 を予測
    prediction_layer = tf.keras.layers.Dense(1)

    inputs = tf.keras.Input(shape=(160, 160, 3))
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)

    return model


def create_dataset():
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE
    )

    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        VALIDATION_DIR,
        shuffle=True,
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE,
    )
    return train_dataset, validation_dataset


def learning_start(model, train_dataset, validation_dataset):
    # モデルをコンパイル
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    # 学習開始
    model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=validation_dataset,
    )
    model.save("result/model.h5")

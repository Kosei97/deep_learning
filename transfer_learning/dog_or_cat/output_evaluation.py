import tensorflow as tf
import tkinter as tk
from tkinter import filedialog
from PIL import Image
import numpy as np
from transfer_learning.dog_or_cat.config import IMG_SIZE


def image_input():
    # ウィンドウのルート設定(サイズ0のタイトルバー非表示に設定 ※単なる非表示ではfiledialogが機能しなくなる)
    root = tk.Tk()
    root.geometry("0x0")
    root.overrideredirect(1)

    image_path = filedialog.askopenfilename()
    root.destroy()

    return image_path


def preprocess_image(image_path):
    axis3_image = np.array(Image.open(image_path).resize(IMG_SIZE))
    axis4_image = axis3_image[np.newaxis, :, :, :]
    return axis4_image


def load_model():
    model = tf.keras.models.load_model(
        "transfer_learning/dog_or_cat/result/model.h5"
    )
    return model


def evaluation_model(image, model):
    prediction = model.predict_on_batch(image).flatten()
    return prediction


def run():
    print("start")
    target_image = preprocess_image(image_input())
    model = load_model()
    output = evaluation_model(target_image, model)
    output = tf.nn.sigmoid(output)
    if output.numpy()[0] < 0.3:
        print("猫である")
    if output.numpy()[0] >= 0.7:
        print("犬である")
    if output.numpy()[0] >= 0.3 and output.numpy()[0] < 0.7:
        print("このAIモデルでは判定が難しい")


if __name__ == "__main__":
    run()

import urllib.request
import shutil
import os
from transfer_learning.dog_or_cat.constant import URL, DATA_DIR


def download_data():
    SAVE_NAME = f"{DATA_DIR}/data.zip"
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    urllib.request.urlretrieve(URL, SAVE_NAME)

    shutil.unpack_archive(
        f"{DATA_DIR}/data.zip",
        DATA_DIR,
    )

    shutil.move(
        f"{DATA_DIR}/cats_and_dogs_filtered/train",
        f"{DATA_DIR}/train",
    )
    shutil.move(
        f"{DATA_DIR}/cats_and_dogs_filtered/validation",
        f"{DATA_DIR}/validation",
    )

    shutil.rmtree(f"{DATA_DIR}/cats_and_dogs_filtered")
    os.remove(f"{DATA_DIR}/data.zip")

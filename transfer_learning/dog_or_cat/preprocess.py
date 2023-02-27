import urllib.request
import shutil
import os

URL = (
    "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"
)
SAVE_NAME = "transfer_learning/dataset/dog_or_cat/data.zip"
if not os.path.exists("transfer_learning/dataset/dog_or_cat"):
    os.mkdir("transfer_learning/dataset/dog_or_cat")
urllib.request.urlretrieve(URL, SAVE_NAME)

shutil.unpack_archive(
    "transfer_learning/dataset/dog_or_cat/data.zip",
    "transfer_learning/dataset/dog_or_cat",
)

shutil.move(
    "transfer_learning/dataset/dog_or_cat/cats_and_dogs_filtered/train",
    "transfer_learning/dataset/dog_or_cat/train",
)
shutil.move(
    "transfer_learning/dataset/dog_or_cat/cats_and_dogs_filtered/validation",
    "transfer_learning/dataset/dog_or_cat/validation",
)

shutil.rmtree("transfer_learning/dataset/dog_or_cat/cats_and_dogs_filtered")
os.remove("transfer_learning/dataset/dog_or_cat/data.zip")

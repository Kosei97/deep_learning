from transfer_learning.dog_or_cat.preprocess import download_data
from transfer_learning.dog_or_cat.learn import (
    create_dataset,
    create_model,
    learning_start,
)


def run():
    download_data()
    train_dataset, validation_dataset = create_dataset()
    model = create_model()
    learning_start(model, train_dataset, validation_dataset)

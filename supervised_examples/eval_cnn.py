import json
from glob import glob

import torch
from audio_encoder.encoder import AudioClassifier
from supervised_examples.prepare_data import get_id_from_path
from supervised_examples.cnn_genre_classification import AudioDataset
from sklearn.model_selection import train_test_split
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import pytorch_lightning as pl

if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_path", default="/media/ml/data_ml/fma_metadata/")
    parser.add_argument("--mp3_path", default="/media/ml/data_ml/fma_small/")

    args = parser.parse_args()

    metadata_path = Path(args.metadata_path)
    mp3_path = Path(args.mp3_path)

    batch_size = 32
    epochs = 64

    CLASS_MAPPING = json.load(open(metadata_path / "mapping.json"))
    id_to_genres = json.load(open(metadata_path / "tracks_genre.json"))
    id_to_genres = {int(k): v for k, v in id_to_genres.items()}

    files = sorted(list(glob(str(mp3_path / "*/*.npy"))))

    labels = [CLASS_MAPPING[id_to_genres[int(get_id_from_path(x))]] for x in files]
    print(len(labels))

    samples = list(zip(files, labels))

    _train, test = train_test_split(
        samples, test_size=0.2, random_state=1337, stratify=[a[1] for a in samples]
    )

    train, val = train_test_split(
        _train, test_size=0.1, random_state=1337, stratify=[a[1] for a in _train]
    )

    train_data = AudioDataset(train, augment=False)
    test_data = AudioDataset(test, augment=False)
    val_data = AudioDataset(val, augment=False)

    train_loader = DataLoader(
        train_data, batch_size=batch_size, num_workers=8, shuffle=True
    )
    val_loader = DataLoader(
        val_data, batch_size=batch_size, num_workers=8, shuffle=True
    )
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=False, num_workers=8
    )

    model_paths = [
        "../models/pretrained-epoch=21.ckpt",
        "../models/scratch-epoch=52.ckpt",
    ]

    models = [AudioClassifier() for a in model_paths]
    for model, path in zip(models, model_paths):
        model.load_state_dict(torch.load(path)["state_dict"])
        model.cuda()

    accuracies = []
    accuracy = pl.metrics.Accuracy()
    accuracy.cuda()

    for model in tqdm(models):
        correct = 0
        model.eval()

        for x, y in tqdm(test_loader):
            x = x.cuda()
            y = y.cuda()

            y_pred = model(x)

            accuracy(y_pred, y)

        accuracies.append(accuracy.compute().item())

    data = {"model_name": model_paths, "accuracies": accuracies}

    json.dump(data, open("test_accuracy.json", "w"), indent=4)

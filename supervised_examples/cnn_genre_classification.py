import json
from glob import glob

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from audio_encoder.audio_processing import random_crop, random_mask, random_multiply
from audio_encoder.encoder import AudioClassifier
from supervised_examples.prepare_data import get_id_from_path


class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, data, max_len=512, augment=True):
        self.data = data
        self.max_len = max_len
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        npy_path = self.data[idx][0]
        label = self.data[idx][1]

        x = np.load(npy_path)

        x = random_crop(x, crop_size=self.max_len)

        if self.augment:
            x = random_mask(x)
            x = random_multiply(x)

        x = torch.tensor(x, dtype=torch.float)
        label = torch.tensor(label, dtype=torch.long)

        return x, label


class DecayLearningRate(pl.Callback):
    def __init__(self):
        self.old_lrs = []

    def on_train_start(self, trainer, pl_module):
        # track the initial learning rates
        for opt_idx, optimizer in enumerate(trainer.optimizers):
            group = []
            for param_group in optimizer.param_groups:
                group.append(param_group["lr"])
            self.old_lrs.append(group)

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        for opt_idx, optimizer in enumerate(trainer.optimizers):
            old_lr_group = self.old_lrs[opt_idx]
            new_lr_group = []
            for p_idx, param_group in enumerate(optimizer.param_groups):
                old_lr = old_lr_group[p_idx]
                new_lr = old_lr * 0.97
                new_lr_group.append(new_lr)
                param_group["lr"] = new_lr
            self.old_lrs[opt_idx] = new_lr_group


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_path")
    parser.add_argument("--mp3_path")
    parser.add_argument("--encoder_path")

    args = parser.parse_args()

    metadata_path = Path(args.metadata_path)
    mp3_path = Path(args.mp3_path)

    batch_size = 64
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

    train_data = AudioDataset(train, augment=True)
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

    model = AudioClassifier()

    if args.encoder_path is not None:
        checkpoint_callback = ModelCheckpoint(
            monitor="valid_acc", mode="max", filepath="models/", prefix="pretrained"
        )
        logger = TensorBoardLogger(
            save_dir=".", name="lightning_logs", version="pretrained"
        )

        ckpt = torch.load(args.encoder_path)

        model.load_state_dict(ckpt["state_dict"], strict=False)

    else:

        checkpoint_callback = ModelCheckpoint(
            monitor="valid_acc", mode="max", filepath="models/", prefix="scratch"
        )

        logger = TensorBoardLogger(
            save_dir=".", name="lightning_logs", version="scratch"
        )

    trainer = pl.Trainer(
        max_epochs=epochs,
        gpus=1,
        logger=logger,
        checkpoint_callback=checkpoint_callback,
        callbacks=[DecayLearningRate()],
        gradient_clip_val=1.0,
    )
    trainer.fit(model, train_loader, val_loader)

    trainer.test(test_dataloaders=test_loader)

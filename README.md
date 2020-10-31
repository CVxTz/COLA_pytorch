# COLA
(Unofficial)-Re-Impementation of the model archiecture of CONTRASTIVE LEARNING OF GENERAL-PURPOSE AUDIO REPRESENTATIONS in PyTorch.

Paper: [CONTRASTIVE LEARNING OF GENERAL-PURPOSE AUDIO REPRESENTATIONS](https://arxiv.org/abs/2010.10915)

Paper's Official Code (In tensorflow): [tf-code](https://github.com/google-research/google-research/tree/master/cola)

### Install
```
python -m pip install .
```

### Run

```
# Data download: download fma data and metadata


wget -c https://os.unil.cloud.switch.ch/fma/fma_metadata.zip
wget -c https://os.unil.cloud.switch.ch/fma/fma_small.zip
wget -c https://os.unil.cloud.switch.ch/fma/fma_large.zip


# Data preparation : prepare json with fma_small labels and pre-compute mel-spectrograms and save them as .npy

python audio_encoder/prepare_data.py --metadata_path "/media/ml/data_ml/fma_metadata/"
python audio_encoder/audio_processing.py --mp3_path "/media/ml/data_ml/fma_large/"
python audio_encoder/audio_processing.py --mp3_path "/media/ml/data_ml/fma_small/"

# Training

# Train with COLA

python audio_encoder/train_encoder.py --mp3_path "/media/ml/data_ml/fma_large/"

# Train Supervised

python supervised_examples/cnn_genre_classification.py --metadata_path "/media/ml/data_ml/fma_metadata/" \
  --mp3_path "/media/ml/data_ml/fma_small/"

python supervised_examples/cnn_genre_classification.py --metadata_path "/media/ml/data_ml/fma_metadata/" \
  --mp3_path "/media/ml/data_ml/fma_small/" \
  --encoder_path "models/encoder.ckpt"

```

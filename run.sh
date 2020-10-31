# Data download:

#wget -c https://os.unil.cloud.switch.ch/fma/fma_metadata.zip
#wget -c https://os.unil.cloud.switch.ch/fma/fma_small.zip
#wget -c https://os.unil.cloud.switch.ch/fma/fma_large.zip
# Data preparation

# python audio_encoder/prepare_data.py --metadata_path "/media/ml/data_ml/fma_metadata/"
# python audio_encoder/audio_processing.py --mp3_path "/media/ml/data_ml/fma_large/"
# python audio_encoder/audio_processing.py --mp3_path "/media/ml/data_ml/fma_small/"

# Training

# python audio_encoder/train_encoder.py --mp3_path "/media/ml/data_ml/fma_large/"

# python supervised_examples/cnn_genre_classification.py --metadata_path "/media/ml/data_ml/fma_metadata/" \
#   --mp3_path "/media/ml/data_ml/fma_small/"

python supervised_examples/cnn_genre_classification.py --metadata_path "/media/ml/data_ml/fma_metadata/" \
  --mp3_path "/media/ml/data_ml/fma_small/" \
  --encoder_path "models/encoder.ckpt"

import torchaudio
from speechbrain.inference.interfaces import foreign_class
classifier = EncoderClassifier.from_hparams(source="Jzuluaga/accent-id-commonaccent_ecapa", savedir="pretrained_models/accent-id-commonaccent_ecapa")
# Irish Example
out_prob, score, index, text_lab = classifier.classify_file("data/audio_data/all/0D2XANLNWB.wav")

breakpoint()
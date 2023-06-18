import os

from trainer import Trainer, TrainerArgs

from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits_multilingual import Vits, VitsArgs, VitsAudioConfig
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.languages import LanguageManager
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor
from TTS.encoder.models.lstm import LSTMSpeakerEncoder

dataset_config = [
   BaseDatasetConfig(
    formatter="festcat3", meta_file_train="metadata_norm.txt", phonemizer="espeak", language="ca", path="/home/usuaris/veu/daniel.gonzalbez/festcat"
   ),
]   


audio_config = VitsAudioConfig(
    sample_rate=16000, win_length=1024, hop_length=256, num_mels=80, mel_fmin=0, mel_fmax=None
)

vitsArgs = VitsArgs(
    use_speaker_embedding=False,
    use_speaker_encoder_as_loss=True,
    use_d_vector_file=True,
    d_vector_dim=256,
    speaker_encoder_model_path="/home/usuaris/veu/daniel.gonzalbez/Multi-speaker-and-Multi-Lingual-TTS/320k.pth.tar",
    speaker_encoder_config_path="/home/usuaris/veu/daniel.gonzalbez/TTS/TTS/config_speaker_enc.json",
)
config = VitsConfig(
    model_args=vitsArgs,
    audio=audio_config,
    run_name="vits_festcat",
    batch_size=16,
    eval_batch_size=8,
    batch_group_size=5,
    num_loader_workers=1,
    num_eval_loader_workers=1,
    run_eval=True,
    test_delay_epochs=-1,
    epochs=1000,
    text_cleaner="esp_cat_cleaners",
    use_phonemes=True,
    phonemizer='espeak',
    phoneme_language="ca",
    phoneme_cache_path=os.path.join('/home/usuaris/veu/daniel.gonzalbez/', "phoneme_cache_cat"),
    compute_input_seq_cache=True,
    print_step=50,
    print_eval=False,
    mixed_precision=True,
    #max_text_len=325,  # change this if you have a larger VRAM than 16GB
    output_path='/home/usuaris/veu/daniel.gonzalbez/logs',
    datasets=dataset_config,
    cudnn_benchmark=False,
    test_sentences=[['Fa un bon dia per passejar.', 'f1_norm'],
                    ["Demà és el millor dia de l'any", 'm2_norm']
    ],
    use_speaker_weighted_sampler=True,
)

ap = AudioProcessor.init_from_config(config)

# INITIALIZE THE TOKENIZER
# Tokenizer is used to convert text to sequences of token IDs.
# config is updated with the default characters if not defined in the config.
tokenizer, config = TTSTokenizer.init_from_config(config)

# LOAD DATA SAMPLES
# Each sample is a list of ```[text, audio_file_path, speaker_name]```
# You can define your custom sample loader returning the list of samples.
# Or define your custom formatter and pass it to the `load_tts_samples`.
# Check `TTS.tts.datasets.load_tts_samples` for more details.
train_samples, eval_samples = load_tts_samples(    
    dataset_config,
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
)
print("TRAIN SAMPLES: ", train_samples[0])
speaker_manager = SpeakerManager(encoder_model_path=vitsArgs.speaker_encoder_model_path, 
                        encoder_config_path=vitsArgs.speaker_encoder_config_path, d_vectors_file_path="/home/usuaris/veu/daniel.gonzalbez/sp_cat_ref_multiple_new256.pth") #references3.pth
speaker_manager.set_ids_from_data(train_samples + eval_samples, parse_key="speaker_name")
config.model_args.num_speakers = speaker_manager.num_speakers



model = Vits(config, ap, tokenizer, speaker_manager = speaker_manager)

#loader = model.get_data_loader(config, False, train_samples, True, 1)

output_path = '/home/usuaris/veu/daniel.gonzalbez/logs'

train_args = TrainerArgs(grad_accum_steps = 1)
trainer = Trainer(
    train_args,
    config,
    output_path,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
)
print("ALL SET")
if __name__ == '__main__':
    trainer.fit()

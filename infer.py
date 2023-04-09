import os

from trainer import Trainer, TrainerArgs
import torch
import torchaudio

from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits, VitsArgs, VitsAudioConfig, VitsDataset
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.languages import LanguageManager
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor
from TTS.encoder.models.lstm import LSTMSpeakerEncoder

checkpoint_path = "/home/usuaris/veu/daniel.gonzalbez/logs/vits_tcstar-April-08-2023_10+58AM-0000000/best_model_4518.pth"
root_path = "/home/usuaris/veu/daniel.gonzalbez"
vitsArgs = VitsArgs(
    use_speaker_embedding=False,
    use_speaker_encoder_as_loss=False,
    speaker_encoder_model_path="/home/usuaris/veu/daniel.gonzalbez/Multi-speaker-and-Multi-Lingual-TTS/best_model.pth.tar",
    speaker_encoder_config_path="/home/usuaris/veu/daniel.gonzalbez/TTS/TTS/config_speaker_enc.json",
)

audio_config = VitsAudioConfig(
    sample_rate=16000, win_length=1024, hop_length=256, num_mels=80, mel_fmin=0, mel_fmax=None
)


config = VitsConfig(
    model_args=vitsArgs,
    audio=audio_config,
    run_name="vits_tcstar",
    batch_size=16,
    eval_batch_size=16,
    batch_group_size=5,
    num_loader_workers=4,
    num_eval_loader_workers=4,
    run_eval=True,
    test_delay_epochs=-1,
    epochs=10,
    text_cleaner="basic_cleaners",
    use_phonemes=False,
    phoneme_language="es",
    phoneme_cache_path=os.path.join('/home/usuaris/veu/daniel.gonzalbez/TTS', "phoneme_cache"),
    compute_input_seq_cache=True,
    print_step=25,
    print_eval=False,
    mixed_precision=True,
    max_text_len=325,  # change this if you have a larger VRAM than 16GB
    output_path='/home/usuaris/veu/daniel.gonzalbez/logs',
    cudnn_benchmark=False,
)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ap = AudioProcessor.init_from_config(config)

# INITIALIZE THE TOKENIZER
# Tokenizer is used to convert text to sequences of token IDs.
# config is updated with the default characters if not defined in the config.
tokenizer, config = TTSTokenizer.init_from_config(config)

language_manager = LanguageManager(config=config)
config.model_args.num_languages = language_manager.num_languages

model = Vits(config, ap, tokenizer)
model.load_checkpoint(config, checkpoint_path, strict=False)

text = "Hola me llamo daniel y voy a comer cacahuetes"

# generate speaker embeddings

# known speakers:
spk_emb = {'72_ref':'T6B72110153_lstm.pt', '73_ref':'T6B73120178_lstm.pt', '75_ref':'T6V75110122_lstm.pt', '76_ref':'T6V76110129_lstm.pt', '79_ref':'T6V79110211_lstm.pt', '80_ref':'T6V80110211_lstm.pt'}
#16x256x1
# new speakers:



#############################

spk_emb = torch.load(root_path + '/' + '72_ref' + '/' + spk_emb['72_ref'])
spk_emb = torch.unsqueeze(spk_emb, 2)
print(spk_emb.shape, "AAA")

sample = {"text": text, "spk_emb": spk_emb}

dataset = VitsDataset(
                model_args=config.model_args,
                samples=sample,
                min_text_len=config.min_text_len,
                max_text_len=config.max_text_len,
                min_audio_len=config.min_audio_len,
                max_audio_len=config.max_audio_len,
                phoneme_cache_path=config.phoneme_cache_path,
                precompute_num_workers=config.precompute_num_workers,
                tokenizer=tokenizer,
            )

tokens = torch.from_numpy(dataset.get_token_ids(None, sample["text"]))

tokens = torch.unsqueeze(tokens, 0) # add batch num


outputs = model.inference(tokens, sample["spk_emb"])

torchaudio.save("caca.wav", outputs["model_outputs"].squeeze().unsqueeze(0), 16000)
print("END")
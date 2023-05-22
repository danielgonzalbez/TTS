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

checkpoint_path = "/home/usuaris/veu/daniel.gonzalbez/logs/vits_tcstar-May-19-2023_11+18AM-0000000/checkpoint_40000.pth"
root_path = "/home/usuaris/veu/daniel.gonzalbez"
vitsArgs = VitsArgs(
    use_speaker_embedding=True,
    num_speakers=1
#    use_speaker_encoder_as_loss=False,
#    use_d_vector_file=False,
#    speaker_encoder_model_path="/home/usuaris/veu/daniel.gonzalbez/Multi-speaker-and-Multi-Lingual-TTS/best_model.pth.tar",
#    speaker_encoder_config_path="/home/usuaris/veu/daniel.gonzalbez/TTS/TTS/config_speaker_enc.json",
)

audio_config = VitsAudioConfig(
    sample_rate=16000, win_length=1024, hop_length=256, num_mels=80, mel_fmin=0, mel_fmax=None
)


config = VitsConfig(
    model_args=vitsArgs,
    audio=audio_config,
    run_name="vits_tcstar",
    batch_size=16,
    eval_batch_size=8,
    batch_group_size=5,
    num_loader_workers=4,
    num_eval_loader_workers=4,
    run_eval=True,
    test_delay_epochs=-1,
    epochs=10,
    text_cleaner="basic_cleaners",
    use_phonemes=True,
    phoneme_language="es",
    phonemizer="espeak",
    phoneme_cache_path=os.path.join('/home/usuaris/veu/daniel.gonzalbez/TTS', "phoneme_cache"),
    compute_input_seq_cache=True,
    print_step=25,
    print_eval=False,
    mixed_precision=True,
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
speaker_manager = SpeakerManager()

text = "Ahora me como un trozo de pan para disfrutar de la tarde y pasear por las calles de delante."
samples = []
samples.append({"text": text, "language": 0, "audio_unique_name": "panpaseo16x1emb", "speaker_name": '72_norm2'})
#samples.append({"text": text, "language": 0, "audio_unique_name": "pan", "speaker_name": '73_norm2'})
#samples.append({"text": text, "language": 0, "audio_unique_name": "pan", "speaker_name": '76_norm2'})
#samples.append({"text": text, "language": 0, "audio_unique_name": "pan", "speaker_name": '75_norm2'})
#samples.append({"text": text, "language": 0, "audio_unique_name": "pan", "speaker_name": '80_norm2'})
#samples.append({"text": text, "language": 0, "audio_unique_name": "pan", "speaker_name": '78_norm2'})
#speaker_manager.set_ids_from_data(samples, parse_key="speaker_name")
speaker_manager.set_ids_from_data(samples, parse_key="speaker_name")
model = Vits(config, ap, tokenizer, speaker_manager=speaker_manager)
model.load_checkpoint(config, checkpoint_path, strict=True)


# generate speaker embeddings

# known speakers:
spk_emb = {'72_ref':'T6B72110122_lstm.pt', '73_ref':'T6B73120178_lstm.pt', '75_ref':'T6V75110122_lstm.pt', '76_ref':'T6V76110129_lstm.pt', '79_ref':'T6V79110211_lstm.pt', '80_ref':'T6V80110211_lstm.pt'}
#16x256x1
# new speakers:



#############################

#spk_emb = torch.load(root_path + '/tcstar/72_ref/' + spk_emb['72_ref'])
#spk_emb = torch.unsqueeze(spk_emb, 2)
#print(spk_emb.shape, "AAA")


dataset = VitsDataset(
                model_args=config.model_args,
                samples=[samples[0]],
                min_text_len=config.min_text_len,
                max_text_len=config.max_text_len,
                min_audio_len=config.min_audio_len,
                max_audio_len=config.max_audio_len,
                phoneme_cache_path=config.phoneme_cache_path,
                precompute_num_workers=config.precompute_num_workers,
                tokenizer=tokenizer,
            )

tokens = torch.from_numpy(dataset.get_token_ids(0, samples[0]["text"]))


tokens = torch.unsqueeze(tokens, 0) # add batch num

print(tokens)

outputs = model.inference(tokens, aux_input={"speaker_ids":torch.Tensor([0]).to(torch.int32)})
print((outputs["model_outputs"]).shape)
torchaudio.save("pan16x1emb.wav", outputs["model_outputs"].squeeze().unsqueeze(0), 16000)
print("END")
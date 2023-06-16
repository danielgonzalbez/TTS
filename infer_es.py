import os

from trainer import Trainer, TrainerArgs
import torch
import torchaudio
import fsspec


from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits2 import Vits, VitsArgs, VitsAudioConfig, VitsDataset
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.languages import LanguageManager
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor
from TTS.encoder.models.lstm import LSTMSpeakerEncoder


checkpoint_path = "/home/usuaris/veu/daniel.gonzalbez/logs/vits_tcstar-June-03-2023_12+02PM-0000000/checkpoint_70000.pth"
root_path = "/home/usuaris/veu/daniel.gonzalbez"

vitsArgs = VitsArgs(
    use_speaker_embedding=False,
    use_speaker_encoder_as_loss=True,
    use_d_vector_file=True,
    speaker_encoder_model_path="/home/usuaris/veu/daniel.gonzalbez/Multi-speaker-and-Multi-Lingual-TTS/320k.pth.tar",
    speaker_encoder_config_path="/home/usuaris/veu/daniel.gonzalbez/TTS/TTS/config_speaker_enc.json",
    #speaker_encoder_model_path="/home/usuaris/veu/daniel.gonzalbez/best_model.pth.tar",
    #speaker_encoder_config_path="/home/usuaris/veu/daniel.gonzalbez/config.json",
    d_vector_dim=256,
)


audio_config = VitsAudioConfig(
    sample_rate=16000, win_length=1024, hop_length=256, num_mels=80, mel_fmin=0, mel_fmax=None
)


config = VitsConfig(
    model_args=vitsArgs,
    audio=audio_config,
    run_name="vits_tcstar",
    batch_size=16,
    eval_batch_size=4,
    batch_group_size=5,
    num_loader_workers=4,
    num_eval_loader_workers=1,
    run_eval=True,
    test_delay_epochs=-1,
    epochs=10,
    text_cleaner="esp_cleaners",
    use_phonemes=True,
    phoneme_language="es",
    phoneme_cache_path=os.path.join('/home/usuaris/veu/daniel.gonzalbez/', "phoneme_cache_cat_sp_def"),
    phonemizer='espeak',
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

speaker_manager = SpeakerManager(encoder_model_path=vitsArgs.speaker_encoder_model_path, 
                        encoder_config_path=vitsArgs.speaker_encoder_config_path, d_vectors_file_path="/home/usuaris/veu/daniel.gonzalbez/sp_cat_ref_uniqueCarlos_new256.pth")



model = Vits(config, ap, tokenizer, speaker_manager=speaker_manager)
model.load_checkpoint(config, checkpoint_path, strict=True)

text1 = "Estamos en el año 2023 y eso conlleva muchas cosas positivas."
text2 = "Qué bien nos lo hemos pasado en casa de Juan! ¿Tú pudiste venir o no?"
text3 = "El director de la NBA es un hombre multimillonario."

# generate speaker embeddings

# known speakers:
spk_emb_dic = {'72_ref':'T6B72120668_lstm.pt', '73_ref':'T6B73120178_lstm.pt', '75_ref':'T6V75110122_lstm.pt', '76_ref':'T6V76110129_lstm.pt', '79_ref':'T6V79110211_lstm.pt', '80_ref':'T6V80110211_lstm.pt',
'm4_ref': 'fcsm4203059_lstm.pt','f1_ref':'fcsf1200485_lstm.pt', 'm5_ref': 'fcsm5940003_lstm.pt', 'carlos_ref': 'zuretts_tts_0009_lstm.pt' }


#spk_emb = torch.load(root_path + '/festcat/m5_ref2/' + spk_emb_dic['m5_ref']).squeeze(-1)
spk_emb = torch.load(root_path + '/tcstar/75_ref2/' + spk_emb_dic['75_ref']).squeeze(-1)

#spk_emb = torch.load(root_path + '/recordingsCarlosdePablo/Carlos_ref2' + '/' + spk_emb_dic['carlos_ref']).squeeze(-1)
print(spk_emb.shape)
sample = {"text": text1, "language": 'es', "audio_unique_name": "numero"}

dataset = VitsDataset(
                model_args=config.model_args,
                samples=[sample],
                min_text_len=config.min_text_len,
                max_text_len=config.max_text_len,
                min_audio_len=config.min_audio_len,
                max_audio_len=config.max_audio_len,
                phoneme_cache_path=config.phoneme_cache_path,
                precompute_num_workers=config.precompute_num_workers,
                tokenizer=tokenizer,
            )

tokens = torch.from_numpy(dataset.get_token_ids(0, sample["text"]))
print(tokens)

tokens = torch.unsqueeze(tokens, 0) # add batch num

print("LEN", tokens)
outputs = model.inference(tokens, aux_input = {"d_vectors":spk_emb})
print((outputs["model_outputs"]).shape)
torchaudio.save("es_m5.wav", outputs["model_outputs"].squeeze().unsqueeze(0), 16000)
print("END")





sample = {"text": text2, "language": 'es', "audio_unique_name": "entonación"}

dataset = VitsDataset(
                model_args=config.model_args,
                samples=[sample],
                min_text_len=config.min_text_len,
                max_text_len=config.max_text_len,
                min_audio_len=config.min_audio_len,
                max_audio_len=config.max_audio_len,
                phoneme_cache_path=config.phoneme_cache_path,
                precompute_num_workers=config.precompute_num_workers,
                tokenizer=tokenizer,
            )

tokens = torch.from_numpy(dataset.get_token_ids(0, sample["text"]))
print(tokens)

tokens = torch.unsqueeze(tokens, 0) # add batch num

print("LEN", tokens)
outputs = model.inference(tokens, aux_input = {"d_vectors":spk_emb})
print((outputs["model_outputs"]).shape)
torchaudio.save("es_m5.wav", outputs["model_outputs"].squeeze().unsqueeze(0), 16000)
print("END")



sample = {"text": text3, "language": 'es', "audio_unique_name": "letras"}

dataset = VitsDataset(
                model_args=config.model_args,
                samples=[sample],
                min_text_len=config.min_text_len,
                max_text_len=config.max_text_len,
                min_audio_len=config.min_audio_len,
                max_audio_len=config.max_audio_len,
                phoneme_cache_path=config.phoneme_cache_path,
                precompute_num_workers=config.precompute_num_workers,
                tokenizer=tokenizer,
            )

tokens = torch.from_numpy(dataset.get_token_ids(0, sample["text"]))
print(tokens)

tokens = torch.unsqueeze(tokens, 0) # add batch num

print("LEN", tokens)
outputs = model.inference(tokens, aux_input = {"d_vectors":spk_emb})
print((outputs["model_outputs"]).shape)
torchaudio.save("es_m5.wav", outputs["model_outputs"].squeeze().unsqueeze(0), 16000)
print("END")
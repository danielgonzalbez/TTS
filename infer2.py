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



def load_file(path: str):
    if path.endswith(".json"):
        with fsspec.open(path, "r") as f:
            return json.load(f)
    elif path.endswith(".pth"):
        with fsspec.open(path, "rb") as f:
            return torch.load(f, map_location="cpu")
    else:
        raise ValueError("Unsupported file type")

output_path = os.path.dirname(os.path.abspath(__file__))
print(os.path.join(output_path, "phoneme_cache"))

checkpoint_path = "/home/usuaris/veu/daniel.gonzalbez/logs/vits_tcstar-May-21-2023_05+19PM-0000000/best_model_25795.pth"
root_path = "/home/usuaris/veu/daniel.gonzalbez"

vitsArgs = VitsArgs(
    use_speaker_embedding=False,
    use_speaker_encoder_as_loss=False,
    use_d_vector_file=True,
    speaker_encoder_model_path="/home/usuaris/veu/daniel.gonzalbez/Multi-speaker-and-Multi-Lingual-TTS/320k.pth.tar",
    speaker_encoder_config_path="/home/usuaris/veu/daniel.gonzalbez/TTS/TTS/config_speaker_enc.json",
    d_vector_dim=256
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
    text_cleaner="basic_cleaners",
    use_phonemes=True,
    phoneme_language="es",
    phoneme_cache_path=os.path.join('/home/usuaris/veu/daniel.gonzalbez/TTS', "phoneme_cache"),
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
                        encoder_config_path=vitsArgs.speaker_encoder_config_path, d_vectors_file_path="/home/usuaris/veu/daniel.gonzalbez/tcstar/references2.pth")

language_manager = LanguageManager(config=config)
config.model_args.num_languages = language_manager.num_languages

model = Vits(config, ap, tokenizer, speaker_manager=speaker_manager)
model.load_checkpoint(config, checkpoint_path, strict=True)

text = "Ahora me como un trozo de pan para disfrutar de la tarde y pasear por las calles de delante."

# generate speaker embeddings

# known speakers:
spk_emb_dic = {'72_ref':'T6B72120668_lstm.pt', '73_ref':'T6B73120178_lstm.pt', '75_ref':'T6V75110122_lstm.pt', '76_ref':'T6V76110129_lstm.pt', '79_ref':'T6V79110211_lstm.pt', '80_ref':'T6V80110211_lstm.pt'}




#############################

embeddings = load_file("/home/usuaris/veu/daniel.gonzalbez/tcstar/references2.pth")
speakers = sorted({x["name"] for x in embeddings.values()})
name_to_id = {name: i for i, name in enumerate(speakers)}
clip_ids = list(set(sorted(clip_name for clip_name in embeddings.keys())))
# cache embeddings_by_names for fast inference using a bigger speakers.json
embeddings_by_names = {}
for x in embeddings.values():
    if x["name"] not in embeddings_by_names.keys():
        embeddings_by_names[x["name"]] = [x["embedding"]]
    else:
        embeddings_by_names[x["name"]].append(x["embedding"])
# Shape : 1x256
print("Original embeddings = ", [(x["embedding"].shape, x["name"]) for x in embeddings.values()])
print("EMBEDDINGS, ", embeddings)
print("EMB ", embeddings["72_norm"]["embedding"])


spk_emb = torch.load(root_path + '/tcstar/72_ref2' + '/' + spk_emb_dic['72_ref']).squeeze(-1)
#spk_emb = spk_emb/abs(spk_emb)
print(spk_emb.shape, "AAA")

sample = {"text": text, "language": 0, "audio_unique_name": "16x1pand_v"}
#sample = {"text": text,"audio_unique_name": "new_paseoss1"}

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
torchaudio.save("corrected2spk72loss.wav", outputs["model_outputs"].squeeze().unsqueeze(0), 16000)
print("END")
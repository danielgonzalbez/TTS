import os

from trainer import Trainer, TrainerArgs
import torch
import torchaudio
import fsspec


from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits_multilingual import Vits, VitsArgs, VitsAudioConfig, VitsDataset
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.languages import LanguageManager
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor
from TTS.encoder.models.lstm import LSTMSpeakerEncoder


checkpoint_path = "/home/usuaris/veu/daniel.gonzalbez/logs/vits_tcstar-June-16-2023_05+35PM-0000000/checkpoint_110000.pth"
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
    use_language_embedding=True,
    language_ids_file="/home/usuaris/veu/daniel.gonzalbez/TTS/TTS/languages.json"
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
    text_cleaner="esp_cat_cleaners",
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
    use_speaker_weighted_sampler=True,
    use_language_weighted_sampler=True,
    speaker_encoder_loss_alpha = 6,
)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ap = AudioProcessor.init_from_config(config)

# INITIALIZE THE TOKENIZER
# Tokenizer is used to convert text to sequences of token IDs.
# config is updated with the default characters if not defined in the config.
tokenizer, config = TTSTokenizer.init_from_config(config)

speaker_manager = SpeakerManager(encoder_model_path=vitsArgs.speaker_encoder_model_path, 
                        encoder_config_path=vitsArgs.speaker_encoder_config_path, d_vectors_file_path="/home/usuaris/veu/daniel.gonzalbez/sp_cat_ref_multiple_new256.pth")

language_manager = LanguageManager(config=config)
config.model_args.num_languages = language_manager.num_languages

model = Vits(config, ap, tokenizer, speaker_manager=speaker_manager, language_manager=language_manager)
model.load_checkpoint(config, checkpoint_path, strict=True)

text_es = [
    "Voy a bailar con los goles del mejor jugador de la historia del fútbol, el astro argentino y sus compañeros. Se merecen todo lo bueno que les pase.",
    "Aquí encontrarás la gran riqueza del mundo hispanohablante y más de 100 textos para aprender español. Lee, escucha, imprime y comparte los textos con tus compañeros y alumnos en las redes sociales.",
    "Según la mecánica cuántica, cabe la posibilidad de que el universo se extinta en un instante. ¿Qué habría después? La pregunta carece de sentido. No hace falta intentar contestarla."
]

text_cat = [
    "El dos de juliol de 2010 es va fer pública la web, des d'on es començava a explicar la proposta de creació d'un nou mitjà de comunicació en català.",
    "Ho ha tornat a fer. Gairebé quinze anys després d'omplir d'orgull i de felicitat els seguidors del Barça en una història d'amor que va començar amb un triplet, Pep Guardiola ha brindat al seu equip aquesta mateixa glòria.",
    "La final va començar de manera una mica estranya, amb el conjunt anglès volent fer veure que deixava jugar un Inter que no es va deixar intimidar i que va viure el seu primer petit triomf."
]


# generate speaker embeddings

# known speakers:
spk_emb_dic = {'72_ref':'T6B72120668_lstm.pt', '73_ref':'T6B73120178_lstm.pt', '75_ref':'T6V75110122_lstm.pt', '76_ref':'T6V76110129_lstm.pt', '79_ref':'T6V79110211_lstm.pt', '80_ref':'T6V80110211_lstm.pt',
'm4_ref': 'fcsm4203059_lstm.pt','f1_ref':'fcsf1200485_lstm.pt', 'm5_ref': 'fcsm5940003_lstm.pt', 'carlos_ref': 'zuretts_tts_0009_lstm.pt' ,'test_ref': 'carlos1_lstm.pt'}

for i, text in enumerate(text_es):
    sample = {"text": text, "language": 'es', "audio_unique_name": "text_ES"+str(i)}

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

    tokens = torch.unsqueeze(tokens, 0) # add batch num
    os.chdir('/home/usuaris/veu/daniel.gonzalbez/test_ref_norm') # Folder with speaker embeddings

    for file in os.listdir():
        if file.endswith('.pt'):
            spk_emb = torch.load(file).squeeze(-1)
            outputs = model.inference(tokens, aux_input = {"d_vectors":spk_emb, "language_ids":torch.tensor([0])})
            torchaudio.save("/home/usuaris/veu/daniel.gonzalbez/test_syn/" + file.replace('_lstm.pt', '_alpha6_es' + str(i)+ '.wav'), outputs["model_outputs"].squeeze().unsqueeze(0), 16000)


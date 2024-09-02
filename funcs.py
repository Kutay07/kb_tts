import re
import os
import time
import torch
import torchaudio
import queue
import shutil
from io import BytesIO
import sounddevice as sd
from TTS.api import TTS
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts


# Global variables
audio_queue = queue.Queue()
sentence_queue = queue.Queue()
model = None
speaker_latents = None
gpt_cond_latent = None
speaker_embedding = None
latent_file = 'speakers.pth'

def initialize_model():
    """Initialize the TTS model and speaker latents."""
    download_tts_model("tts_models/multilingual/multi-dataset/xtts_v2", "./models/")
    config_path = "./Models/tts_models--multilingual--multi-dataset--xtts_v2/config.json"
    checkpoint_dir = "./Models/tts_models--multilingual--multi-dataset--xtts_v2/"
    global model, gpt_cond_latent, speaker_embedding, speaker_latents
    speaker_latents = load_speaker_latents()
    model = load_model(config_path, checkpoint_dir)
    list_speakers()

def download_tts_model(model_name: str, target_path: str):
    default_download_dir = os.path.expanduser(os.path.join("~", "AppData", "Local", "tts"))

    TTS(model_name=model_name, progress_bar=True, gpu=False)

    model_folder_name = model_name.replace("/", "--")
    model_src_path = os.path.join(default_download_dir, model_folder_name)
    
    if not os.path.exists(model_src_path):
        raise FileNotFoundError(f"Model not found in the default download directory: {model_src_path}")
    
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    
    target_model_path = os.path.join(target_path, model_folder_name)
    shutil.copytree(model_src_path, target_model_path)
    
    print(f"Model '{model_name}' has been successfully copied to '{target_model_path}'.")

def load_model(config_path: str, model_path: str):
    print("Loading TTS model...")
    config = XttsConfig()
    config.load_json(config_path)
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=model_path, use_deepspeed=False)
    
    if torch.cuda.is_available():
        model.cuda()
    print("TTS model loaded.")
    return model

def load_speaker_latents():
    if not os.path.exists(latent_file):
        print(f"Speaker latents file {latent_file} not found. Cerating...")
        torch.save({}, latent_file)
    else:
        print(f"Speaker latents file {latent_file} found. Loading...")
        all_tensors = torch.load(latent_file)
    return all_tensors

def get_speaker_latents(speaker_name):
    if f'gpt_cond_latent_{speaker_name}' not in speaker_latents or f'speaker_embedding_{speaker_name}' not in speaker_latents:
        print(f"Speaker latents for {speaker_name} not found. Computing...")
        gpt_cond_latent, speaker_embedding = compute_speaker_latents(speaker_name)
    else:
        print(f"Speaker latents for {speaker_name} found. loading...")
        gpt_cond_latent = speaker_latents[f'gpt_cond_latent_{speaker_name}']
        speaker_embedding = speaker_latents[f'speaker_embedding_{speaker_name}']
    return gpt_cond_latent, speaker_embedding

def compute_speaker_latents(audio_path: str):
    print("Computing speaker latents...")
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=[audio_path])
    return gpt_cond_latent, speaker_embedding

def save_speaker_latents(audio_path: str, name: str):
    print("Saving speaker latents...")
    global speaker_latents
    speaker_latents = load_speaker_latents()
    gpt_cond_latent, speaker_embedding = get_speaker_latents(audio_path)
    speaker_latents[f'gpt_cond_latent_{name}'] = gpt_cond_latent
    speaker_latents[f'speaker_embedding_{name}'] = speaker_embedding
    torch.save(speaker_latents, latent_file)

def list_speakers():
    speakers = [key.replace('gpt_cond_latent_', '') for key in speaker_latents.keys() if 'gpt_cond_latent_' in key]
    print(f"Found {len(speakers)} speakers.")
    print(speakers)
    return speakers

def split_sentences(text):
    sentences = re.split(r'(?<=[.!?]) +', text)
    print(f"Split text into {len(sentences)} sentences.")
    return sentences

async def synthesize_sentences(text: str, language: str, speaker: str):
    gpt_cond_latent, speaker_embedding = get_speaker_latents(speaker)
    sentences = split_sentences(text)
    print(f"Synthesizing {len(sentences)} sentences...")
    print("Synthesis has started...")
    t0 = time.time()
    for i, sentence in enumerate(sentences):
        chunk = model.inference(sentence, language, gpt_cond_latent, speaker_embedding)

        if i == 0:
            print(f"Time to first chunk: {time.time()}")
        print(f"Received chunk {i}")

        sentence_queue.put(chunk)
    sentence_queue.put(None)
    print(f"Synthesis completed: {time.time() - t0}")

async def synthesize_chunks(text: str, language: str, speaker: str):
    gpt_cond_latent, speaker_embedding = get_speaker_latents(speaker)
    print("Synthesis has started...")
    t0 = time.time()
    chunks = model.inference_stream(text, language, gpt_cond_latent, speaker_embedding)
    wav_chunks = []

    for i, chunk in enumerate(chunks):
        if i == 0:
            print(f"Time to first chunk: {time.time()}")
        print(f"Received chunk {i} of audio length {chunk.shape[-1]}")

        audio_queue.put(chunk)
        wav_chunks.append(chunk)
    audio_queue.put(None)
    print(f"Synthesis completed: {time.time() - t0}")
    return wav_chunks

async def synthesize_to_json(text: str, language: str, speaker: str):
    print("Synthesis has started...")
    t0 = time.time()
    gpt_cond_latent, speaker_embedding = get_speaker_latents(speaker)
    out = model.inference(text, language, gpt_cond_latent, speaker_embedding)
    print(f"Synthesis completed: {time.time() - t0}")
    return torch.tensor(out["wav"]).numpy().tolist()

async def synthesize_to_file(text: str, language: str, speaker: str, filename: str = "output.wav"):
    print("Synthesis has started...")
    t0 = time.time()
    gpt_cond_latent, speaker_embedding = get_speaker_latents(speaker)
    out = model.inference(text, language, gpt_cond_latent, speaker_embedding)
    file_path = os.path.join(filename)
    torchaudio.save(file_path, torch.tensor(out["wav"]).unsqueeze(0), 24000)
    print(f"Synthesis completed: {time.time() - t0}")
    print(f"Audio file saved as {filename}.wav")
    return file_path

def stream_chunks():
    print("Streaming audio...")
    while True:
        chunk = audio_queue.get()
        if chunk is None:
            break
        yield chunk.numpy().tobytes()

def stream_sentences():
    print("Streaming audio...")
    while True:
        chunk = sentence_queue.get()
        if chunk is None:
            break
        yield chunk.numpy().tobytes()

# Speaker embedding and GPT latent preproccessing functions

def save_all_tensors_to_file(filename='all_tensors.pth', **tensors):
    """
    Birden fazla PyTorch tensörünü tek bir dosyaya kaydeder.

    Args:
    - filename (str): Kaydedilecek dosyanın adı.
    - tensors: Kaydedilecek tüm tensörler için anahtar-değer çiftleri.
    """
    torch.save(tensors, filename)
    print(f"All tensors saved to {filename}")

def load_all_tensors_from_file(filename='speakers.pth'):
    """
    Tek bir dosyadan birden fazla PyTorch tensörünü yükler.

    Args:
    - filename (str): Yüklenecek dosyanın adı.

    Returns:
    - dict: Dosyadan yüklenen tüm tensörlerin bir sözlüğü.
    """
    print("Loading tensors...")
    all_tensors = torch.load(filename)
    print(f"All tensors loaded from {filename}")
    return all_tensors

def extract_name_from_filename(filename):
    """
    Dosya adından köşeli parantezler içindeki ismi çıkarır.

    Args:
    - filename (str): İşlenecek dosya adı.

    Returns:
    - str: Köşeli parantezler arasındaki isim.
    """
    # Dosya adını tam yol yerine sadece isim olarak al
    base_filename = os.path.basename(filename)
    match = re.search(r"\[(.*?)\]", base_filename)
    print(f"\nextracted name: {match.group(1)} \nfilename: {filename} \nbase_filename: {base_filename}\n")
    return match.group(1) if match else None

def process_and_save_all_audio_files(model, directory_path, output_filename='all_latents.pth'):
    """
    Belirtilen klasördeki tüm ses dosyalarını işleyerek GPT koşullu latent ve konuşmacı gömme tensörlerini oluşturur ve tek bir dosyaya kaydeder.

    Args:
    - model: GPT modeli.
    - directory_path (str): İşlenecek ses dosyalarının bulunduğu klasör yolu.
    - output_filename (str): Kaydedilecek dosyanın adı.
    """
    # Tüm tensörleri saklamak için bir sözlük oluştur
    all_tensors = {}
    
    # Belirtilen klasördeki tüm .wav dosyalarını bulun
    audio_files = [f for f in os.listdir(directory_path) if f.endswith('.wav')]
    
    # Her ses dosyası için döngü
    for idx, audio_file in enumerate(audio_files):
        # Köşeli parantezler içindeki ismi çıkar
        name = extract_name_from_filename(audio_file)
        if not name:
            print(f"Name not found in {audio_file}, skipping.")
            continue

        audio_path = os.path.join(directory_path, audio_file)
        print(f"Processing file {idx + 1}/{len(audio_files)}: {audio_path}")
        
        # Ses dosyasını işleyin ve tensörleri alın
        gpt_cond_latent, speaker_embedding = compute_speaker_latents(model, audio_path)
        
        # Sözlüğe ekle
        all_tensors[f'gpt_cond_latent_{name}'] = gpt_cond_latent
        all_tensors[f'speaker_embedding_{name}'] = speaker_embedding
    
    # Tüm tensörleri tek bir dosyaya kaydet
    save_all_tensors_to_file(output_filename, **all_tensors)
    print(f"All tensors processed and saved to {output_filename}")

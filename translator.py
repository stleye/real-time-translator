import warnings
warnings.filterwarnings("ignore")
import os
os.environ["HF_HUB_OFFLINE"] = "1"

import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
import threading
import queue
import tempfile
import argparse
import torch
import mlx_whisper
from deep_translator import GoogleTranslator
from silero_vad import load_silero_vad, get_speech_timestamps

LANGUAGES = {
    "auto": None, "chinese": "zh", "english": "en",
    "spanish": "es", "portuguese": "pt", "japanese": "ja",
    "korean": "ko", "arabic": "ar", "russian": "ru",
    "french": "fr", "german": "de",
}

MODELS = {
    "tiny":           "mlx-community/whisper-tiny-mlx",
    "small":          "mlx-community/whisper-small-mlx",
    "medium":         "mlx-community/whisper-medium-mlx",
    "large-v3":       "mlx-community/whisper-large-v3-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
    "large-v3-turbo-q4": "mlx-community/whisper-large-v3-turbo-q4",
}

FS = 16000
BLOCK = 512  # ~32ms por bloque

def parse_args():
    p = argparse.ArgumentParser(description="Traductor de audio en tiempo real")
    p.add_argument("--lang", default="auto", choices=LANGUAGES.keys())
    p.add_argument("--target", default="es")
    p.add_argument("--model", default="large-v3-turbo-q4", choices=MODELS.keys())
    p.add_argument("--device", type=int, default=None)
    p.add_argument("--list-devices", action="store_true")
    p.add_argument("--min-seconds", type=float, default=1.0, help="Segundos mínimos de voz antes de traducir")
    p.add_argument("--silence-seconds", type=float, default=0.8, help="Segundos de silencio para cortar frase")
    return p.parse_args()

def listar_dispositivos():
    print("\nDispositivos de entrada disponibles:")
    for i, d in enumerate(sd.query_devices()):
        if d['max_input_channels'] > 0:
            print(f"  [{i}] {d['name']}")
    print()

def main():
    args = parse_args()

    if args.list_devices:
        listar_dispositivos()
        return

    listar_dispositivos()

    if args.device is None:
        entrada = input("Ingresá el número del dispositivo (Enter para default): ").strip()
        device_id = int(entrada) if entrada else None
    else:
        device_id = args.device

    print(f"\nCargando VAD...")
    vad = load_silero_vad()

    model_path = MODELS[args.model]
    idioma_fuente = LANGUAGES.get(args.lang)

    print(f"Modelo: {args.model} (MLX — Apple Silicon)")
    print(f"Idioma fuente: {args.lang}")
    print(f"Traduciendo a: {args.target}")
    print(f"Silencio para corte: {args.silence_seconds}s")
    print("\nEscuchando... Ctrl+C para detener.\n")

    cola = queue.Queue()
    audio_buffer = []
    silencio_bloques = 0
    voz_detectada = False
    silence_bloques_max = int(args.silence_seconds * FS / BLOCK)
    min_bloques = int(args.min_seconds * FS / BLOCK)
    max_bloques = int(15 * FS / BLOCK)  # cortar a los 15s como máximo

    def callback(indata, frames, time, status):
        nonlocal silencio_bloques, voz_detectada

        bloque = indata[:, 0].copy()

        # Normalizar para VAD
        nivel = np.max(np.abs(bloque))
        if nivel > 0.001:
            bloque_norm = bloque / nivel * 0.9
        else:
            bloque_norm = bloque

        # Detectar voz con Silero VAD
        tensor = torch.from_numpy(bloque_norm).float()
        prob = vad(tensor, FS).item()
        hay_voz = prob > 0.5

        if hay_voz:
            audio_buffer.append(bloque)
            silencio_bloques = 0
            voz_detectada = True
        elif voz_detectada:
            audio_buffer.append(bloque)
            silencio_bloques += 1
            if silencio_bloques >= silence_bloques_max and len(audio_buffer) >= min_bloques:
                duracion = len(audio_buffer) * BLOCK / FS
                print(f"[VAD] corte por silencio — {duracion:.1f}s acumulados")
                audio = np.concatenate(audio_buffer)
                audio_buffer.clear()
                silencio_bloques = 0
                voz_detectada = False
                nivel = np.max(np.abs(audio))
                if nivel > 0.001:
                    audio = audio * (0.9 / nivel)
                cola.put(audio)

        # Corte forzado si el buffer crece demasiado (ej: música continua)
        if len(audio_buffer) >= max_bloques:
            duracion = len(audio_buffer) * BLOCK / FS
            print(f"[VAD] corte forzado — {duracion:.1f}s acumulados, voz={voz_detectada}")
            audio = np.concatenate(audio_buffer)
            audio_buffer.clear()
            silencio_bloques = 0
            nivel = np.max(np.abs(audio))
            if nivel > 0.001:
                audio = audio * (0.9 / nivel)
            cola.put(audio)

    def procesar():
        while True:
            audio = cola.get()
            if audio is None:
                break

            tmp = tempfile.mktemp(suffix=".wav")
            wav.write(tmp, FS, (audio * 32767).astype(np.int16))

            resultado = mlx_whisper.transcribe(
                tmp,
                path_or_hf_repo=model_path,
                language=idioma_fuente,
            )
            texto = resultado["text"].strip()
            lang_detectado = resultado.get("language", "?")

            if texto:
                traducido = GoogleTranslator(source='auto', target=args.target).translate(texto)
                print(f"[{lang_detectado}] → {traducido}\n")

    hilo = threading.Thread(target=procesar, daemon=True)
    hilo.start()

    try:
        with sd.InputStream(samplerate=FS, channels=1, dtype='float32',
                            blocksize=BLOCK, device=device_id, callback=callback):
            print("VAD activo — esperando voz...\n")
            while True:
                sd.sleep(100)
    except KeyboardInterrupt:
        print("\nDetenido.")
        cola.put(None)

if __name__ == "__main__":
    main()

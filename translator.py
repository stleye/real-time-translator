import warnings
warnings.filterwarnings("ignore")

import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
import threading
import queue
import tempfile
import argparse
import mlx_whisper
from deep_translator import GoogleTranslator

LANGUAGES = {
    "auto": None, "chinese": "zh", "english": "en",
    "spanish": "es", "portuguese": "pt", "japanese": "ja",
    "korean": "ko", "arabic": "ar", "russian": "ru",
    "french": "fr", "german": "de",
}

MODELS = {
    "tiny":            "mlx-community/whisper-tiny-mlx",
    "small":           "mlx-community/whisper-small-mlx",
    "medium":          "mlx-community/whisper-medium-mlx",
    "large-v3":        "mlx-community/whisper-large-v3-mlx",
    "large-v3-turbo":  "mlx-community/whisper-large-v3-turbo",
}

def parse_args():
    p = argparse.ArgumentParser(description="Traductor de audio en tiempo real")
    p.add_argument("--lang", default="auto", choices=LANGUAGES.keys(), help="Idioma fuente (default: auto)")
    p.add_argument("--target", default="es", help="Idioma destino (default: es)")
    p.add_argument("--model", default="large-v3-turbo", choices=MODELS.keys(), help="Modelo Whisper")
    p.add_argument("--chunk", type=int, default=8, help="Segundos por chunk (default: 8)")
    p.add_argument("--device", type=int, default=None, help="ID del dispositivo de entrada")
    p.add_argument("--list-devices", action="store_true", help="Listar dispositivos y salir")
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

    model_path = MODELS[args.model]
    idioma_fuente = LANGUAGES.get(args.lang)

    print(f"\nModelo: {args.model} (MLX — Apple Silicon)")
    print(f"Idioma fuente: {args.lang}")
    print(f"Traduciendo a: {args.target}")
    print(f"Chunk: {args.chunk}s")
    print("\nEscuchando... Ctrl+C para detener.\n")

    FS = 16000
    cola = queue.Queue()
    buffer = []

    def callback(indata, frames, time, status):
        buffer.append(indata.copy())

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
                            device=device_id, callback=callback):
            while True:
                sd.sleep(args.chunk * 1000)
                if buffer:
                    audio = np.concatenate(buffer)
                    buffer.clear()
                    nivel = np.max(np.abs(audio))
                    if nivel > 0.001:
                        audio = audio * (0.9 / nivel)
                        cola.put(audio)
                    else:
                        print("[silencio]")
    except KeyboardInterrupt:
        print("\nDetenido.")
        cola.put(None)

if __name__ == "__main__":
    main()

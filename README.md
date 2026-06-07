# Real-Time Translator

Traductor de audio en tiempo real usando Whisper (MLX) y Google Translate. Detecta el idioma automáticamente y traduce al idioma destino con baja latencia en Apple Silicon.

## Requisitos

- Mac con Apple Silicon (M1/M2/M3/M4)
- Python 3.10+
- BlackHole 2ch (para capturar audio del sistema)

```bash
python -m venv .venv
source .venv/bin/activate
pip install mlx-whisper sounddevice deep-translator scipy numpy
```

## Uso

```bash
# Auto-detectar idioma, traducir al español
python translator.py --target es --device 2

# Especificar idioma fuente
python translator.py --lang chinese --target es --device 2

# Ver dispositivos de audio disponibles
python translator.py --list-devices

# Cambiar modelo y tamaño de chunk
python translator.py --target es --model large-v3-turbo --chunk 8
```

## Opciones

| Argumento | Default | Descripción |
|---|---|---|
| `--lang` | auto | Idioma fuente (auto, chinese, english, spanish, portuguese, japanese, korean, arabic, russian, french, german) |
| `--target` | es | Idioma destino (código ISO: es, en, fr, de, pt...) |
| `--model` | large-v3-turbo | Modelo Whisper (tiny, small, medium, large-v3, large-v3-turbo) |
| `--chunk` | 8 | Segundos por chunk de audio |
| `--device` | — | ID del dispositivo de entrada (ver --list-devices) |

## Capturar audio del sistema (WebSDR, YouTube, etc.)

1. Instalá [BlackHole 2ch](https://existential.audio/blackhole/)
2. En Audio MIDI Setup creá un **Multi-Output Device** con tus parlantes + BlackHole 2ch
3. Seleccioná Multi-Output Device como salida en Preferencias de Sonido
4. Corré el traductor con `--device 2` (o el ID que corresponda a BlackHole)

Así escuchás el audio por los parlantes y el traductor lo captura simultáneamente.

## Modelos disponibles

Los modelos se descargan automáticamente de HuggingFace la primera vez.

| Modelo | Tamaño | Velocidad en M4 |
|---|---|---|
| tiny | ~40 MB | Muy rápido |
| small | ~250 MB | Rápido |
| medium | ~800 MB | Moderado |
| large-v3-turbo | ~800 MB | Rápido (recomendado) |
| large-v3 | ~1.5 GB | Preciso |

## Caso de uso

Desarrollado para traducir transmisiones de onda corta capturadas via [WebSDR](http://websdr.org/) en tiempo real.

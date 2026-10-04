# Neuro-Enhance

**Neuro-Enhance - An open-source, GPU-accelerated AI image & photo enhancer powered by neural networks.**<br>
**Neuro-Enhance - ein quelloffener, GPU-beschleunigter KI-Bild- und Fotoverbesserer auf Basis neuronaler Netze.**

[![CI](https://github.com/DerAlexmann/Neuro-Enhance/actions/workflows/ci.yml/badge.svg)](https://github.com/DerAlexmann/Neuro-Enhance/actions/workflows/ci.yml)

Supports hardware acceleration via NVIDIA CUDA / RTX series GPUs.

*NVIDIA and RTX are trademarks of NVIDIA Corporation.*

*[Deutsche Fassung: README.md](README.md)*

> **Early development.** The classic basic adjustments already run on the GPU; the AI
> features will follow in the next versions.

## System requirements

Neuro-Enhance computes on the graphics card only. **Without an NVIDIA RTX graphics card
the program does not start**; instead it reports what was found and what is missing.
There is deliberately no fallback mode on the CPU.

| | Minimum |
|---|---|
| Operating system | Windows 10 or 11, 64-bit |
| Graphics card | NVIDIA GeForce RTX 2000 series or newer, including professional RTX cards (RTX A series, RTX Ada) |
| Video memory | 4 GB for AI features; below that classic filters only |
| Graphics driver | NVIDIA 570.65 or newer |
| Python (only when run as a script) | 3.10 or newer |

**Not supported** are GTX cards – including the GTX 16 series, which belongs to the same
generation as the RTX 2000 but has no Tensor Cores – and graphics cards from other
vendors.

### Feature tiers

What is available depends mainly on the video memory. Neuro-Enhance detects the tier at
start-up; features of a higher tier are shown greyed out with a note.

| Tier | Video memory | Examples | Scope |
|---|---|---|---|
| – | below 4 GB | RTX 2050 (laptop) | classic filters only |
| S | 4 GB | RTX 3050 Laptop | classic filters, small AI models |
| M | 6–8 GB | RTX 2060, 3060 Ti, 4060, 5060 | all restoration models: upscaling, denoising, background removal, object selection |
| L | 12–16 GB | RTX 3060 12 GB, 4070, 5070 Ti | in addition larger models and generative fill |
| XL | 24 GB and more | RTX 3090, 4090, 5090 | full feature set |

Precision: all RTX cards compute in FP16; from the RTX 4000 (Ada) also in FP8, from the
RTX 5000 (Blackwell) also in FP4. The values are shown on the "About & copyright" tab.

## What it does already

- **Basic adjustments on the GPU**: white balance (temperature, tint), exposure,
  contrast, highlights, shadows, vibrance, saturation and sharpening with adjustable
  radius.
- **Tone curve** for luminance and for red, green and blue separately, with a histogram
  in the background. Click to add a point, drag to move it, double-click to remove it;
  the luminance curve leaves the colours untouched.
- **Clarity**: more or less local contrast in the midtones.
- **Dehaze** using the dark channel prior – also in reverse, to add haze.
- **Non-destructive**: every change is recomputed from the untouched original.
  Double-click resets a slider, "Before" shows the original while the button is held.
- **Fast**: the preview is computed at screen resolution in a few milliseconds; the
  computing time is shown in the status bar. Saving uses the full resolution.
- **Colour-accurate**: processing happens in linear light with 32-bit floating point.
  RGB colour profiles such as Adobe RGB, ProPhoto RGB or Display P3 are converted to sRGB
  on the GPU without a detour through 8 bits; colours outside sRGB are kept until the
  output. The EXIF orientation is applied; EXIF data and an alpha channel are kept when
  saving.
- **16 bits and RAW**: opens and saves TIFF and PNG with 16 bits per channel. RAW files
  from practically every camera (CR2, CR3, NEF, ARW, RAF, ORF, RW2, DNG and more) are
  developed in linear light without automatic brightening – the full range of the sensor
  is kept, and exposure controls the brightness. For Bayer sensors, i.e. almost every
  camera, demosaicing runs on the GPU (Malvar-He-Cutler): a 24-megapixel RAW opens in
  about 0.15 s instead of almost a second. Fujifilm X-Trans and other sensors are still
  developed by LibRaw.
- **Formats**: opens JPEG, PNG, TIFF, WebP, BMP and RAW (HEIC with the optional
  `pillow-heif` package); saves JPEG and WebP with 8 bits, PNG and TIFF with 8 or 16 bits.
  For images with more than 8 bits the save dialog suggests a 16-bit TIFF.

| Key | Action |
|---|---|
| `Ctrl`+`O` | open an image (or drag a file onto the canvas) |
| `Ctrl`+`S` | save as … – the original is never overwritten silently |

## Planned

- **More classic filters**: LUTs, HSL per colour range, denoising, geometry.
- **Finer demosaicing**: a more elaborate method (such as RCD) for even fewer colour
  fringes in fine structures, and X-Trans on the GPU.
- **AI features**: upscaling, denoising, deblurring, background removal, click-to-select
  objects, object removal, depth maps for synthetic depth of field.

## Running it

**As a script:**

```bash
pip install -r requirements.txt
python Neuro-Enhance.pyw
```

The packages bring CuPy and the required CUDA libraries from NVIDIA (just over 1 GB in
total); a separate CUDA Toolkit installation is not needed. The first time a filter is
used, the graphics card compiles the matching program once; this takes one to three
seconds and is kept for all later starts.

A ready-made executable will follow with the first release.

## Privacy

Neuro-Enhance works entirely on your own computer. No images are uploaded and no usage
data is sent. In later versions the program downloads AI models **only on explicit
request**, from the source stated for each model.

## A note on AI results

AI methods such as upscaling or denoising add image details that were not present in the
original. Processed images are therefore not suitable as evidence or for documentation
purposes.

## Contributing

Bug reports, suggestions and translations are welcome – [CONTRIBUTING.md](CONTRIBUTING.md)
explains structure, style and tests (in German; issues and pull requests in English are
just as welcome). Please report security issues not as an issue but as described in
[SECURITY.md](SECURITY.md).

## Licence and legal notices

[MIT](LICENSE) – Copyright 2026 Alexander Unverhau.
Created with assistance of Claude AI.

Third-party components and their licences are listed in [NOTICE](NOTICE). AI models are
subject to their own licences, which are shown before downloading.

**Trademarks:** NVIDIA, RTX, GeForce, CUDA and TensorRT are trademarks or registered
trademarks of NVIDIA Corporation in the U.S. and other countries. Qt is a trademark of The
Qt Company Ltd., Python a trademark of the Python Software Foundation, Windows a trademark
of Microsoft Corporation. All other trademarks are the property of their respective
owners.

Neuro-Enhance is an independent project and is not affiliated with, endorsed or sponsored
by NVIDIA Corporation.

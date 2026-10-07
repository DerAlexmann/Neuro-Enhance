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
- **Colour ranges (HSL)**: hue, saturation and luminance separately for red, orange,
  yellow, green, aqua, blue, purple and magenta. Computed in the perceptually uniform
  OkLCh colour space with soft transitions between the ranges; greys stay untouched.
- **LUTs**: load looks as `.cube` files (3D and 1D), tetrahedral interpolation, with a
  strength slider.
- **Noise reduction**: luminance noise with non-local means, colour noise with a
  colour-guided filter that keeps colour edges – even between colours of equal
  brightness. The scaled-down preview shows less noise than the saved image – use the
  100 % view to judge it.
- **AI denoise** with SCUNet, trained on real camera noise and for fidelity rather than
  invented detail: computed once over the whole image (an 18 MP photo takes about 20
  seconds on an RTX 4060, 11 with TensorRT); after that a strength slider instantly blends
  between original and denoised image – in the preview, the 100 % view and the export
  alike. Highlights above white, for example from RAWs, are preserved.
- **AI sharpen** for slight focus blur with Restormer, trained on real out-of-focus
  photos from a DSLR and for fidelity: eyelashes, hair and skin texture become clearer
  again, without the halos of classic sharpening. Computed once over the whole image
  (24 MP on an RTX 4060: about three minutes, 45 seconds with TensorRT); after that a
  strength slider responds instantly. If AI denoise has been computed, the denoised image
  is sharpened; only the fine part of the correction is applied, brightness and colour
  stay unchanged.
- **Subject & background** with BiRefNet: the AI detects the subject – people, animals,
  objects – including hair in about two seconds. After that, exposure, contrast,
  saturation, temperature and blur of the background can be set separately, for example
  to darken or soften it; "Invert" applies the sliders to the subject. The edge can be
  softened and shifted, and a mask view tints the background for checking. When saving
  as PNG or TIFF, the background can be made transparent. Needs at least 6 GB of graphics
  memory.
- **Select an object by clicking** with Segment Anything 2 (SAM 2.1), in the same card:
  left-click in the image adds an object or an area, right-click removes one, Ctrl+Z
  undoes the last click. Every click shows the selection within a fraction of a second;
  the sliders, "Invert" and transparent saving work on it like on the detected subject.
  Works from 4 GB of graphics memory.
- **Remove objects** with LaMa: click an object (SAM 2) or paint over it with the brush,
  for example spots or wires – "Remove" fills the spot within a fraction of a second with
  what could be behind it. Several removals build on each other, the last one can be
  undone; the sliders keep working on the whole image. From 4 GB of graphics memory.
- **Geometry**: rotate by 90°, flip, straighten, vertical and horizontal perspective, crop
  with a frame and fixed aspect ratios. Empty corners after straightening are cropped
  away automatically.
- **Lens**: correct distortion, vignetting and colour fringes (chromatic aberration). All
  geometry and lens corrections are resampled in a single bicubic step.
- **AI upscaling** by 2 × or 4 × when saving, with Real-ESRGAN via ONNX Runtime on the
  GPU: a fast model with a denoise strength slider (from 4 GB of video memory) and a
  large model with more sharpness (from feature tier M). Processing runs in half
  precision (FP16) on the tensor cores and in tiles sized to the video memory; with the
  fast model an RTX 4060 upscales a 24 MP image to 96 MP in just under 6 seconds. The
  optional [TensorRT](#tensorrt-optional) makes it about twice as fast again.
- **100 % view**: zoom to 100, 200 and 400 % with the mouse wheel, pan by dragging,
  double-click toggles between fit and 100 %. It shows a part of the image at full
  resolution – exactly what will be saved.
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
  camera, demosaicing runs on the GPU – with RCD (ratio corrected demosaicing), which
  leaves particularly few colour fringes in fine patterns. A 24-megapixel RAW opens in
  about 0.2 s instead of almost a second. Fujifilm X-Trans and other sensors are still
  developed by LibRaw.
- **Formats**: opens JPEG, PNG, TIFF, WebP, BMP and RAW (HEIC with the optional
  `pillow-heif` package); saves JPEG and WebP with 8 bits, PNG and TIFF with 8 or 16 bits.
  For images with more than 8 bits the save dialog suggests a 16-bit TIFF.

| Key | Action |
|---|---|
| `Ctrl`+`O` | open an image (or drag a file onto the canvas) |
| `Ctrl`+`S` | save as … – the original is never overwritten silently |
| `Ctrl`+`0` / `Ctrl`+`1` | fit / 100 % |
| mouse wheel, drag, double-click | zoom, pan, toggle between fit and 100 % |
| `Enter` / `Esc` | apply / cancel crop |

## Planned

- **Lens profiles**: distortion, vignetting and colour fringes corrected automatically
  from a lens database (lensfun).
- **X-Trans on the GPU**: demosaicing for Fujifilm sensors on the graphics card as
  well.
- **More AI features**: denoising, deblurring, background removal, click-to-select
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

### TensorRT (optional)

With TensorRT the AI runs about twice as fast again – on an RTX 4060 the large model
upscales a 12 MP image in 24 instead of 54 seconds:

```bash
pip install -r requirements-tensorrt.txt
```

That is about 1.8 GB to download (2.8 GB installed) from NVIDIA's package index.
Neuro-Enhance detects TensorRT by itself; "Info & Copyright" shows the version. The
first time a model is used, TensorRT builds an engine for the graphics card once – this
takes one to three minutes (up to ten for AI sharpen and subject detection), and a
window says so. The engines are then kept in the
`modelle/tensorrt` folder. If TensorRT does not work, the program computes with CUDA as
it does without it.

TensorRT is licensed under a proprietary NVIDIA licence that applies on installation.
It does not allow TensorRT to be passed on together with Neuro-Enhance – which is why
it is not part of any ready-made executable and is always installed separately.

## Privacy

Neuro-Enhance works entirely on your own computer. No images are uploaded and no usage
data is sent. The AI also runs locally on the graphics card. The only network access is
downloading an AI model – and only when you explicitly ask for it in the "AI upscaling",
"AI denoise", "AI sharpen", "Subject & background" or "Remove objects" card and confirm the prompt that names source, size and
licence.

## AI models

The models are not part of the program. "Download model" fetches them from releases of
this project: in the "AI upscaling" card from
[modelle-1](https://github.com/DerAlexmann/Neuro-Enhance/releases/tag/modelle-1) the fast
model with just under 10 MB and the large one with 64 MB, in the "AI denoise" card from
[modelle-2](https://github.com/DerAlexmann/Neuro-Enhance/releases/tag/modelle-2) SCUNet
with 71 MB, in the "AI sharpen" card from
[modelle-3](https://github.com/DerAlexmann/Neuro-Enhance/releases/tag/modelle-3) Restormer
with 101 MB, in the "Subject & background" card from
[modelle-4](https://github.com/DerAlexmann/Neuro-Enhance/releases/tag/modelle-4) BiRefNet
with 177 MB and for "Click object" from
[modelle-5](https://github.com/DerAlexmann/Neuro-Enhance/releases/tag/modelle-5) SAM 2
with 147 MB, in the "Remove objects" card from
[modelle-6](https://github.com/DerAlexmann/Neuro-Enhance/releases/tag/modelle-6) LaMa
with 196 MB. Every file is checked against its SHA-256 checksum before it is used. They are stored in the `modelle`
folder next to the program or, if that is read-only, in
`%LOCALAPPDATA%\Neuro-Enhance\modelle`.

The models are the official weights of [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
(BSD 3-Clause, Copyright 2021 Xintao Wang), [SCUNet](https://github.com/cszn/SCUNet)
(Apache-2.0, Copyright 2022 Kai Zhang), [Restormer](https://github.com/swz30/Restormer)
(MIT, Copyright 2022 Syed Waqas Zamir), [BiRefNet](https://github.com/ZhengPeng7/BiRefNet)
(MIT, Copyright 2024 ZhengPeng) and [SAM 2](https://github.com/facebookresearch/sam2)
(Apache-2.0, Copyright Meta Platforms, Inc. and affiliates) and
[LaMa](https://github.com/advimman/lama) (Apache-2.0, Copyright 2021 Samsung Research),
converted to ONNX. To reproduce this yourself:

1. Download `realesr-general-x4v3.pth`, `realesr-general-wdn-x4v3.pth` and
   `RealESRGAN_x4plus.pth` from the
   [Real-ESRGAN releases](https://github.com/xinntao/Real-ESRGAN/releases), and
   `scunet_color_real_psnr.pth` from the
   [KAIR release v1.0](https://github.com/cszn/KAIR/releases/tag/v1.0), where the SCUNet
   weights are published, and `single_image_defocus_deblurring.pth` from the
   [Restormer release v1.0](https://github.com/swz30/Restormer/releases/tag/v1.0) and
   `BiRefNet_lite-general-2K-epoch_232.pth` from the
   [BiRefNet release v1](https://github.com/ZhengPeng7/BiRefNet/releases/tag/v1) and
   [`sam2.1_hiera_small.pt`](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt)
   from Meta and `big-lama.zip` of LaMa (the download named in the
   [LaMa instructions](https://github.com/advimman/lama), hosted on
   [Hugging Face](https://huggingface.co/smartywu/big-lama)) into `modelle/quellen/`
   and unpack it there.
2. `pip install torch` (only needed for this step).
3. `python werkzeuge/modelle_exportieren.py` – this writes the ONNX models to `modelle/`
   and checks them against PyTorch.

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

# STANCE: Motion-Coherent Video Generation via Sparse-to-Dense Anchored Encoding

[ZhiFei Chen]()$^{{\*}}$, [Tianshuo Xu]()$^{{\*}}$, [Leyi Wu]()$^{{\*}}$, [Luozhou Wang](), [Dongyu Yan](), [Zihan You](), [Wenting Luo](),[Guo Zhang](), [Yingcong Chen](https://www.yingcong.me)$^{\**}$

HKUST(GZ), HKUST, XMU, MIT

${\*}$: Equal contribution.
\**: Corresponding author.

<a href="#"><img src="https://img.shields.io/badge/Project_Page-Coming_Soon-lightgrey"></a> <a href="#"><img src="https://img.shields.io/badge/Paper-Under_Review-blue"></a> <a href="#"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Gradio%20Demo-Coming%20Soon-lightgrey"></a>

---

## 🎏 Introduction

**STANCE** is a controllable image-to-video framework that keeps motion consistent while preserving appearance. Users provide a keyframe plus **instance masks**, **coarse 2D arrows** (direction/speed), an optional **depth delta** (2.5D), and **per-instance mass**. We convert these sparse hints into **dense, pixel-aligned motion cues** and inject them with **Dense RoPE** tokens so the control remains strong after tokenization. We also **jointly predict RGB + a lightweight structural map** (depth or segmentation) to stabilize temporal coherence. 

<details>
<summary>CLICK for the full abstract-style summary</summary>

* **Problem.** Purely visual video diffusion looks great but often drifts or “hovers” near contacts; sparse control maps get washed out after encoding.
* **Key idea.** Turn human-editable hints into a **dense, 2.5D instance cue** per object; keep those cues salient in token space via **Dense RoPE** (spatially addressable motion tokens anchored on the first frame). Train RGB **with** an auxiliary structural head to act as a geometry/consistency witness. 
* **Result.** Better direction/speed/mass faithfulness, cleaner contact onsets, and less drift—without frame-by-frame trajectories. 

</details>

---

## 💡 Method at a glance

> *Pipeline figure placeholder*
> `<img src="assets/pipeline.png" width="80%">`
> *Overall Architecture*

**Instance Cues (Sparse → Dense, 2.5D).** From per-instance arrows + masks (+ optional depth delta), we rasterize a **dense in-mask vector field** and append a **scalar ∆z** channel (camera-relative), disambiguating out-of-plane intent under camera motion. Training uses per-instance averaged flow (+ monocular depth) to match the test-time cue format. 

**Dense RoPE (Token-dense, spatially addressable control).** Downsampling makes low-res control maps too sparse. We **extract non-zero sites**, enforce a **fixed motion-token budget**, and **tag them with first-frame RoPE** so their spatial identity persists over time—keeping control strong post-encoding. 

**Joint Auxiliary Generation (RGB + Depth/Seg).** We duplicate the video token stream so the model predicts both RGB and a structural map under the **same** cues/positions; a tiny domain tag distinguishes modalities. This anchors geometry and reduces drift while RGB handles appearance. 

---

## 🎮 What can STANCE edit?

* **Speed & direction sweeps:** Increasing |v₀| yields longer travel and earlier contact; rotating the arrow rotates the trajectory while preserving appearance. 
* **Mass sweeps:** Changing mass flips post-contact outcomes (e.g., light object deflects vs. heavy object pushes through). 
* **Real-world tabletop demos:** Identity-preserving motion and plausible chain reactions from a phone-captured keyframe. 

> *Applications figure placeholder*
> `<img src="assets/applications.png" width="80%">`

---

## ⚙️ Installation

We recommend **Python ≥ 3.9**, **PyTorch ≥ 2.3** with CUDA 12.x.

```bash
# create env
conda create -n stance python=3.9 -y
conda activate stance

# install torch (pick your CUDA build)
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio

# essentials
pip install -r requirements.txt     # we'll provide pinned deps with the code release

# (optional) segmentation for masks — SAM / SAM2
pip install git+https://github.com/facebookresearch/segment-anything.git
# or:
# pip install git+https://github.com/facebookresearch/sam2.git
```

> **Backbone.** We fine-tune a **CogVideoX-1.5 (5B) image-to-video** backbone; default generation is **512×512, 49 frames @ 16 FPS**. 

---

## 📦 Data

We provide Kubric rendering scripts (to be released) covering **200k** rigid-interaction clips across (i) simple multi-object collisions and (ii) composite realistic scenes. We randomize object shape, mass, initial velocity, placement/pose, and backgrounds; we keep camera intrinsics/extrinsics consistent within a scene. Held-out validation/test avoid asset/scene leakage. 



## 💫 Training

```bash
# single-node example (configs are placeholders)
python train.py \
  --config configs/stance_cogvidx.yaml \
  --data data/stance_cues \
  --output runs/stance_cogvidx

# or a simple launcher
bash scripts/train.sh
```

* **Cue injection:** Instance Cues (u, v, ∆z, mass) are concatenated with video inputs and turned into motion tokens via **Dense RoPE**. 
* **Objective:** Standard diffusion loss with a single weight for the auxiliary head (depth or seg). 

---



## 🚩 Features / Roadmap

* [✅] Code release (training & inference)
* [✅] Kubric dataset & generation scripts
* [ ] Pretrained checkpoints (Dense RoPE; RGB+Depth / RGB+Seg)
* [ ] Gradio Demo for better usage


---


## 📄 BibTeX

> Update after arXiv goes live

```bibtex
@inproceedings{STANCE2026,
  title   = {STANCE: Motion-Coherent Video Generation via Sparse-to-Dense Anchored Encoding},
  author  = {TBD},
  booktitle = {ICLR},
  year    = {2026}
}
```



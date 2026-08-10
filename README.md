# Agentic Medical AI Lab — brain tumour segmentation

**AI in Life Science Summer School · Aarhus, 17–18 August**
Kim Beuschau Mouridsen, Aarhus University

You will build, evaluate and stress-test a medical image segmentation model —
using **Claude Code** to write the software, and your own judgement to decide
whether the result can be believed.

You do not need to be a programmer. You need to be a good scientist.

---

## Setup — do this *before* Monday

**1. Install Python 3.11 or 3.12** from [python.org](https://www.python.org/downloads/)
   *(Windows: tick **"Add Python to PATH"** during install. Avoid the Microsoft Store version.)*

**2. Install Claude Code** — see [code.claude.com/docs](https://code.claude.com/docs/en/quickstart).
   *(Windows: also install [Git for Windows](https://git-scm.com/download/win) — it gives Claude Code a proper shell.)*

**3. Get this repository and check your setup**

```bash
git clone https://github.com/ORG/agentic-medical-ai-lab
cd agentic-medical-ai-lab
python -m venv .venv
```

Activate the environment:

| | |
|---|---|
| **macOS / Linux** | `source .venv/bin/activate` |
| **Windows** | `.venv\Scripts\activate` |

Then:

```bash
python -m pip install -r requirements.txt
python verify.py
```

`verify.py` prints a checklist. **If it says ALL GOOD, you are ready.**
If not, it tells you exactly what to fix.

> **Stuck?** Don't lose your morning to it. Click
> **Code → Codespaces → Create codespace** on GitHub — you get the whole
> environment, data included, in the browser. Nothing is lost.

---

## What's here

```
data/cases/       60 cases · 4 MRI channels · expert tumour masks
verify.py         environment check — run this first
```

The exercises are not in this repository — they are handed out during the lab.
There is no solution code here either, by design. Everything you produce is
something you build by directing Claude — that is the whole point of the two
days.

---

## The data

60 brain MRI cases (`case_001` … `case_060`), each with four co-registered
channels and an expert-drawn tumour mask:

| Channel | |
|---|---|
| **FLAIR** | fluid-attenuated — tumour and oedema appear bright |
| **T1w** | anatomy |
| **T1-Gd** | after contrast — active tumour enhances |
| **T2w** | tumour and oedema bright |

Volumes are `240 × 240 × 155`, 1 mm isotropic, skull-stripped and
co-registered. Labels are 1 = oedema, 2 = non-enhancing core, 3 = enhancing
tumour; we treat **any label > 0 as "tumour"**.

### Work in two tiers

| Tier | Cases | Use it for |
|---|---|---|
| **Tiny** — `tier_tiny.txt` | 15 | **Developing.** Runs in seconds even on a slow laptop. Every large effect — normalisation, thresholding — reproduces fully. |
| **Standard** — `tier_standard.txt` | 60 | **Reporting.** Quote your final numbers on all 60. Subtle effects (e.g. multi-channel vs single-channel) only become statistically distinguishable here. |

Develop on the tiny set, report on the standard set. That is ordinary practice —
and a lesson in itself: on 15 cases you cannot tell a four-channel model from a
single-channel one (p = 0.28); on 60 you can (p = 0.004). Your sample size
decides what you are entitled to conclude.

A larger 424-case set exists for anyone who wants to push further — ask.

The cases carry neutral identifiers on purpose: for these two days we want you
reasoning from the images in front of you, not from published results for a
named benchmark. The source and citation are therefore held back until the end
of the lab, and published here with the rest of the course material afterwards.
The data is CC-BY-SA 4.0; code and teaching material are MIT.

---

## Ground rules for the two days

1. **Ask Claude for its plan before it writes files.**
2. **Never accept a number you have not seen a figure for.**
3. **Decide what a good result looks like before you produce one.**
4. **Report what you found, not what you hoped for.**

These are the whole course, compressed.

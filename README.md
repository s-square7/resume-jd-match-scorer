# Resume-JD Siamese BiLSTM Match Scorer

> A deep neural network that reads a candidate's résumé and a job description with a **shared encoder** and predicts how well they match — Weak / Medium / Strong — along with a human-readable skill-coverage explanation.

<p>
<img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python">
<img src="https://img.shields.io/badge/TensorFlow-2.15+-orange.svg" alt="TensorFlow">
<img src="https://img.shields.io/badge/Task-Text%20Pair%20Classification-green.svg" alt="Task">
<img src="https://img.shields.io/badge/License-MIT-lightgrey.svg" alt="License">
</p>

**Author:** Shuvam Saren · M.Tech (Computer Science & Data Processing), Department of Mathematics, **IIT Kharagpur**

**Submitted for:** Summer Training and Internship Programme on **Machine Learning & Agentic AI**
**Organised by:** Electronics & ICT Academy, **Indian Institute of Technology Roorkee**

---

## Table of Contents

- [Problem](#problem)
- [Approach](#approach)
- [Architecture](#architecture)
- [Repository Structure](#repository-structure)
- [Quickstart](#quickstart)
- [Dataset](#dataset)
- [Results](#results)
- [Design Decisions](#design-decisions)
- [Limitations](#limitations)
- [Tech Stack](#tech-stack)
- [License](#license)

---

## Problem

Recruiters screen thousands of résumés against a single job description. Keyword filters are brittle — they miss paraphrased skills and reward keyword stuffing. This project treats the task as **semantic text-pair classification**: given `(résumé, job description)`, predict a three-level match score.

| Label | Class | Meaning |
|:-----:|:------|:--------|
| `0` | Weak | Poor fit — little genuine skill overlap |
| `1` | Medium | Partial fit — some required skills present |
| `2` | Strong | Ground-truth pairing from the dataset |

## Approach

A **Siamese network** encodes both documents with the *same* weights, so the résumé and the JD land in a shared representation space and can be compared directly. Similarity is then learned rather than hand-coded.

```
u = encode(résumé)          # shared tower
v = encode(job description) # shared tower
features = [ u , v , |u − v| , u ⊙ v ]  →  Dense head  →  softmax(3)
```

The `|u − v|` term captures **distance** and `u ⊙ v` captures **alignment** — together they give the classifier a far richer signal than plain concatenation.

## Architecture

```
 résumé_text ──┐
               ├─▶ shared TextVectorization (30k vocab, len 320)
 job_desc  ────┘       ↓
                shared Embedding(128, mask_zero=True)
                       ↓
                shared Bidirectional LSTM(64)
                       ↓
                 GlobalMaxPooling1D
                       ↓
        u , v ─▶ concat[ u, v, |u−v|, u⊙v ]  (512-d)
                       ↓
              Dense(128, relu) + L2 → Dropout(0.3)
                       ↓
              Dense(64, relu) + L2
                       ↓
              Dense(3, softmax)  →  Weak / Medium / Strong
```

**Regularization:** weight sharing, embedding masking, dropout, light L2 on the dense head, and early stopping on validation loss.

## Repository Structure

```
resume-jd-match-scorer/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── src/                                  # importable pipeline
│   ├── preprocessing.py                  # JSONL loading + 3-class pair construction
│   ├── model.py                          # Siamese BiLSTM definition
│   ├── train.py                          # end-to-end training driver
│   ├── evaluate.py                       # macro-F1, per-class report, confusion matrix
│   ├── predict.py                        # single résumé–JD pair inference
│   ├── explain.py                        # data-driven skill-coverage explanation
│   └── skills_vocab.json                 # skill bank derived from the dataset
│
├── notebooks/
│   └── resume_jd_siamese_bilstm.ipynb    # self-contained, Colab-ready walkthrough
│
├── data/
│   ├── README.md                         # how to obtain the full dataset
│   └── train_sample.jsonl                # 800-record sample for a quick run
│
├── docs/
│   └── Project_Report_Resume_JD_Deep_Neural_Network.docx
│
├── outputs/                              # generated: curves, reports, plots
└── models/                               # generated: saved Keras model
```

## Quickstart

```bash
git clone https://github.com/s-square7/resume-jd-match-scorer.git
cd resume-jd-match-scorer

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# smoke run on the bundled 800-record sample
python src/train.py
```

To train on the full corpus, drop `train.jsonl` into `data/` — the driver prefers it over the sample automatically.

Predict on a single pair:

```python
from src.predict import predict_match
predict_match(model, resume_text, jd_text)
```

Artefacts land in `outputs/` (`training_history.png`, `classification_report.txt`, `confusion_matrix.png`, `prediction_result.md`, `class_mapping.json`) and the trained model in `models/`.

## Dataset

JSONL records with `Job-Description`, `Resume`, and a ground-truth `Skills` list spanning **10,000+ skills across many non-technical domains** (healthcare, trades, administration — not just software). See [`data/README.md`](data/README.md).

Label construction (in `preprocessing.py`):

- **Strong** — the dataset's own résumé↔JD pairing.
- **Medium** — a partially-overlapping donor résumé.
- **Weak** — a donor chosen for **low ground-truth skill overlap** (Jaccard < 0.10) with the JD.

A sanity check confirms the classes are semantically ordered by mean JD-skill coverage in the résumé:

| Class | Mean skill coverage |
|:------|:-------------------:|
| Strong | ≈ 0.96 |
| Medium | ≈ 0.70 |
| Weak | ≈ 0.09 |

## Results

Headline metric is **macro-F1**, not accuracy. Weak is easy to separate while Medium and Strong are near-identical text, so plain accuracy is inflated and hides the interesting failure mode.

Every run writes the full per-class report and confusion matrix to `outputs/` so the Medium↔Strong confusion stays visible.

> Fill this table in from your own `outputs/classification_report.txt` after a full run:
>
> | Metric | Value |
> |:-------|:-----:|
> | Accuracy | — |
> | **Macro-F1** | — |
> | F1 (Weak) | — |
> | F1 (Medium) | — |
> | F1 (Strong) | — |

## Design Decisions

Five choices where this departs from a textbook baseline:

1. **An honest Weak class.** The naïve recipe labels a *randomly shuffled* résumé as Weak. That is trivially separable and occasionally mislabels a genuine match. Selecting the donor by low skill overlap makes a Weak label mean *poor fit*.
2. **Macro-F1 over accuracy.** Class difficulty is deeply uneven; macro-F1 refuses to let the easy class carry the score.
3. **Domain-general explanations.** A hardcoded ~30-item tech-skill bank returned "None found" for most records in this multi-domain corpus. The explainer now uses each record's ground-truth `Skills` with inflection-tolerant matching (*management → managing*) and falls back to a dataset-derived bank for free text.
4. **`tf.string` inference fix.** Single-pair prediction previously passed a fixed-width NumPy string array (`<U…`) that the Keras string input layer rejects; it now uses `tf.constant` tensors.
5. **Parity between notebook and package.** The Colab notebook and `src/` implement the same pipeline, so results reproduce either way.

## Limitations

The skill-coverage explainer is a **lexical aid, not the classifier**. Heavily paraphrased soft skills (*"problem-solving skills"* → *"solving sudden problems"*) may still be reported as missing. Semantic match quality is decided by the neural model; the skill list is human-readable feedback layered on top.

Further work: swap the BiLSTM towers for a pretrained sentence encoder, add cross-attention between the two towers, and calibrate the softmax scores before using them as a ranking.

## Tech Stack

`Python` · `TensorFlow / Keras` · `scikit-learn` · `pandas` · `NumPy` · `Matplotlib` · `Seaborn`

## License

Released under the [MIT License](LICENSE).

---

<sub>Developed as part of the Summer Training and Internship Programme on Machine Learning & Agentic AI, Electronics & ICT Academy, IIT Roorkee.</sub>

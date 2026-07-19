"""
Preprocessing and label construction for the Resume-JD Match Scorer.

Author : Shuvam Saren (25MA60R16)
         M.Tech, Computer Science & Data Processing
         Department of Mathematics, IIT Kharagpur

The source dataset provides, per record, one job description together with a
matched resume and a deliberately weakened ("unmatched") resume, plus a curated
ground-truth `Skills` list. The blueprint asks for a THREE-class scorer
(Weak / Medium / Strong), so we must synthesise a third class.

Design of the three classes (this is the part most worth scrutinising):

    Strong (2) : JD + Resume-matched      -> the true positive.
    Medium (1) : JD + Resume-unmatched    -> HARD negative. Same candidate, a few
                                             skills or a date removed. Text is
                                             near-identical to the Strong resume,
                                             so telling Medium from Strong is the
                                             genuinely difficult decision.
    Weak   (0) : JD + a donor resume from  -> EASY negative, but made HONEST: the
                 a different record            donor is chosen to have LOW skill
                                             overlap with this JD (see below),
                                             so a Weak label really does mean a
                                             poor match, not just a random pairing.

The original sample simply shuffled resumes for the Weak class. That makes Weak
trivially separable and can occasionally pair a JD with a same-domain resume,
which mislabels a genuine match as Weak. We fix both issues by rejecting donors
whose ground-truth skills overlap the JD's skills above a threshold.

Because Medium and Strong are near-identical text, ACCURACY ALONE IS MISLEADING:
the easy Weak class inflates it. Always read the per-class precision/recall/F1
and the confusion matrix (evaluate.py reports both, plus macro-F1).
"""

import json
import random
import re

import pandas as pd


LABEL_NAMES = {0: "Weak Match", 1: "Medium Match", 2: "Strong Match"}

# Maximum Jaccard overlap between a donor resume's JD-skills and the target JD's
# skills for the donor to still count as a valid Weak (poor-match) pairing.
WEAK_MAX_SKILL_OVERLAP = 0.10
# How many random donors to try before giving up and accepting the last one.
WEAK_MAX_TRIES = 12


def clean_text(text):
    """Collapse whitespace so the vectorizer sees clean, single-spaced text."""
    text = str(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def read_jsonl(path):
    """Read a JSON-Lines file into a list of dicts."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _skill_set(record):
    """Ground-truth skills for a record, lower-cased and de-duplicated."""
    skills = record.get("Skills", []) or []
    return {str(s).strip().lower() for s in skills if str(s).strip()}


def _jaccard(a, b):
    if not a and not b:
        return 0.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def build_labeled_pairs(source_records, row_limit=None, seed=42):
    """
    Turn the raw records into a supervised DataFrame with three balanced classes.

    Returns columns:
        resume_text, job_description, match_label, label_name,
        jd_skills (ground-truth JD skills as a list, used by the explainer/eval)
    """
    if row_limit is not None:
        source_records = source_records[:row_limit]

    rng = random.Random(seed)
    n = len(source_records)

    jd_texts = [clean_text(r["Job-Description"]) for r in source_records]
    matched = [clean_text(r["Resume-matched"]) for r in source_records]
    unmatched = [clean_text(r["Resume-unmatched"]) for r in source_records]
    skill_sets = [_skill_set(r) for r in source_records]

    def pick_weak_donor(target_idx):
        """Pick a resume from another record with low skill overlap with this JD."""
        target_skills = skill_sets[target_idx]
        best_idx = None
        best_overlap = 1.0
        for _ in range(WEAK_MAX_TRIES):
            j = rng.randrange(n)
            if j == target_idx:
                continue
            overlap = _jaccard(target_skills, skill_sets[j])
            if overlap < best_overlap:
                best_overlap, best_idx = overlap, j
            if overlap <= WEAK_MAX_SKILL_OVERLAP:
                return j
        # Fall back to the lowest-overlap donor we saw (still better than random).
        return best_idx if best_idx is not None else (target_idx + 1) % n

    rows = []
    for idx in range(n):
        jd = jd_texts[idx]
        jd_skills = sorted(skill_sets[idx])
        weak_idx = pick_weak_donor(idx)

        rows.append({
            "resume_text": matched[idx],
            "job_description": jd,
            "match_label": 2,
            "label_name": LABEL_NAMES[2],
            "jd_skills": jd_skills,
        })
        rows.append({
            "resume_text": unmatched[idx],
            "job_description": jd,
            "match_label": 1,
            "label_name": LABEL_NAMES[1],
            "jd_skills": jd_skills,
        })
        rows.append({
            "resume_text": matched[weak_idx],
            "job_description": jd,
            "match_label": 0,
            "label_name": LABEL_NAMES[0],
            "jd_skills": jd_skills,
        })

    return pd.DataFrame(rows)

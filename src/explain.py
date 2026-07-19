"""
Explanation layer: which required skills a resume covers and which it misses.

Author : Shuvam Saren (25MA60R16), IIT Kharagpur

The original sample matched text against a hardcoded list of ~30 tech skills.
That dataset, however, spans 10,000+ skills across many non-tech domains
(supply chain, nursing, mechanical, finance, ...), so a tech-only bank returned
"None found" for most records.

This version is DATA-DRIVEN:
  * When the ground-truth JD skills are known (they are, for every dataset
    record), we use them directly -- exact and domain-correct.
  * For arbitrary free text we fall back to a bank built from the dataset's own
    `Skills` vocabulary (skills_vocab.json), matched with word boundaries so
    that "java" does not spuriously fire inside "javascript".
"""

import json
import re
from pathlib import Path

_VOCAB_PATH = Path(__file__).with_name("skills_vocab.json")


def _load_skill_bank():
    try:
        with open(_VOCAB_PATH, "r", encoding="utf-8") as f:
            bank = json.load(f)
    except FileNotFoundError:
        # Minimal safe fallback if the vocab file is missing.
        bank = ["python", "sql", "java", "communication skills", "project management"]
    # Longer phrases first so multi-word skills are preferred when scanning.
    return sorted({str(s).strip().lower() for s in bank if str(s).strip()},
                  key=lambda s: (-len(s), s))


SKILL_BANK = _load_skill_bank()
# Pre-compile one alternation regex with word boundaries for fast scanning.
if SKILL_BANK:
    _pattern = re.compile(
        r"(?<![a-z0-9])(" + "|".join(re.escape(s) for s in SKILL_BANK) + r")(?![a-z0-9])"
    )
else:
    _pattern = None


def extract_skills(text):
    """Return the set of bank skills that appear (word-bounded) in the text."""
    if _pattern is None:
        return set()
    return set(m.group(1) for m in _pattern.finditer(str(text).lower()))


# Generic words that carry no discriminative signal inside a skill phrase.
_STOPWORDS = {
    "and", "or", "to", "of", "the", "a", "an", "in", "on", "with", "for",
    "ability", "skills", "skill", "knowledge", "understanding", "proficiency",
    "experience", "strong", "good", "excellent", "using", "use",
}
_PREFIX = 4  # characters used for inflection-tolerant prefix matching


def _content_tokens(skill):
    toks = re.findall(r"[a-z0-9]+", skill.lower())
    return [t for t in toks if t not in _STOPWORDS]


def _token_covered(token, resume_tokens):
    """A single skill token is covered by the resume (inflection-tolerant)."""
    if len(token) < _PREFIX:
        # Short tokens (sql, aws, db2, java) must match as whole words.
        return token in resume_tokens
    head = token[:_PREFIX]
    for rt in resume_tokens:
        # Bidirectional prefix so 'analysis' matches 'analyzing', etc.
        if rt.startswith(head) or token.startswith(rt[:_PREFIX]):
            return True
    return False


def _skill_covered(skill, resume_lower, resume_tokens):
    # Exact word-bounded phrase match is the strongest signal.
    if re.search(r"(?<![a-z0-9])" + re.escape(skill) + r"(?![a-z0-9])", resume_lower):
        return True
    tokens = _content_tokens(skill)
    if not tokens:
        return False
    # Require EVERY content token to be present (conservative -> few false hits).
    return all(_token_covered(t, resume_tokens) for t in tokens)


def compare_skills(resume_text, job_description, jd_skills=None):
    """
    Compute common and missing skills for a resume/JD pair.

    If `jd_skills` (ground-truth list) is provided, it is treated as the
    authoritative set of required skills and matched against the resume with
    inflection-tolerant logic; otherwise skills are extracted from the
    job-description text using the data-driven bank.

    Note: this is a lexical aid. Heavily paraphrased soft skills may still be
    reported as missing -- the neural model, not this matcher, judges semantic
    match quality. The list is for human-readable feedback only.
    """
    if jd_skills:
        required = {str(s).strip().lower() for s in jd_skills if str(s).strip()}
        resume_lower = str(resume_text).lower()
        resume_tokens = set(re.findall(r"[a-z0-9]+", resume_lower))
        covered, missing = set(), set()
        for skill in required:
            (covered if _skill_covered(skill, resume_lower, resume_tokens)
             else missing).add(skill)
        return {"common_skills": sorted(covered), "missing_skills": sorted(missing)}

    resume_skills = extract_skills(resume_text)
    jd_skills_extracted = extract_skills(job_description)
    return {
        "common_skills": sorted(resume_skills & jd_skills_extracted),
        "missing_skills": sorted(jd_skills_extracted - resume_skills),
    }

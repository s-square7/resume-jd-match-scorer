"""
Single-pair prediction + human-readable explanation.

Author : Shuvam Saren (25MA60R16), IIT Kharagpur
"""

import numpy as np
import tensorflow as tf

from explain import compare_skills


LABEL_NAMES = {0: "Weak Match", 1: "Medium Match", 2: "Strong Match"}


def predict_match(model, resume_text, job_description, jd_skills=None):
    """Predict the match class for one resume/JD pair and explain the skills.

    Pass `jd_skills` (the record's ground-truth Skills list) when available for
    precise, domain-correct common/missing-skill feedback.
    """
    # Use tf.string constants -- a numpy string array gives a fixed-width
    # dtype (e.g. '<U21664') that the Keras string input layer rejects.
    probabilities = model.predict(
        (tf.constant([str(resume_text)], dtype=tf.string),
         tf.constant([str(job_description)], dtype=tf.string)),
        verbose=0,
    )[0]
    predicted_class = int(np.argmax(probabilities))
    skill_info = compare_skills(resume_text, job_description, jd_skills=jd_skills)
    return {
        "predicted_class": predicted_class,
        "predicted_label": LABEL_NAMES[predicted_class],
        "confidence": float(probabilities[predicted_class]),
        "probabilities": {LABEL_NAMES[i]: float(probabilities[i]) for i in range(3)},
        **skill_info,
    }

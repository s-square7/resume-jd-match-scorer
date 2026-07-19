r"""
Siamese BiLSTM match-scorer model.

Author : Shuvam Saren (25MA60R16), IIT Kharagpur

Architecture (matches the project blueprint):

    resume_text --\                                         /-- Dense(128,relu)
                   >-- shared TextVectorization             |   Dropout(0.3)
    job_desc   --/    -> shared Embedding(mask_zero)        |-- Dense(64,relu)
                      -> shared BiLSTM(64)                  |-- Dense(3,softmax)
                      -> GlobalMaxPooling1D                 /
    u = encode(resume), v = encode(jd)
    features = concat[u, v, |u-v|, u*v]

The two towers SHARE weights, so a resume and a JD are read by the same encoder
and land in the same representation space -- the core idea of a Siamese network.
The comparison block |u-v| (distance) and u*v (alignment) gives the classifier a
richer signal than plain concatenation.
"""

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Bidirectional,
    Concatenate,
    Dense,
    Dropout,
    Embedding,
    GlobalMaxPooling1D,
    Input,
    LSTM,
    Lambda,
    Multiply,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


def build_siamese_bilstm_model(vectorizer, vocab_size, embed_dim=128,
                               lstm_units=64, dropout_rate=0.3,
                               l2_reg=1e-5, learning_rate=1e-3):
    """Build and compile the Siamese BiLSTM classifier.

    `l2_reg` adds light L2 weight decay on the dense head (regularization
    section of the blueprint); set to 0.0 to disable.
    """
    resume_input = Input(shape=(), dtype=tf.string, name="resume_text")
    jd_input = Input(shape=(), dtype=tf.string, name="job_description")

    # Shared layers -> identical processing for both towers (weight sharing).
    embedding = Embedding(vocab_size, embed_dim, mask_zero=True,
                          name="shared_embedding")
    bilstm = Bidirectional(LSTM(lstm_units, return_sequences=True),
                           name="shared_bilstm")
    pool = GlobalMaxPooling1D(name="global_max_pool")

    def encode(text_input):
        x = vectorizer(text_input)
        x = embedding(x)
        x = bilstm(x)
        return pool(x)

    u = encode(resume_input)
    v = encode(jd_input)

    abs_diff = Lambda(lambda t: tf.abs(t[0] - t[1]), name="abs_difference")([u, v])
    product = Multiply(name="elementwise_product")([u, v])
    features = Concatenate(name="comparison_features")([u, v, abs_diff, product])

    reg = l2(l2_reg) if l2_reg else None
    x = Dense(128, activation="relu", kernel_regularizer=reg)(features)
    x = Dropout(dropout_rate)(x)
    x = Dense(64, activation="relu", kernel_regularizer=reg)(x)
    output = Dense(3, activation="softmax", name="match_class")(x)

    model = Model([resume_input, jd_input], output)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

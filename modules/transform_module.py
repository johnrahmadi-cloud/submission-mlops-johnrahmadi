"""
transform_module.py

Modul preprocessing untuk pipeline TFX pada dataset Telco Customer Churn.
Digunakan oleh komponen Transform.
"""

import tensorflow as tf
import tensorflow_transform as tft

# Fitur kategorikal (string) yang akan di-encode menjadi index integer
CATEGORICAL_FEATURES = {
    "gender": 2,
    "Partner": 2,
    "Dependents": 2,
    "PhoneService": 2,
    "MultipleLines": 3,
    "InternetService": 3,
    "OnlineSecurity": 3,
    "OnlineBackup": 3,
    "DeviceProtection": 3,
    "TechSupport": 3,
    "StreamingTV": 3,
    "StreamingMovies": 3,
    "Contract": 3,
    "PaperlessBilling": 2,
    "PaymentMethod": 4,
}

# SeniorCitizen sudah berupa 0/1 di dataset asli, tetap diperlakukan sebagai kategorikal
CATEGORICAL_INT_FEATURES = {
    "SeniorCitizen": 2,
}

# Fitur numerik yang akan dinormalisasi
NUMERICAL_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

LABEL_KEY = "Churn"


def transformed_name(key):
    """Menambahkan suffix '_xf' pada nama fitur hasil transformasi."""
    return key + "_xf"


def fill_missing(x):
    """Mengganti nilai kosong pada SparseTensor dengan default value, lalu densify."""
    if isinstance(x, tf.sparse.SparseTensor):
        default_value = "" if x.dtype == tf.string else 0
        x = tf.sparse.to_dense(
            tf.SparseTensor(x.indices, x.values, [x.dense_shape[0], 1]),
            default_value,
        )
    return tf.squeeze(x, axis=1)


def convert_num_to_one_hot(label_tensor, num_labels=2):
    """Konversi label index menjadi representasi one-hot."""
    one_hot_tensor = tf.one_hot(label_tensor, num_labels)
    return tf.reshape(one_hot_tensor, [-1, num_labels])


def preprocessing_fn(inputs):
    """
    Fungsi preprocessing utama yang dipanggil oleh komponen Transform TFX.

    Args:
        inputs: dictionary berisi fitur mentah (raw features) dari ExampleGen.

    Returns:
        outputs: dictionary berisi fitur yang sudah ditransformasi.
    """
    outputs = {}

    # Normalisasi fitur numerik menjadi z-score
    for feature in NUMERICAL_FEATURES:
        outputs[transformed_name(feature)] = tft.scale_to_z_score(
            fill_missing(inputs[feature])
        )

    # Encode fitur kategorikal string menjadi index integer, lalu one-hot
    for feature, num_labels in CATEGORICAL_FEATURES.items():
        int_value = tft.compute_and_apply_vocabulary(
            fill_missing(inputs[feature]),
            top_k=num_labels,
        )
        outputs[transformed_name(feature)] = convert_num_to_one_hot(
            int_value, num_labels=num_labels
        )

    # Fitur yang sudah berbentuk integer (SeniorCitizen), tetap di-one-hot
    for feature, num_labels in CATEGORICAL_INT_FEATURES.items():
        outputs[transformed_name(feature)] = convert_num_to_one_hot(
            tf.cast(fill_missing(inputs[feature]), tf.int64),
            num_labels=num_labels,
        )

    # Label: sudah berupa angka 0/1 sejak proses cleaning data awal,
    # jadi cukup di-cast, tidak perlu vocabulary lookup.
    outputs[transformed_name(LABEL_KEY)] = tf.cast(
        fill_missing(inputs[LABEL_KEY]), tf.int64
    )

    return outputs

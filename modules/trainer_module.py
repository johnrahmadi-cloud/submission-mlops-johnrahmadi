"""
trainer_module.py

Modul untuk melatih model klasifikasi biner (Churn) menggunakan Keras,
dipanggil oleh komponen Trainer TFX.
"""

import tensorflow as tf
import tensorflow_transform as tft
from tensorflow.keras import layers
from tfx.components.trainer.fn_args_utils import FnArgs

from modules.transform_module import (
    CATEGORICAL_FEATURES,
    CATEGORICAL_INT_FEATURES,
    NUMERICAL_FEATURES,
    LABEL_KEY,
    transformed_name,
)

LEARNING_RATE = 1e-3
BATCH_SIZE = 64
NUM_EPOCHS = 10


def gzip_reader_fn(filenames):
    """Membaca TFRecord terkompresi GZIP hasil output Transform."""
    return tf.data.TFRecordDataset(filenames, compression_type="GZIP")


def input_fn(file_pattern, tf_transform_output, batch_size=BATCH_SIZE):
    """Membuat tf.data.Dataset dari hasil Transform untuk training/eval."""
    transformed_feature_spec = tf_transform_output.transformed_feature_spec().copy()

    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transformed_feature_spec,
        reader=gzip_reader_fn,
        label_key=transformed_name(LABEL_KEY),
    )
    return dataset


def _get_model(tf_transform_output):
    """Membangun arsitektur model Keras (functional API)."""
    input_features = []

    for key in NUMERICAL_FEATURES:
        input_features.append(
            tf.keras.Input(shape=(1,), name=transformed_name(key))
        )

    for key, dim in {**CATEGORICAL_FEATURES, **CATEGORICAL_INT_FEATURES}.items():
        input_features.append(
            tf.keras.Input(shape=(dim,), name=transformed_name(key))
        )

    concatenate = layers.concatenate(input_features)
    deep = layers.Dense(256, activation="relu")(concatenate)
    deep = layers.Dropout(0.3)(deep)
    deep = layers.Dense(128, activation="relu")(deep)
    deep = layers.Dropout(0.3)(deep)
    deep = layers.Dense(64, activation="relu")(deep)
    outputs = layers.Dense(1, activation="sigmoid")(deep)

    model = tf.keras.Model(inputs=input_features, outputs=outputs)
    model.compile(
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(),
            tf.keras.metrics.AUC(),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
        ],
    )
    model.summary()
    return model


def _get_serve_tf_examples_fn(model, tf_transform_output):
    """Membuat signature serving_default agar model bisa menerima raw tf.Example."""
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(LABEL_KEY, None)
        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
        transformed_features = model.tft_layer(parsed_features)
        return model(transformed_features)

    return serve_tf_examples_fn


def run_fn(fn_args: FnArgs):
    """Entry point yang dipanggil oleh komponen Trainer TFX."""
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    train_dataset = input_fn(fn_args.train_files, tf_transform_output, BATCH_SIZE)
    eval_dataset = input_fn(fn_args.eval_files, tf_transform_output, BATCH_SIZE)

    model = _get_model(tf_transform_output)

    log_dir = fn_args.model_run_dir
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, update_freq="batch"
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_binary_accuracy", mode="max", patience=3
    )

    model.fit(
        train_dataset,
        validation_data=eval_dataset,
        steps_per_epoch=fn_args.train_steps,
        validation_steps=fn_args.eval_steps,
        epochs=NUM_EPOCHS,
        callbacks=[tensorboard_callback, early_stopping],
    )

    signatures = {
        "serving_default": _get_serve_tf_examples_fn(
            model, tf_transform_output
        ).get_concrete_function(
            tf.TensorSpec(shape=[None], dtype=tf.string, name="examples")
        ),
    }

    tf.saved_model.save(
        model, fn_args.serving_model_dir, signatures=signatures
    )

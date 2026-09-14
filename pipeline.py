"""
pipeline.py

Mendefinisikan TFX pipeline untuk proyek klasifikasi Telco Customer Churn.
Dijalankan menggunakan Apache Beam sebagai orchestrator (BeamDagRunner).
"""

import os
from tfx.components import (
    CsvExampleGen,
    StatisticsGen,
    SchemaGen,
    ExampleValidator,
    Transform,
    Trainer,
    Evaluator,
    Pusher,
)
from tfx.components.trainer.executor import GenericExecutor
from tfx.dsl.components.base import executor_spec
from tfx.dsl.components.common.resolver import Resolver
from tfx.dsl.experimental.latest_blessed_model_resolver import (
    LatestBlessedModelResolver,
)
from tfx.orchestration import metadata, pipeline
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner
from tfx.proto import example_gen_pb2, trainer_pb2, pusher_pb2
from tfx.types import Channel
from tfx.types.standard_artifacts import Model, ModelBlessing
import tensorflow_model_analysis as tfma

PIPELINE_NAME = "johnrahmadi-pipeline"

# Path-path penting
DATA_ROOT = "data"
TRANSFORM_MODULE_FILE = "modules/transform_module.py"
TRAINER_MODULE_FILE = "modules/trainer_module.py"
OUTPUT_BASE = os.path.join("johnrahmadi-pipeline")
SERVING_MODEL_DIR = os.path.join("serving_model", PIPELINE_NAME)

PIPELINE_ROOT = os.path.join(OUTPUT_BASE, PIPELINE_NAME)
METADATA_PATH = os.path.join(OUTPUT_BASE, "metadata", PIPELINE_NAME, "metadata.db")


def init_components(
    data_root,
    transform_module,
    trainer_module,
    serving_model_dir,
    train_steps=1000,
    eval_steps=500,
):
    """Menginisialisasi seluruh komponen pipeline TFX."""

    # 1. ExampleGen: split data train (80%) dan eval (20%)
    output_config = example_gen_pb2.Output(
        split_config=example_gen_pb2.SplitConfig(
            splits=[
                example_gen_pb2.SplitConfig.Split(name="train", hash_buckets=8),
                example_gen_pb2.SplitConfig.Split(name="eval", hash_buckets=2),
            ]
        )
    )
    example_gen = CsvExampleGen(input_base=data_root, output_config=output_config)

    # 2. StatisticsGen: menghitung statistik deskriptif dataset
    statistics_gen = StatisticsGen(examples=example_gen.outputs["examples"])

    # 3. SchemaGen: membuat schema berdasarkan statistik
    schema_gen = SchemaGen(
        statistics=statistics_gen.outputs["statistics"], infer_feature_shape=True
    )

    # 4. ExampleValidator: mendeteksi anomali pada data
    example_validator = ExampleValidator(
        statistics=statistics_gen.outputs["statistics"],
        schema=schema_gen.outputs["schema"],
    )

    # 5. Transform: preprocessing fitur
    transform = Transform(
        examples=example_gen.outputs["examples"],
        schema=schema_gen.outputs["schema"],
        module_file=transform_module,
    )

    # 6. Trainer: melatih model
    trainer = Trainer(
        module_file=trainer_module,
        custom_executor_spec=executor_spec.ExecutorClassSpec(GenericExecutor),
        examples=transform.outputs["transformed_examples"],
        transform_graph=transform.outputs["transform_graph"],
        schema=schema_gen.outputs["schema"],
        train_args=trainer_pb2.TrainArgs(num_steps=train_steps),
        eval_args=trainer_pb2.EvalArgs(num_steps=eval_steps),
    )

    # 7. Resolver: mengambil model blessed terakhir sebagai baseline evaluasi
    model_resolver = Resolver(
        strategy_class=LatestBlessedModelResolver,
        model=Channel(type=Model),
        model_blessing=Channel(type=ModelBlessing),
    ).with_id("Latest_blessed_model_resolver")

    # 8. Evaluator: mengevaluasi performa model terhadap baseline
    eval_config = tfma.EvalConfig(
        model_specs=[tfma.ModelSpec(label_key="Churn")],
        slicing_specs=[
            tfma.SlicingSpec(),
            tfma.SlicingSpec(feature_keys=["Contract"]),
        ],
        metrics_specs=[
            tfma.MetricsSpec(
                metrics=[
                    tfma.MetricConfig(class_name="AUC"),
                    tfma.MetricConfig(class_name="Precision"),
                    tfma.MetricConfig(class_name="Recall"),
                    tfma.MetricConfig(
                        class_name="BinaryAccuracy",
                        threshold=tfma.MetricThreshold(
                            value_threshold=tfma.GenericValueThreshold(
                                lower_bound={"value": 0.6}
                            ),
                            change_threshold=tfma.GenericChangeThreshold(
                                direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                                absolute={"value": -1e-3},
                            ),
                        ),
                    ),
                ]
            )
        ],
    )

    evaluator = Evaluator(
        examples=example_gen.outputs["examples"],
        model=trainer.outputs["model"],
        baseline_model=model_resolver.outputs["model"],
        eval_config=eval_config,
    )

    # 9. Pusher: mendorong model yang lolos evaluasi ke direktori serving
    pusher = Pusher(
        model=trainer.outputs["model"],
        model_blessing=evaluator.outputs["blessing"],
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory=serving_model_dir
            )
        ),
    )

    return (
        example_gen,
        statistics_gen,
        schema_gen,
        example_validator,
        transform,
        trainer,
        model_resolver,
        evaluator,
        pusher,
    )


def init_pipeline(components, pipeline_root, metadata_path):
    """Membungkus komponen ke dalam objek TFX Pipeline."""
    beam_args = [
        "--direct_running_mode=multi_processing",
        "--direct_num_workers=0",
    ]

    return pipeline.Pipeline(
        pipeline_name=PIPELINE_NAME,
        pipeline_root=pipeline_root,
        components=components,
        enable_cache=True,
        metadata_connection_config=metadata.sqlite_metadata_connection_config(
            metadata_path
        ),
        beam_pipeline_args=beam_args,
    )


if __name__ == "__main__":
    components = init_components(
        data_root=DATA_ROOT,
        transform_module=TRANSFORM_MODULE_FILE,
        trainer_module=TRAINER_MODULE_FILE,
        serving_model_dir=SERVING_MODEL_DIR,
        train_steps=1000,
        eval_steps=500,
    )

    tfx_pipeline = init_pipeline(components, PIPELINE_ROOT, METADATA_PATH)
    BeamDagRunner().run(pipeline=tfx_pipeline)

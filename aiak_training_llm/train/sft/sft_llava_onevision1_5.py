from megatron.core.enums import ModelType

from aiak_training_llm.train.megatron_trainer import MegatronTrainer
from aiak_training_llm.train.trainer_builder import register_model_trainer
from aiak_training_llm.utils.constants import TrainingPhase, VisionLanguageModelFamilies


@register_model_trainer(
    model_family=[VisionLanguageModelFamilies.LLAVA_ONEVISION1_5], training_phase=TrainingPhase.SFT
)
def default_pretrain_trainer(train_args):
    """build trainer"""
    from aiak_training_llm.train.pretrain import pretrain_llava_onevision1_5

    if train_args.encoder_pipeline_model_parallel_size in [0, None]:
        model_type = ModelType.encoder_or_decoder
    else:
        model_type = ModelType.encoder_and_decoder
    trainer = MegatronTrainer(
        train_args=train_args,
        train_valid_test_dataset_provider=pretrain_llava_onevision1_5.train_valid_test_dataset_provider,
        model_provider=pretrain_llava_onevision1_5.model_provider,
        model_type=model_type,
        forward_step_func=pretrain_llava_onevision1_5.forward_step,
    )

    return trainer

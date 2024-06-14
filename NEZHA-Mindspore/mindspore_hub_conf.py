

"""Bert hub interface for bert base"""

from src.tinybert_model import BertModel
from src.tinybert_model import BertConfig
import mindspore.common.dtype as mstype

tinybert_student_net_cfg = BertConfig(
    seq_length=128,
    vocab_size=30522,
    hidden_size=768,
    num_hidden_layers=6,
    num_attention_heads=12,
    intermediate_size=3072,
    hidden_act="gelu",
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    max_position_embeddings=512,
    type_vocab_size=2,
    initializer_range=0.02,
    use_relative_positions=False,
    dtype=mstype.float32,
    compute_type=mstype.float32,
    do_quant=True,
    embedding_bits=2,
    weight_bits=2,
    weight_clip_value=3.0,
    cls_dropout_prob=0.1,
    activation_init=2.5,
    is_lgt_fit=False
)


def create_network(name, *args, **kwargs):
    """
    Create tinybert network.
    """
    if name == "ternarybert":
        if "seq_length" in kwargs:
            tinybert_student_net_cfg.seq_length = kwargs["seq_length"]
        is_training = kwargs.get("is_training", False)
        return BertModel(tinybert_student_net_cfg, is_training, *args)
    raise NotImplementedError(f"{name} is not implemented in the repo")

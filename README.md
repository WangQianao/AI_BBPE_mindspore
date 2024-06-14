# 论文相关内容

## . 论文介绍
    - 背景介绍

    `预训练语言模型通过在大规模语料库上进行预训练，能够捕捉文本中的深层语境信息，因此在各种自然语言理解 (NLU) 任务中取得了巨大成功。采用 NEZHA  的架构作为底层预训练语言模型，结果表明，使用字节级子词训练的 NEZHA 在多个多语言 NLU 任务中的表现始终优于 Google 多语言 BERT 和 vanilla NEZHA。`
    - 论文方法

    `使用 BBPE：字节级 BPE（即字节对编码）训练多语言预训练语言模型的实践。BBPE 已被 GPT-2/3 和 Roberta等预训练语言模型采用
`

## . 数据集介绍
    - XNLI 数据集
        - `https://github.com/facebookresearch/XNLI`
    - 具体数据集
        - `https://dl.fbaipublicfiles.com/XNLI/XNLI-1.0.zip`
    - `包含“跨语言自然语言推理”（XNLI）数据。XNLI 包含 MNLI 任务的开发和测试集，包含 15 种语言：ar：阿拉伯语 bg：保加利亚语 de：德语 el：希腊语 en：英语 es：西班牙语 fr：法语 hi：印地语 ru：俄语 sw：斯瓦希里语 th：泰语 tr：土耳其语 ur：乌尔都语 vi：越南语 zh：简体中文`

## . Pipeline
    - 本作业将基于论文官方代码仓库实现，将 tensorflow 版本的网络模型转换成 mindspore 版本的模型。

# TensorFlow 实现版本
    - Tesla P100-PCIE-16GB

##  安装依赖包
        - `Package                Version`
        - `---`
        - `absl-py                1.3.0`
        - `APScheduler            3.9.1`
        - `arrow                  1.2.3`
        - `astor                  0.8.1`
        - `async-timeout          4.0.2`
        - `attrs                  22.1.0`
        - `backcall               0.2.0`
        - `backports.zoneinfo     0.2.1`
        - `bayesian-optimization  1.0.1`
        - `binaryornot            0.4.4`
        - `boto3                  1.4.4`
        - `botocore               1.5.95`
        - `certifi                2022.9.24`
        - `cffi                   1.15.1.post20240308173724`
        - `chardet                5.2.0`
        - `charset-normalizer     2.0.12`
        - `click                  8.1.3`
        - `cloudpickle            2.2.0`
        - `colorama               0.4.6`
        - `configparser           5.3.0`
        - `cookiecutter           2.6.0`
        - `cryptography           3.4.7`
        - `cycler                 0.11.0`
        - `Cython                 0.29.21`
        - `debugpy                1.6.3`
        - `decorator              5.1.1`
        - `defusedxml             0.7.1`
        - `Deprecated             1.2.13`
        - `dnspython              2.2.1`
        - `docutils               0.19`
        - `easydict               1.9`
        - `entrypoints            0.4`
        - `ephemeral-port-reserve 1.1.4`
        - `esdk-obs-python        3.21.4`
        - `exceptiongroup         1.0.4`
        - `filelock               3.0.12`
        - `fonttools              4.38.0`
        - `funcsigs               1.0.2`
        - `future                 0.18.2.post20200723173923`
        - `gast                   0.5.3`
        - `grpcio                 1.50.0`
        - `h5py                   2.8.0`
        - `huaweicloudsdkcore     3.1.96`
        - `huaweicloudsdkcsms     3.1.96`
        - `hyperopt               0.1.2`
        - `idna                   3.4`
        - `importlib-metadata     5.0.0`
        - `importlib-resources    5.10.0`
        - `iniconfig              1.1.1`
        - `ipykernel              6.7.0`
        - `ipython                7.34.0`
        - `ipython-genutils       0.2.0`
        - `jedi                   0.18.1`
        - `jinja2                 3.1.4`
        - `jmespath               0.10.0`
        - `joblib                 1.2.0`
        - `jsonschema             4.17.0`
        - `jupyter-client         7.4.6`
        - `jupyter-core           4.11.2`
        - `Keras                  2.2.4`
        - `Keras-Applications     1.0.8`
        - `Keras-Preprocessing    1.1.2`
        - `kiwisolver             1.4.5`
        - `lazy-import            0.2.2`
        - `lxml                   4.9.1`
        - `ma-cau                 1.1.7`
        - `ma-cau-adapter         1.1.3`
        - `ma-cli                 1.2.3`
        - `ma-tensorboard         1.0.0`
        - `Markdown               3.4.1`
        - `markdown-it-py         2.2.0`
        - `MarkupSafe             2.1.1`
        - `matplotlib             3.5.2`
        - `matplotlib-inline      0.1.6`
        - `mdurl                  0.1.2`
        - `mock                   4.0.3`
        - `modelarts              1.4.20`
        - `moxing-framework       2.0.1.rc0.ffd1c0c8`
        - `nest-asyncio           1.5.6`
        - `networkx               2.6.3`
        - `numpy                  1.17.0`
        - `opencv-python          4.1.2.30`
        - `opencv-python-headless 4.5.5.64`
        - `packaging              21.3`
        - `pandas                 1.1.5`
        - `parso                  0.8.3`
        - `pathlib2               2.3.7.post1`
        - `pexpect                4.8.0`
        - `pickleshare            0.7.5`
        - `Pillow                 6.2.0`
        - `pip                    21.0.1`
        - `pkgutil-resolve-name   1.3.10`
        - `pluggy                 1.0.0`
        - `prettytable            0.7.2`
        - `prompt-toolkit         3.0.32`
        - `protobuf               3.20.1`
        - `psutil                 5.8.0`
        - `ptyprocess             0.7.0`
        - `pyasn1                 0.5.1`
        - `pycocotools            2.0.0`
        - `pycparser              2.21`
        - `Pygments               2.13.0`
        - `pymongo                4.3.2`
        - `pyparsing              3.0.9`
        - `pyrsistent             0.19.2`
        - `pytest                 7.2.0`
        - `python-dateutil        2.8.2`
        - `python-slugify         8.0.4`
        - `pytz                   2022.6`
        - `PyYAML                 5.1`
        - `pyzmq                  24.0.1`
        - `ray                    0.8.0`
        - `redis                  4.3.4`
        - `requests               2.27.1`
        - `requests-futures       1.0.1`
        - `requests-toolbelt      1.0.0`
        - `rich                   13.7.1`
        - `s3transfer             0.1.13`
        - `scikit-learn           0.22.1`
        - `scipy                  1.2.2`
        - `semantic-version       2.10.0`
        - `setuptools             65.5.1`
        - `simplejson             3.19.2.post20240307201052`
        - `six                    1.16.0`
        - `sklearn                0.0`
        - `sortedcontainers       2.2.2`
        - `statistics             1.0.3.5`
        - `tabulate               0.8.9`
        - `tenacity               8.2.3`
        - `tensorboard            1.13.1`
        - `tensorboardX           2.0`
        - `tensorflow-estimator   1.13.0`
        - `tensorflow-gpu         1.13.1`
        - `termcolor              2.1.0`
        - `terminaltables         3.1.10`
        - `text-unidecode         1.3`
        - `tomli                  2.0.1`
        - `tornado                6.2`
        - `tqdm                   4.64.1`
        - `traitlets              5.5.0`
        - `typing-extensions      4.4.0`
        - `tzlocal                5.1`
        - `urllib3                1.26.12`
        - `watchdog               2.0.0`
        - `wcwidth                0.2.5`
        - `Werkzeug               2.2.2`
        - `wheel                  0.38.4`
        - `wrapt                  1.14.1`
        - `zipp                   3.10.0`

## 下载预训练模型
        - `NEZHA-base-multilingual-11-cased`
    - `使用经过 Byte BPE 标记化的多语言预训练模型 NEZHA-base-multilingual-11-cased。目前该模型涵盖 11 种语言（按字母顺序）：阿拉伯语、德语、英语、西班牙语、法语、意大利语、马来语、波兰语、葡萄牙语、俄语和泰语。如果您想使用我们的多语言预训练 NEZHA 模型，请使用 Byte BPE 中的 tokenizationBBPE.py 作为标记器（即，用此 tokenizationBBPE.py 替换原始 tokenization.py）。`

##  替换 tokenizer
        - 使用 BBPE 替换原始 tokenizer
        - `<BBPE.bbpe.tokenization.FullTokenizer object at 0x7f7b40665710>`

##  开始运行
        - 执行 `run_clf.sh`

        - `CUDA_VISIBLE_DEVICES=1 python../run_classifier.py --task_name=xnli  --do_train=true  --do_eval=true  --do_train_and_eval=true  --do_predict=false --data_dir=./data/xnli/ --save_checkpoints_steps=200 --vocab_file=./nezha/vocab_bpe_xling-10W.txt --bert_config_file=./nezha/bert_base_rel_config_vocab_100503.json --init_checkpoint=./nezha/model.ckpt-571752 --max_seq_length=128 --train_batch_size=32 --eval_batch_size=32 --predict_batch_size=32 --num_train_epochs=2 --output_dir=./output/xnli/`

## 模型参数
  
        - `INFO:tensorflow:**** Trainable Variables ****`
        - `INFO:tensorflow:  name = bert/embeddings/word_embeddings:0, shape = (100503, 768), *INIT_FROM_CKPT*`
        - `INFO:tensorflow:  name = bert/embeddings/token_type_embeddings:0, shape = (2, 768), *INIT_FROM_CKPT*`
        - `INFO:tensorflow:  name = bert/embeddings/LayerNorm/beta:0, shape = (768,), *INIT_FROM_CKPT*`
        - `INFO:tensorflow:  name = bert/embeddings/LayerNorm/gamma:0, shape = (768,), *INIT_FROM_CKPT*`
        - `INFO:tensorflow:  name = bert/encoder/layer_0/attention/self/query/kernel:0, shape = (768, 768), *INIT_FROM_CKPT*`
        - `INFO:tensorflow:  name = bert/encoder/layer_0/attention/self/query/bias:0, shape = (768,), *INIT_FROM_CKPT*`
        - `INFO:tensorflow:  name = bert/encoder/layer_0/attention/self/key/kernel:0, shape = (768, 768), *INIT_FROM_CKPT*`
        - `INFO:tensorflow:  name = bert/encoder/layer_0/attention/self/key/bias:0, shape = (768,), *INIT_FROM_CKPT*`
        - `INFO:tensorflow:  name = bert/encoder/layer_0/attention/self/value/kernel:0, shape = (768, 768), *INIT_FROM_CKPT*`
        - `INFO:tensorflow:  name = bert/encoder/layer_0/attention/self/value/bias:0, shape = (768,), *INIT_FROM_CKPT*`
        - `INFO:tensorflow:  name = bert/encoder/layer_0/attention/output/dense/kernel:0, shape = (768, 768), *INIT_FROM_CKPT*`
        - `INFO:tensorflow:  name = bert/encoder/layer_0/attention/output/dense/bias:0, shape = (768,), *INIT_FROM_CKPT*`
        - `INFO:tensorflow:  name = bert/encoder/layer_0/attention/output/LayerNorm/beta:0, shape = (768,), *INIT_FROM_CKPT*`
        - `INFO:tensorflow:  name = bert/encoder/layer_0/attention/output/LayerNorm/gamma:0, shape = (768,), *INIT_FROM_CKPT*`
        - `INFO:tensorflow:  name = bert/encoder/layer_0/intermediate/dense/kernel:0, shape = (768, 3072), *INIT_FROM_CKPT*`
        - `INFO:tensorflow:  name = bert/encoder/layer_0/intermediate/dense/bias:0, shape = (3072,), *INIT_FROM_CKPT*`
        - `INFO:tensorflow:  name = bert/encoder/layer_0/output/dense/kernel:0, shape = (3072, 768), *INIT_FROM_CKPT*`
        - `INFO:tensorflow:  name = bert/encoder/layer_0/output/dense/bias:0, shape = (768,), *INIT_FROM_CKPT*`
        - `INFO:tensorflow:  name = bert/encoder/layer_0/output/LayerNorm/beta:0, shape = (768,), *INIT_FROM_CKPT*`
        - `INFO:tensorflow:  name = bert/encoder/layer_0/output/LayerNorm/gamma:0, shape = (768,), *INIT_FROM_CKPT*`
        - `INFO:tensorflow:  name = bert/encoder/layer_1/attention/self/query/kernel:0, shape = (768, 768), *INIT_FROM_CKPT*`
        - `INFO:tensorflow:  name = bert/encoder/layer_1/attention/self/query/bias:0, shape = (768,), *INIT_FROM_CKPT*`
        - `INFO:tensorflow:  name
## 模型效果
```
INFO tensorflow: ***** Eval results *****
INFO tensorflow: eval_accuracy = 0.64564927
INFO tensorflow: eval_loss = 0.8978614
INFO tensorflow: global_step = 116
INFO tensorflow: loss = 0.8978608

Process finished with exit code 0

```
# mindspore实现版本
代码仓库：https://github.com/XiShuFan/MAMO_mindspore
## mindspore框架介绍
MindSpore是华为推出的一款人工智能计算框架，主要用于开发AI应用和模型。它的特点如下:
- 框架设计：MindSpore采用静态计算图设计，PyTorch采用动态计算图设计。静态计算图在模型编译时确定计算过程，动态计算图在运行时确定计算过程。静态计算图通常更高效，动态计算图更灵活；
- 设备支持：MindSpore在云端和边缘端都有较好的支持，可以在Ascend、CPU、GPU等硬件上运行；
- 自动微分：MindSpore提供自动微分功能，可以自动求导数，简化模型训练过程；
- 运算符和层：MindSpore提供丰富的神经网络层和运算符，覆盖CNN、RNN、GAN等多种模型；
- 训练和部署：MindSpore提供方便的模型训练和部署功能，支持ONNX、CANN和MindSpore格式的模型导出，可以部署到Ascend、GPU、CPU等硬件；
## 数据集迁移
tensorflow使用.tsv文件保存数据集
## 模型迁移


使用到的部分api
tf.layers.Dense --- >mindspore.nn.Dense


tf.squeeze ---> mindspore.Tensor.squeeze

tf.reshape ---> mindspore.ops.Reshape

tf.one_hot ---> mindspore.ops.OneHot

tf.nn.dropout ---> mindspore.nn.Dropout

tf.matmul ---> mindspore.ops.operations.MatMul

tf.saturate_cast--->mindspore.ops.operations.Cast

tf.nn.softmax ---> mindspore.nn.Softmax

tf.tanh ---> mindspore.ops.operations.Tanh

tf.transpose ---> mindspore.ops.operations.Transpose

```
class BertModel(object):
  def __init__(...):
    config = copy.deepcopy(config)
    if not is_training:
      config.hidden_dropout_prob = 0.0
      config.attention_probs_dropout_prob = 0.0
    input_shape = get_shape_list(input_ids, expected_rank=2)
    batch_size = input_shape[0]
    seq_length = input_shape[1]
    if input_mask is None:
      input_mask = tf.ones(shape=[batch_size, seq_length], dtype=tf.int32)
    if token_type_ids is None:
      token_type_ids = tf.zeros(shape=[batch_size, seq_length], dtype=tf.int32)
    with tf.variable_scope(scope, default_name="bert", custom_getter=get_custom_getter(compute_type)):
      with tf.variable_scope("embeddings"):
        (self.embedding_output, self.embedding_table) = embedding_lookup(...)
        self.embedding_output = embedding_postprocessor(...)
      with tf.variable_scope("encoder"):
        attention_mask = create_attention_mask_from_input_mask(
            input_ids, input_mask)
        self.all_encoder_layers = transformer_model(
            input_tensor=tf.saturate_cast(self.embedding_output, compute_type),
            attention_mask=attention_mask,
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            intermediate_act_fn=get_activation(config.hidden_act),
            hidden_dropout_prob=config.hidden_dropout_prob,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
            initializer_range=config.initializer_range,
            do_return_all_layers=True,
            use_relative_position=config.use_relative_position,
            compute_type=compute_type)

      self.sequence_output = tf.cast(self.all_encoder_layers[-1], tf.float32)
      with tf.variable_scope("pooler"):
        first_token_tensor = tf.squeeze(self.sequence_output[:, 0:1, :], axis=1)
        self.pooled_output = tf.layers.dense(
            first_token_tensor,
            config.hidden_size,
            activation=tf.tanh,
            kernel_initializer=create_initializer(config.initializer_range))
```

Mindspore模型
```
class BertModel(nn.Cell):
    def __init__(self,...):
        super(BertModel, self).__init__()
        config = copy.deepcopy(config)
        if not is_training:
            config.hidden_dropout_prob = 0.0
            config.attention_probs_dropout_prob = 0.0
        self.seq_length = config.seq_length
        self.hidden_size = config.hidden_size
        self.num_hidden_layers = config.num_hidden_layers
        self.embedding_size = config.hidden_size
        self.token_type_ids = None
        self.last_idx = self.num_hidden_layers - 1
        output_embedding_shape = [-1, self.seq_length, self.embedding_size]
        self.bert_embedding_lookup = EmbeddingLookup(...)
        self.bert_embedding_postprocessor = EmbeddingPostprocessor(...)
        self.bert_encoder = BertTransformer(...)
        self.cast = P.Cast()
        self.dtype = config.dtype
        self.cast_compute_type = SaturateCast(dst_type=config.compute_type)
        self.slice = P.StridedSlice()
        self.squeeze_1 = P.Squeeze(axis=1)
        if config.do_quant:
            self.dense = DenseQuant(...).to_float(config.compute_type)
        else:
            self.dense = nn.Dense(...).to_float(config.compute_type)
        self._create_attention_mask_from_input_mask = CreateAttentionMaskFromInputMask(config)

    def construct(self, input_ids, token_type_ids, input_mask):
        word_embeddings, embedding_tables = self.bert_embedding_lookup(input_ids)
        embedding_output = self.bert_embedding_postprocessor(token_type_ids, word_embeddings)
        attention_mask = self._create_attention_mask_from_input_mask(input_mask)
        encoder_output, encoder_layers, layer_atts = self.bert_encoder(self.cast_compute_type(embedding_output),
                                                                       attention_mask)
        sequence_output = self.cast(encoder_output[self.last_idx], self.dtype)
        batch_size = P.Shape()(input_ids)[0]
        sequence_slice = self.slice(sequence_output, (0, 0, 0),(batch_size, 1, self.hidden_size),(1, 1, 1))
        first_token = self.squeeze_1(sequence_slice)
        pooled_output = self.dense(first_token)
        pooled_output = self.cast(pooled_output, self.dtype)
        encoder_outputs = ()
        for output in encoder_layers:
            encoder_outputs += (self.cast(output, self.dtype),)
        attention_outputs = ()
        for output in layer_atts:
            attention_outputs += (self.cast(output, self.dtype),)
        return sequence_output, pooled_output, embedding_tables, encoder_outputs, attention_outputs
```

## 模型效果
```
eval step: 68, Accuracy: 64.43041355355395
eval step: 69, Accuracy: 64.86934047621686
eval step: 70, Accuracy: 64.40608531717132
eval step: 71, Accuracy: 64.56146057674341
eval step: 72, Accuracy: 64.46960291263362
eval step: 73, Accuracy: 64.66819786703067
eval step: 74, Accuracy: 64.64251107451254
eval step: 75, Accuracy: 64.35778666048294
eval step: 76, Accuracy: 64.93177232324845
eval step: 77, Accuracy: 65.03449320275891
The best Accuracy: 65.057986

Process finished with exit code 0

```
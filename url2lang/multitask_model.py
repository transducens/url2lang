
import os
import logging

import url2lang.utils.utils as utils

import torch
import torch.nn as nn
import transformers

logger = logging.getLogger("url2lang")

# Adapted from https://colab.research.google.com/github/zphang/zphang.github.io/blob/master/files/notebooks/Multi_task_Training_with_Transformers_NLP.ipynb#scrollTo=aVX5hFlzmLka
#class MultitaskModel(transformers.PreTrainedModel):
class MultitaskModel(nn.Module):
    # Not using class from https://github.com/huggingface/transformers/blob/8298e4ec0291ee0d8bfd4fc620d3ab824e8b7bb4/src/transformers/modeling_utils.py#L972
    #  since it is very difficult to create the model we want to: https://github.com/huggingface/transformers/issues/18969

    def __init__(self, config):
        """
        Setting MultitaskModel up as a PretrainedModel allows us
        to take better advantage of Trainer features
        """
        cls = self.__class__

        super().__init__()

        self.config = config

        # Load values from the configuration
        c = config.to_dict()
        pretrained_model = c["_name_or_path"]
        self.tasks_names = c["u2l_tasks"]
        self.tasks_kwargs = c["u2l_tasks_kwargs"]

        # Load base model and tasks
        heads, heads_config = cls.get_task_heads(self.tasks_names, self.tasks_kwargs, pretrained_model)
        shared_encoder, task_models_dict = cls.get_base_and_heads(heads, heads_config, config, pretrained_model)
        #self.encoder = shared_encoder
        self._base_model_prefix = shared_encoder.base_model_prefix
        self.set_base_model(shared_encoder) # Workaround for https://github.com/huggingface/transformers/issues/18969

        self.update_task_models(task_models_dict)

    def update_task_models(self, task_models_dict):
        self.task_models_dict = task_models_dict
        self.task_models_dict_modules = nn.ModuleDict(task_models_dict)

    def load_model(self, model_input, device=None):
        if device:
            checkpoint = torch.load(model_input, map_location=device)
        else:
            checkpoint = torch.load(model_input)

        self.get_base_model().load_state_dict(checkpoint["encoder"])
        self.task_models_dict_modules.load_state_dict(checkpoint["tasks"])

        return self

    def save_model(self, model_output):
        torch.save({
            "encoder": self.get_base_model().state_dict(),
            "tasks": self.task_models_dict_modules.state_dict(),
        }, model_output)

    @classmethod
    def get_task_heads(cls, tasks, tasks_kwargs, model_source):
        heads = {}
        heads_config = {}

        for task in tasks:
            if task == "language-identification":
                heads[task] = ClassificationHead
            elif task == "mlm":
                logger.warning("MLM head implementation is for Roberta and BERT like models (i.e. it is slightly different from models like Albert)")

                heads[task] = MLMHead
            else:
                raise Exception(f"Unknown task: {task}")

            task_kwargs = tasks_kwargs[task] if tasks_kwargs and task in tasks_kwargs else {}
            heads_config[task] = transformers.AutoConfig.from_pretrained(model_source, **task_kwargs)

        return heads, heads_config

    def get_base_model(self):
        return getattr(self, self._base_model_prefix)

    def set_base_model(self, model):
        return setattr(self, self._base_model_prefix, model)

    def get_head(self, task):
        return self.task_models_dict_modules[task]

    def get_tasks_names(self):
        return self.tasks_names

    @classmethod
    def get_base_and_heads(cls, heads, heads_config, config, model_name):
        shared_encoder = transformers.AutoModel.from_pretrained(model_name)
        task_models_dict = {}

        for task_name, head_cls in heads.items():
            task_models_dict[task_name] = head_cls(heads_config[task_name])

        return shared_encoder, task_models_dict

    @classmethod
    def create(cls, model_name, tasks, tasks_kwargs):
        """
        This creates a MultitaskModel using the model class and config objects
        from single-task models.

        We do this by creating each single-task model, and having them share
        the same encoder transformer.
        """
        config = transformers.AutoConfig.from_pretrained(model_name)
        c = config.to_dict()
        c["u2l_tasks"] = tasks
        c["u2l_tasks_kwargs"] = tasks_kwargs

        config.update(c)

        instance = cls(config)

        return instance

    def forward(self, task_name, *args, **kwargs):
        if "encoder_output" in kwargs and kwargs["encoder_output"] is not None:
            output = kwargs["encoder_output"]
        else:
            # If necessary, remove attrs which doesn't match with transformers models
            if "encoder_output" in kwargs:
                del kwargs["encoder_output"]

            output = self.get_base_model()(*args, **kwargs)

            try:
                output = getattr(output, "last_hidden_state")
            except AttributeError:
                output = output[0]

        logits = self.task_models_dict_modules[task_name](output)

        return {
            "logits": logits,
            "encoder_output": output,
        }

# Heads
#  From https://github.com/nyu-mll/jiant/blob/de5437ae710c738a0481b13dc9d266dd558c43a4/jiant/proj/main/modeling/heads.py
#  From https://github.com/huggingface/transformers/blob/820c46a707ddd033975bc3b0549eea200e64c7da/src/transformers/models/xlm_roberta/modeling_xlm_roberta.py#L1453
#  From https://github.com/huggingface/transformers/blob/820c46a707ddd033975bc3b0549eea200e64c7da/src/transformers/models/xlm_roberta/modeling_xlm_roberta.py#L1139

class ClassificationHead(nn.Module):
    def __init__(self, config):
        """From RobertaClassificationHead"""
        super().__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
        self.num_labels = config.num_labels

    def forward(self, features, pool_in_first_token=True):
        pooled = features

        if pool_in_first_token:
            pooled = features[:, 0, :]  # take <s> token (equiv. to [CLS])

        x = self.dropout(pooled)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        logits = self.out_proj(x)

        return logits

class MLMHead(nn.Module):
    """From RobertaLMHead"""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.activation = transformers.models.bert.modeling_bert.ACT2FN[config.hidden_act if config.hidden_act else "gelu"]

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size), requires_grad=True)

        # Need a link between the two variables so that the bias is correctly resized with
        # `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, unpooled):
        x = self.dense(unpooled)
        x = self.activation(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        logits = self.decoder(x) + self.bias
        return logits

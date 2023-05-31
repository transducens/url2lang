
import os
import sys
import time
import random
import logging
import argparse
import tempfile
import contextlib
from datetime import datetime
import copy

import url2lang.utils.utils as utils
from url2lang.inference import (
    inference,
    interactive_inference,
    inference_with_heads,
)
from url2lang.metrics import (
    get_metrics,
    get_metrics_task_specific,
    plot_statistics,
)
import url2lang.preprocess as preprocess
from url2lang.multitask_model import MultitaskModel
import url2lang.dataset as dataset

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CyclicLR, LambdaLR
from torch.optim import Adam, AdamW, SGD
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    WeightedRandomSampler,
    SequentialSampler,
)
import transformers
from transformers import (
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
import sklearn

# Disable (less verbose) 3rd party logging
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL.PngImagePlugin").setLevel(logging.WARNING)

# Logging
logger = logging.getLogger("url2lang")
logger_verbose = {"tokens": logging}

# Other
DEBUG = bool(int(os.environ["U2L_DEBUG"])) if "U2L_DEBUG" in os.environ else False
_lr_scheduler_args = {
    "none": {},
    "linear": {
        "nargs": 1,
        "metavar": ("warmup_steps",),
        "default": ("10%",), # '%' is optional, and if not provided, absolute number of steps is taken
        "type": utils.argparse_nargs_type(str),
    },
    "CLR": {
        "nargs": 6,
        "metavar": ("max_lr", "step_size", "mode", "gamma", "max_lr_factor", "step_size_factor"),
        "default": (8e-5, 2000, "triangular2", 1.0, 4, 2),
        "type": utils.argparse_nargs_type(float, int, str, float, {"type": int, "choices": (3, 4)},
                                            {"type": int, "choices": tuple(range(2,8+1))}),
    },
    "inverse_sqrt": {
        "nargs": 1,
        "metavar": ("warmup_steps",),
        "default": ("10%",), # '%' is optional, and if not provided, absolute number of steps is taken
        "type": utils.argparse_nargs_type(str),
    }
}
_optimizer_args = {
    "none": {},
    "adam": {
        "nargs": 4,
        "metavar": ("beta1", "beta2", "eps", "weight_decay"),
        "default": (0.9, 0.999, 1e-08, 0.0),
        "type": utils.argparse_nargs_type(float, float, float, float),
    },
    "adamw": {
        "nargs": 4,
        "metavar": ("beta1", "beta2", "eps", "weight_decay"),
        "default": (0.9, 0.999, 1e-08, 0.01),
        "type": utils.argparse_nargs_type(float, float, float, float),
    },
    "sgd": {
        "nargs": 2,
        "metavar": ("momentum", "weight_decay"),
        "default": (0.0, 0.0),
        "type": utils.argparse_nargs_type(float, float),
    }
}
_unknown_lang_label = "unk"
_langs_to_detect_alpha_3 = [_unknown_lang_label]
# Languages from: https://commoncrawl.github.io/cc-crawl-statistics/plots/languages
_langs_to_detect_alpha_3 += [
    "eng", "deu", "rus", "fra", "zho", "spa",
    "jpn", "ita", "nld", "pol", "por", "ces",
    "vie", "tur", "ind", "swe", "ara", "fas",
    "kor", "hun", "ell", "ron", "dan", "fin",
    "tha", "slk", "nor", "ukr", "bul", "cat",
    "srp", "hrv", "slv", "lit", "hin", "est",
    "heb", "lat", "ben", "lav", "msa", "bos",
    "sqi", "tam", "glg", "isl", "aze", "kat",
    "mkd", "eus", "hye", "nep", "urd", "mon",
    "mal", "kaz", "mar", "tel", "nno", "bel",
    "uzb", "guj", "kan", "mya", "khm", "cym",
    "epo", "tgl", "sin", "afr", "tat", "swa",
    "gle", "pan", "kur", "kir", "tgk", "mlt",
    "fao", "ori", "lao", "som", "ltz", "oci",
    "amh", "fry", "bak", "pus", "san", "bre",
    "mlg", "hau", "tuk", "war", "asm", "cos",
    "div", "jav", "ceb", "kin", "hat", "zul",
    "gla", "bod", "xho", "yid", "snd", "mri",
    "uig", "roh", "sun", "kal", "yor", "tir",
    "abk", "bih", "haw", "hmn", "ina", "que",
    "grn", "ibo", "nya", "sco", "sna", "sot",
    "smo", "vol", "glv", "orm", "ile", "syr",
    "aar", "dzo", "iku", "kha", "lin", "lug",
    "mfe", "aka", "aym", "bis", "chr", "crs",
    "fij", "ipk", "nso", "run", "sag", "ssw",
    "ton", "tsn", "wol", "zha", "got", "kas",
    "lif", "nau", "sux", "tso", "ven"]
_lang2id = {_lang: idx for idx, _lang in enumerate(_langs_to_detect_alpha_3)} # id range: [0, len(_langs_to_detect_alpha_3) - 1]
_id2lang = {idx: _lang for _lang, idx in _lang2id.items()}

def get_lr_scheduler(scheduler, optimizer, *args, **kwargs):
    scheduler_instance = None
    mandatory_args = ""

    def check_args(num_args, str_args):
        if len(args) != num_args:
            raise Exception(f"LR scheduler: '{scheduler}' mandatory args: {str_args}")

    if scheduler == "none":
        pass
    elif scheduler == "linear":
        mandatory_args = "num_warmup_steps, num_training_steps"

        check_args(2, mandatory_args)

        scheduler_instance = get_linear_schedule_with_warmup(optimizer, *args, **kwargs)
    elif scheduler == "CLR": # CyclicLR
        mandatory_args = "base_lr, max_lr"

        check_args(2, mandatory_args)

        scheduler_instance = CyclicLR(optimizer, *args, **kwargs)
    elif scheduler == "inverse_sqrt":
        mandatory_args = "num_warmup_steps"

        check_args(1, mandatory_args)

        if optimizer is None:
            raise Exception(f"Optimizer not provided, so the selected LR scheduler can't be configured: {scheduler}")

        def inverse_sqrt(current_step):
            num_warmup_steps = args[0]

            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))

            # From https://fairseq.readthedocs.io/en/latest/_modules/fairseq/optim/lr_scheduler/inverse_square_root_schedule.html
            # In fairseq they set directly the LR to the optimizer, but we use it for a LR scheduler, so for us is a value which will multiply the LR
            initial_lr = optimizer.defaults["lr"]
            decay_factor = initial_lr * num_warmup_steps**0.5
            lr = decay_factor * current_step**-0.5

            return lr / initial_lr # This step makes that the multiplication of initial_lr doesn't affect, but the previous lines are just being similar
                                   #  to the version of fairseq

        scheduler_instance = LambdaLR(optimizer, inverse_sqrt, **kwargs)
    else:
        raise Exception(f"Unknown LR scheduler: {scheduler}")

    logger.debug("LR scheduler: '%s' mandatory args: %s: %s", scheduler, mandatory_args, str(args))
    logger.debug("LR scheduler: '%s' optional args: %s", scheduler, str(kwargs))

    return scheduler_instance

def load_model(tasks, tasks_kwargs, model_input="", pretrained_model="", device=""):
    if len(tasks) == 0:
        raise Exception("At least 1 head is mandatory")
    if set(tasks) != set(tasks_kwargs):
        raise Exception("Different tasks provided to 'tasks' and 'tasks_kwargs': "
                        f"{set(tasks)} vs {set(tasks_kwargs)}")

    multitask_model = MultitaskModel.create(pretrained_model, tasks, tasks_kwargs)

    if model_input:
        logger.info("Loading model: '%s'", model_input)

        multitask_model = multitask_model.load_model(model_input)

    # Move model to device
    if device:
        multitask_model = multitask_model.to(device)

    return multitask_model

def load_tasks_kwargs(all_tasks, auxiliary_tasks, regression):
    all_tasks_kwargs = {}
    total_auxiliary_tasks = 0
    num_labels = 1 if regression else len(_id2lang.keys())

    if "language-identification" in all_tasks:
        all_tasks_kwargs["language-identification"] = {
            "num_labels": num_labels,
        }
    if "mlm" in auxiliary_tasks:
        all_tasks_kwargs["mlm"] = {}

        logger.info("Using auxiliary task: mlm")

        total_auxiliary_tasks += 1

    if total_auxiliary_tasks == 0:
        logger.info("Not using any auxiliary task")

    if total_auxiliary_tasks != len(auxiliary_tasks):
        # We forgot something (e.g. update the code according to the new auxiliary tasks)
        raise Exception("The specified auxiliary tasks could not be loaded (bug): "
                        f"{' '.join(auxiliary_tasks)} ({len(auxiliary_tasks)})")

    return all_tasks_kwargs

def load_tokenizer(pretrained_model):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

    return tokenizer

def get_amp_context_manager(cuda_amp, use_cuda):
    use_cuda = torch.cuda.is_available()
    amp_context_manager = contextlib.nullcontext()
    amp_grad_scaler = None
    _cuda_amp = cuda_amp

    # Configure AMP context manager
    if cuda_amp and use_cuda:
        amp_context_manager = torch.cuda.amp.autocast()
        amp_grad_scaler = torch.cuda.amp.GradScaler()
        _cuda_amp = True

        logger.debug("AMP enabled for CUDA")
    elif cuda_amp:
        _cuda_amp = False
        logger.warning("AMP could not be enabled")

    return amp_context_manager, amp_grad_scaler, _cuda_amp

def load_dataset(filename_dataset, set_desc, shard_id, **kwargs):
    logger.debug("Allocated memory before starting tokenization (%s): %d", set_desc, utils.get_current_allocated_memory_size())

    file_dataset = open(filename_dataset[shard_id], mode="rt", errors="backslashreplace")
    input_data = []
    output_data = []

    # Read data from input files
    batch = utils.tokenize_batch_from_iterator(
                file_dataset, kwargs["tokenizer"], kwargs["batch_size"],
                f=lambda u: preprocess.preprocess_url(u, remove_protocol_and_authority=kwargs["remove_authority"],
                                                      remove_positional_data=kwargs["remove_positional_data_from_resource"],
                                                      separator=kwargs["url_separator"], lower=kwargs["lower"],
                                                      stringify_instead_of_tokenization=kwargs["stringify_instead_of_tokenization"]),
                auxiliary_tasks=kwargs["auxiliary_tasks"])

    for batch_urls in batch:
        input_data.extend(batch_urls["urls"])
        output_data.extend(batch_urls["labels_task_language_identification"])

        if len(input_data) != len(output_data):
            raise Exception(f"Different lengths for input and output data in {set_desc} set: {len(input_data)} vs {len(output_data)}")

    urls_lang_count = [len([l for l in output_data if l == c]) for c in sorted(_id2lang.keys())]
    total_data = sum(urls_lang_count)

    if total_data != len(input_data):
        raise Exception(f"Number of URLs per lang doesn't match the input data ({set_desc}): "
                        f"{total_data} != {len(input_data)}")

    langs_without_data = set()

    for lang, lang_id in _lang2id.items():
        q = urls_lang_count[lang_id]

        if q > 0:
            p = q * 100.0 / total_data

            logger.info("%d pairs of URLs loaded (%s): lang %s (%s %%)", q, set_desc, lang, f"{p:.2f}")
        else:
            langs_without_data.add(lang)

    if len(langs_without_data) != 0:
        langs_without_data = sorted(list(langs_without_data))

        logger.warning("Langs without data: %s", ", ".join(langs_without_data))

    if set_desc == "train":
        min_train_samples = min(urls_lang_count)
        classes_count = np.array(urls_lang_count)

        if 0 in classes_count:
            # TODO any solution?

            logger.warning("Can't check out if data is imbalanced because there are classes without data")
        else:
            min_classes_weights = min_train_samples / classes_count

            if kwargs["imbalanced_strategy"] == "none":
                # Is the data imbalanced? If so, warn about it

                for cw in min_classes_weights:
                    if cw < 0.9:
                        logger.warning("Your data seems to be imbalanced and you did not select any imbalanced data strategy")
                        break

    logger.debug("Allocated memory after tokenization (%s): %d", set_desc, utils.get_current_allocated_memory_size())

    # Prepare datasets data
    dataset_tasks_data = {}

    # Datasets
    dataset_instance = dataset.SmartBatchingURLsDataset(input_data, output_data, kwargs["tokenizer"],
                                                        kwargs["max_length_tokens"], regression=kwargs["regression"], set_desc=set_desc,
                                                        remove_instead_of_truncate=kwargs["remove_instead_of_truncate"],
                                                        imbalanced_strategy=kwargs["imbalanced_strategy"],
                                                        tasks_data=dataset_tasks_data)

    logger.debug("Allocated memory after encoding the data: %d", utils.get_current_allocated_memory_size())

    if set_desc == "train":
        logger.debug("Total tokens in file %d (train): %d", shard_id + 1, dataset_instance.total_tokens)
    else:
        logger.debug("Total tokens (%s): %d", set_desc, dataset_instance.total_tokens)

    # Remove data in order to free memory
    del input_data
    del output_data
    del dataset_tasks_data

    logger.debug("Allocated memory after removing pairs of URLs (str): %d", utils.get_current_allocated_memory_size())

    dataloader_instance = dataset_instance.get_dataloader(kwargs["batch_size"], kwargs["device"], kwargs["force_cpu"],
                                                          kwargs["dataset_workers"], max_tokens=kwargs["max_tokens"])

    file_dataset.close()

    return dataset_instance, dataloader_instance

def main(args):
    # https://discuss.pytorch.org/t/calculating-f1-score-over-batched-data/83348
    #logger.warning("Some metrics are calculated on each batch and averaged, so the values might not be fully correct (e.g. F1)")

    apply_inference = args.inference
    multiple_shards = False

    if not apply_inference:
        filename_dataset_train = args.dataset_train_filename.split(':')
        filename_dataset_dev = args.dataset_dev_filename.split(':')
        filename_dataset_test = args.dataset_test_filename.split(':')

        if len(filename_dataset_train) > 1:
            multiple_shards = True

            logger.info("Multiple train files were provided: %d: one per epoch will be used using round-robin", len(filename_dataset_train))
        if len(filename_dataset_dev) > 1:
            logger.warning("Multiple dev files were provided, but only the first one will be used")
        if len(filename_dataset_test) > 1:
            logger.warning("Multiple test files were provided, but only the first one will be used")

        # Discard dev/test files if needed
        filename_dataset_dev = filename_dataset_dev[:1]
        filename_dataset_test = filename_dataset_test[:1]

    # Args
    batch_size = args.batch_size
    block_size = args.block_size
    max_tokens = args.max_tokens if args.max_tokens > 0 else None
    epochs = args.epochs # BE AWARE! "epochs" might be fake due to --train-until-patience
    force_cpu = args.force_cpu
    use_cuda = utils.use_cuda(force_cpu=force_cpu) # Will be True if possible and False otherwise
    device = torch.device("cuda:0" if use_cuda else "cpu")
    is_device_gpu = device.type.startswith("cuda")
    pretrained_model = args.pretrained_model
    max_length_tokens = args.max_length_tokens
    model_input = utils.resolve_path(args.model_input)
    model_output = utils.resolve_path(args.model_output)
    seed = args.seed
    plot = args.plot
    plot_path = utils.resolve_path(args.plot_path)
    inference_from_stdin = args.inference_from_stdin
    parallel_likelihood = args.parallel_likelihood
    threshold = args.threshold
    imbalanced_strategy = args.imbalanced_strategy # TODO remove or adapt
    patience = args.patience
    do_not_load_best_model = args.do_not_load_best_model
    remove_authority = args.remove_authority
    remove_positional_data_from_resource = args.remove_positional_data_from_resource
    log_directory = args.log_directory
    regression = args.regression
    train_until_patience = args.train_until_patience
    url_separator = args.url_separator
    url_separator_new_token = args.url_separator_new_token
    learning_rate = args.learning_rate
    re_initialize_last_n_layers = max(0, args.re_initialize_last_n_layers)
    scheduler_str = args.lr_scheduler
    lr_scheduler_args = args.lr_scheduler_args # Content might vary depending on the value of scheduler_str
    cuda_amp = args.cuda_amp
    llrd = args.llrd
    lock_file = args.lock_file
    stringify_instead_of_tokenization = args.stringify_instead_of_tokenization
    lower = args.lowercase
    auxiliary_tasks = args.auxiliary_tasks if args.auxiliary_tasks else []
    auxiliary_tasks_weights = args.auxiliary_tasks_weights
    auxiliary_tasks_flags = args.auxiliary_tasks_flags if args.auxiliary_tasks_flags else []
    freeze_embeddings_layer = args.freeze_embeddings_layer
    waiting_time = args.waiting_time
    remove_instead_of_truncate = args.remove_instead_of_truncate
    optimizer_str = args.optimizer
    optimizer_args = args.optimizer_args # Content might vary depending on the value of optimizer_str
    best_dev_metric = args.best_dev_metric
    task_dev_metric = args.task_dev_metric
    dataset_workers = args.dataset_workers
    pre_load_shards = args.pre_load_shards

    if auxiliary_tasks:
        _auxiliary_tasks_weights = {}

        if not auxiliary_tasks_weights:
            for task in auxiliary_tasks:
                _auxiliary_tasks_weights[task] = 1.0
        elif len(auxiliary_tasks) != len(auxiliary_tasks_weights):
            raise Exception("You need to provide weights either for all the auxiliary tasks or for none of them")
        else:
            for task, weight in zip(auxiliary_tasks, auxiliary_tasks_weights):
                _auxiliary_tasks_weights[task] = weight

        auxiliary_tasks_weights = _auxiliary_tasks_weights
        auxiliary_tasks_weights["language-identification"] = 1.0

        logger.debug("Auxiliary tasks weights: %s", str(auxiliary_tasks_weights))

    auxiliary_tasks = sorted(list(set(utils.get_tuple_if_is_not_tuple(auxiliary_tasks))))
    all_tasks = ["language-identification"] + auxiliary_tasks

    if not block_size:
        block_size = batch_size
    if batch_size < block_size:
        logger.warning("Block size has to be less than or equal to batch size: updating block size to batch size: %d", batch_size)

        block_size = batch_size

    if lock_file and utils.exists(lock_file):
        logger.warning("Lock file ('%s') exists: finishing training", lock_file)

        sys.exit(0)
    if lock_file:
        logger.debug("Lock file will be created if the training finishes: %s", lock_file)

    amp_context_manager, amp_grad_scaler, cuda_amp = get_amp_context_manager(cuda_amp, use_cuda)

    if scheduler_str in ("linear",) and train_until_patience:
        # Depending on the LR scheduler, the training might even stop at some point (e.g. linear LR scheduler will set the LR=0 if the run epochs is greater than the provided epochs)
        logger.warning("You set a LR scheduler ('%s' scheduler) which conflicts with --train-until-patince: you might want to check this out and change the configuration", scheduler_str)

    # Enable cuDNN benchmark
    if use_cuda:
        torch.backends.cudnn.benchmark = True

    # Disable parallelism since throws warnings
    os.environ["TOKENIZERS_PARALLELISM"] = "False"

    if apply_inference and not model_input:
        logger.warning("Flag --model-input is recommended when --inference is provided: waiting %d seconds before proceed", waiting_time)

        time.sleep(waiting_time)

    logger.debug("Pretrained model architecture: %s", pretrained_model)

    if (do_not_load_best_model or not model_output) and not apply_inference:
        logger.warning("Final dev and test evaluation will not be carried out with the best model")

    if plot_path and not plot:
        raise Exception("--plot is mandatory if you set --plot-path")

    if seed >= 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        logger.debug("Deterministic values enabled (not fully-guaranteed): seed %d", seed)
    else:
        logger.warning("Deterministic values disable (you set a negative seed)")

    if max_length_tokens > 512:
        logger.warning("HuggingFace models can handle a max. of 512 tokens at once and you set %d: changing value to 512")

        max_length_tokens = 512
    if max_tokens and max_tokens < max_length_tokens:
        logger.warning("The specified max_tokens has to be greater or equal that the max length tokens of the model: "
                       "changing value from %d to %d", max_tokens, max_length_tokens)

        max_tokens = max_length_tokens

    logger.info("Device: %s", device)

    if not apply_inference:
        logger.debug("Train data file/s: %s", ' '.join(filename_dataset_train))
        logger.debug("Dev data file: %s", filename_dataset_dev[0])
        logger.debug("Test data file: %s", filename_dataset_test[0])

    all_tasks_kwargs = load_tasks_kwargs(all_tasks, auxiliary_tasks, regression)
    model = load_model(all_tasks, all_tasks_kwargs, model_input=model_input, pretrained_model=pretrained_model, device=device)

    if model_output:
        logger.info("Model will be stored: %s", model_output)

        if utils.exists(model_output):
            if args.overwrite_output_model:
                logger.warning("Provided output model does exist and it will be updated: waiting %d seconds before proceed", waiting_time)

                time.sleep(waiting_time)
            else:
                raise Exception(f"Provided output model does exist: {model_output}")

    tokenizer = load_tokenizer(pretrained_model)
    fine_tuning = not args.do_not_fine_tune
    freeze_whole_model = args.freeze_whole_model
    model_embeddings_size = model.get_base_model().base_model.embeddings.word_embeddings.weight.shape[0]

    if freeze_whole_model and fine_tuning:
        logger.warning("The whole model won't be frozen since fine-tuning is enabled")

        freeze_whole_model = False

    if url_separator_new_token:
        # Add new special token (URL separator)
        num_added_toks = tokenizer.add_tokens([url_separator], special_tokens=True)

        logger.debug("New tokens added to tokenizer: %d", num_added_toks)

        if not model_input:
            model.get_base_model().resize_token_embeddings(len(tokenizer))

            if freeze_embeddings_layer:
                logger.warning("Embeddings layer is frozen, and new tokens will not be trained")
        elif model_embeddings_size + 1 == len(tokenizer):
            logger.warning("You've loaded a model which does not have the new token, so the results might be unexpected")

            model.get_base_model().resize_token_embeddings(len(tokenizer))

    model_embeddings_size = model.get_base_model().base_model.embeddings.word_embeddings.weight.shape[0]

    if model_embeddings_size != len(tokenizer):
        logger.error("Embedding layer size does not match with the tokenizer size: %d vs %d", model_embeddings_size, len(tokenizer))

    if apply_inference:
        interactive_inference(model, tokenizer, batch_size, max_length_tokens, device, amp_context_manager,
                              inference_from_stdin=inference_from_stdin, remove_authority=remove_authority,
                              parallel_likelihood=parallel_likelihood, threshold=threshold, url_separator=url_separator,
                              remove_positional_data_from_resource=remove_positional_data_from_resource, lower=lower,
                              auxiliary_tasks=auxiliary_tasks, auxiliary_tasks_flags=auxiliary_tasks_flags)

        logger.info("Done!")

        # Stop execution
        return

    if regression:
        if imbalanced_strategy == "weighted-loss":
            logger.warning("Incompatible weight strategy ('%s'): regression can't be applied with the selected strategy: "
                           "it will not be applied", imbalanced_strategy)

            imbalanced_strategy = "none"

    # Unfreeze model layers
    for param in model.parameters():
        param.requires_grad = True

    # Freeze layers of the model, if needed
    if not fine_tuning:
        for param in model.parameters() if freeze_whole_model else model.get_base_model().parameters():
            param.requires_grad = False

    # Freeze embeddings layer, if needed
    for param in model.get_base_model().base_model.embeddings.parameters():
        param.requires_grad = not freeze_embeddings_layer

    last_layer_output = utils.get_layer_from_model(model.get_base_model().base_model.encoder.layer[-1], name="output.dense.weight")

    # Re-initilize last N layers from the pre-trained model
    if fine_tuning and re_initialize_last_n_layers > 0:
        utils.do_reinit(model.get_base_model().base_model, re_initialize_last_n_layers)

    # Load first shard
    dataset_static_args = {
        "regression": regression,
        "remove_instead_of_truncate": remove_instead_of_truncate,
        "imbalanced_strategy": imbalanced_strategy,
        "batch_size": batch_size,
        "device": device,
        "force_cpu": force_cpu,
        "dataset_workers": dataset_workers,
        "max_tokens": max_tokens,
        "remove_authority": remove_authority,
        "remove_positional_data_from_resource": remove_positional_data_from_resource,
        "url_separator": url_separator,
        "lower": lower,
        "stringify_instead_of_tokenization": stringify_instead_of_tokenization,
        "auxiliary_tasks": auxiliary_tasks,
        "auxiliary_tasks_flags": auxiliary_tasks_flags,
        "tokenizer": tokenizer,
        "max_length_tokens": max_length_tokens,
    }

    load_all_shards = []
    training_steps_per_epoch = 0
    count_labels_task_per_class = {}

    if pre_load_shards:
        load_all_shards = list(range(1, len(filename_dataset_train)))

    for shard_id in load_all_shards + [0]: # We load the first shard the last one
        if multiple_shards:
            logger.info("Loading shard (train): %d", shard_id)

        dataset_train, dataloader_train = \
            load_dataset(filename_dataset_train, "train", shard_id, **dataset_static_args)

        training_steps_per_epoch += len(dataloader_train) # BE AWARE! "dataloader_train" might change per epoch due to sharding

        for head_task in all_tasks:
            if head_task in ("language-identification",):
                if head_task not in count_labels_task_per_class:
                    count_labels_task_per_class[head_task] = {}

                all_classes = np.unique(dataset_train.labels[head_task])

                for c in all_classes:
                    if c not in count_labels_task_per_class[head_task]:
                        count_labels_task_per_class[head_task][c] = 0

                    count_labels_task_per_class[head_task][c] += len([l for l in dataset_train.labels[head_task] if l == c])

    dataset_dev, _ = \
        load_dataset(filename_dataset_dev, "dev", 0, **dataset_static_args)
    dataset_test, _ = \
        load_dataset(filename_dataset_test, "test", 0, **dataset_static_args)

    classes = len(_id2lang.keys()) # TODO is it ok?
    criteria = {}
    training_steps = training_steps_per_epoch * epochs # BE AWARE! "epochs" might be fake due to --train-until-patience

    # Get criterion for each head task
    for head_task in all_tasks:
        if head_task in ("language-identification",):
            all_classes = sorted(count_labels_task_per_class[head_task].keys())
            n_samples = sum([count_labels_task_per_class[head_task][c] for c in all_classes])
            n_classes = len(all_classes)
            classes_weights = [n_samples / (n_classes * count_labels_task_per_class[head_task][c]) for c in all_classes] # Same formula used in sklearn
            classes_weights = torch.as_tensor(classes_weights, dtype=torch.float)

            if DEBUG:
                classes_weights_sklearn = \
                    torch.as_tensor(sklearn.utils.class_weight.compute_class_weight("balanced",
                                                                                    classes=np.unique(dataset_train.labels[head_task]),
                                                                                    y=dataset_train.labels[head_task].numpy()),
                                    dtype=torch.float)

                if (classes_weights != classes_weights_sklearn).any().item():
                    if not multiple_shards or pre_load_shards:
                        logger.error("Own classes weights and sklearn version with different values: %s vs %s", classes_weights, pre_load_shards)
                    else:
                        logger.warning("Own classes weights and sklearn version with different values (it is expected due to loading "
                                       "multiple shards): %s vs %s", classes_weights, pre_load_shards)

            loss_weight = classes_weights if imbalanced_strategy == "weighted-loss" else None

            logger.debug("Classes weights (task '%s'): %s", head_task, str(classes_weights))

            # https://medium.com/dejunhuang/learning-day-57-practical-5-loss-function-crossentropyloss-vs-bceloss-in-pytorch-softmax-vs-bd866c8a0d23
            if regression:
                # Regression
                criterion = nn.BCEWithLogitsLoss(reduction="mean") # Raw input, not normalized
                                                                   #  (i.e. sigmoid is applied in the loss function)
            else:
                # Binary classification
                criterion = nn.CrossEntropyLoss(weight=loss_weight, reduction="mean") # Raw input, not normalized
                                                                                      #  (i.e. softmax is applied in the loss function)
        elif head_task == "mlm":
            criterion = nn.CrossEntropyLoss()
        else:
            raise Exception(f"Unknown head task: {head_task}")

        criterion = criterion.to(device)

        criteria[head_task] = criterion

    if llrd:
        if optimizer_str == "adam":
            model_parameters = utils.get_model_parameters_applying_llrd(model, learning_rate, weight_decay=0.0)
        else:
            if optimizer_str != "adamw":
                logger.warning("Using LLRD with the configuration of AdamW optimizer even '%s' was selected", optimizer_str)

            model_parameters = utils.get_model_parameters_applying_llrd(model, learning_rate, weight_decay=0.01) # AdamW
    else:
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())

    logger.debug("Optimizer args: %s", optimizer_args)

    if optimizer_str == "none":
        optimizer = None

        logger.debug("Be aware that even with the optimizer disabled minor changes might be observed while training since the model is "
                     "not in inference mode, so layers like Dropout have a random component which is enabled")
    elif optimizer_str == "adam":
        optimizer_kwargs = {
            "betas": tuple(optimizer_args[0:2]),
            "eps": optimizer_args[2],
            "weight_decay": optimizer_args[3],
        }
        optimizer = Adam(model_parameters, lr=learning_rate, **optimizer_kwargs)
    elif optimizer_str == "adamw":
        optimizer_kwargs = {
            "betas": tuple(optimizer_args[0:2]),
            "eps": optimizer_args[2],
            "weight_decay": optimizer_args[3],
        }
        optimizer = AdamW(model_parameters, lr=learning_rate, **optimizer_kwargs)
    elif optimizer_str == "sgd":
        optimizer_kwargs = {
            "momentum": optimizer_args[0],
            "weight_decay": optimizer_args[1],
        }
        optimizer = SGD(model_parameters, lr=learning_rate, **optimizer_kwargs)
    else:
        raise Exception(f"Unknown optimizer: {optimizer_str}")

    # Get LR scheduler args
    scheduler_args = []
    scheduler_kwargs = {}

    logger.debug("LR scheduler args: %s", lr_scheduler_args)

    if scheduler_str == "none":
        pass
    elif scheduler_str == "linear":
        if lr_scheduler_args[0][-1] == '%':
            scheduler_args = [int((float(lr_scheduler_args[0][:-1]) / 100.0) * training_steps), training_steps]

            if multiple_shards and not pre_load_shards:
                logger.warning("LR scheduler: '%s': multiple train files were provided, but only the first "
                            "one has been used for calculating the total number of steps, which affects the selecter LR scheduler", scheduler_str)
        else:
            scheduler_args = [int(lr_scheduler_args[0]), training_steps]
    elif scheduler_str == "CLR":
        scheduler_max_lr, scheduler_step_size, scheduler_mode, scheduler_gamma, scheduler_max_lr_factor, scheduler_step_size_factor \
            = lr_scheduler_args

        if learning_rate > scheduler_max_lr:
            new_scheduler_max_lr = learning_rate * scheduler_max_lr_factor # Based on the CLR paper (possible values are [3.0, 4.0])

            logger.warning("LR scheduler: '%s': provided LR (%f) is greater than provided max. LR (%f): setting max. LR to %f",
                           scheduler_str, learning_rate, scheduler_max_lr, new_scheduler_max_lr)

            scheduler_max_lr = new_scheduler_max_lr
        if scheduler_step_size <= 0:
            scheduler_step_size = scheduler_step_size_factor * training_steps_per_epoch # Based on the CLR paper (possible values are [2, ..., 8])

            logger.warning("LR scheduler: '%s': provided step size is 0 or negative: setting value to %d", scheduler_str, scheduler_step_size)

            if multiple_shards and not pre_load_shards:
                logger.warning("LR scheduler: '%s': multiple train files were provided, but only the first "
                               "one has been used for calculating the number of steps, which affects the selecter LR scheduler", scheduler_str)

        scheduler_args = [learning_rate, scheduler_max_lr]
        scheduler_kwargs = {
            "step_size_up": scheduler_step_size,
            "step_size_down": scheduler_step_size,
            "mode": scheduler_mode,
            "gamma": scheduler_gamma,
            "cycle_momentum": False, # https://github.com/pytorch/pytorch/issues/73910
        }
    elif scheduler_str == "inverse_sqrt":
        if lr_scheduler_args[0][-1] == '%':
            scheduler_args = [int((float(lr_scheduler_args[0][:-1]) / 100.0) * training_steps)]

            if multiple_shards and not pre_load_shards:
                logger.warning("LR scheduler: '%s': multiple train files were provided, but only the first "
                               "one has been used for calculating the total number of steps, which affects the selecter LR scheduler", scheduler_str)
        else:
            scheduler_args = [int(lr_scheduler_args[0])]
    else:
        raise Exception(f"Unknown LR scheduler: {scheduler}")

    scheduler = get_lr_scheduler(scheduler_str, optimizer, *scheduler_args, **scheduler_kwargs)

    best_values_minimize = False
    best_values_maximize = True

    if best_dev_metric == "loss":
        best_values_minimize = True
        best_values_maximize = False
    elif best_dev_metric == "Macro-F1":
        pass
    elif best_dev_metric == "MCC":
        pass
    else:
        raise Exception(f"Unknown best dev metric: {best_dev_metric}")

    best_values_binary_func_comp = (lambda a, b: a > b) if best_values_minimize else (lambda a, b: a < b)

    if not best_values_minimize ^ best_values_maximize:
        raise Exception("You can either minimize or maximize")

    logger.debug("Best values are being %s", "minimized" if best_values_minimize else "maximized")

    show_statistics_every_batches = 50
    final_loss = 0.0
    final_acc = 0.0
    final_acc_per_class = np.zeros(classes)
    final_f1_per_class = np.zeros(classes)
    final_macro_f1 = 0.0
    final_mcc = 0.0
    best_dev = np.inf * (1 if best_values_minimize else -1)
    best_train = np.inf * (1 if best_values_minimize else -1)
    stop_training = False
    epoch = 0
    current_patience = 0
    total_blocks_per_batch = 1 if max_tokens else max(int(np.ceil(batch_size / block_size)), 1)

    # Statistics
    batch_loss = []
    batch_acc = []
    batch_acc_classes = {0: [], 1: []}
    batch_macro_f1 = []
    epoch_train_loss, epoch_dev_loss = [], []
    epoch_train_acc, epoch_dev_acc = [], []
    epoch_train_acc_classes, epoch_dev_acc_classes = {0: [], 1: []}, {0: [], 1: []}
    epoch_train_macro_f1, epoch_dev_macro_f1 = [], []

    if task_dev_metric not in all_tasks:
        raise Exception(f"Selected task not found in the available tasks: '{task_dev_metric}' not in {str(all_tasks)}")

    # Start training!
    while not stop_training:
        logger.info("Epoch %d", epoch + 1)

        epoch_loss = 0.0
        epoch_acc = 0.0
        epoch_acc_per_class = np.zeros(classes)
        epoch_f1_per_class = np.zeros(classes)
        epoch_macro_f1 = 0.0
        all_outputs = {aux_task: [] for aux_task in all_tasks}
        all_labels = {aux_task: [] for aux_task in all_tasks}
        total_train_tokens = 0
        total_train_tokens_with_padding = 0
        idx = -1

        model.train()

        for batch in dataloader_train:
            if max_tokens and batch is None:
                # Batch is under construction using max_tokens...
                continue

            idx += 1
            batch_outputs = {aux_task: [] for aux_task in all_tasks}
            batch_labels = {aux_task: [] for aux_task in all_tasks}
            loss_value = None
            tasks_loss_value = {t: 0.0 for t in all_tasks}

            #optimizer.zero_grad() # https://discuss.pytorch.org/t/model-zero-grad-or-optimizer-zero-grad/28426/6
            model.zero_grad()

            # Process in block_size blocks in order to avoid OOM errors, but use batch_size for update the model
            for inputs_and_outputs in utils.get_data_from_batch(batch, None if max_tokens else block_size, None):
                total_train_tokens += sum([len(urls[urls != tokenizer.pad_token_id]) for urls in inputs_and_outputs["urls"]])
                total_train_tokens_with_padding += sum([len(urls) for urls in inputs_and_outputs["urls"]])

                # Inference
                results = inference_with_heads(model, all_tasks, tokenizer, inputs_and_outputs, amp_context_manager,
                                               criteria=criteria, tasks_weights=auxiliary_tasks_weights, device=device)

                # Main task
                loss = results["_internal"]["total_loss"] # Multiple losses if auxiliary tasks were used
                loss /= total_blocks_per_batch # Gradient accumulation

                # Results
                if loss_value is None:
                    loss_value = loss.cpu().detach().numpy() # Accumulated loss of all tasks
                else:
                    loss_value += loss.cpu().detach().numpy()

                if "language-identification" in all_tasks:
                    _outputs = results["language-identification"]["outputs_classification"]
                    _labels = inputs_and_outputs["labels_task_language_identification"].cpu().detach()
                    _labels = torch.round(_labels).type(torch.long) if regression else _labels

                    batch_outputs["language-identification"].extend(_outputs.tolist())
                    batch_labels["language-identification"].extend(_labels.tolist())
                if "mlm" in all_tasks:
                    pass

                if optimizer is not None:
                    if cuda_amp:
                        amp_grad_scaler.scale(loss).backward()
                    else:
                        loss.backward()

                for t in results.keys():
                    if t.startswith("_"):
                        # It is not a task, but some internal value
                        continue

                    # Losses per task of a whole batch (cumulate loss of blocks per task)
                    tasks_loss_value[t] += results[t]["loss_detach"].numpy()

                    # Drop immediate buffers (https://gist.github.com/thomwolf/ac7a7da6b1888c2eeac8ac8b9b05d3d3)
                    loss.detach_()

            if cuda_amp:
                # https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-clipping

                amp_grad_scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            for aux_task in all_tasks:
                all_outputs[aux_task].extend(batch_outputs[aux_task])
                all_labels[aux_task].extend(batch_labels[aux_task])

            # Get metrics
            log = (idx + 1) % show_statistics_every_batches == 0
            batch_labels_tensor = torch.as_tensor(batch_labels[task_dev_metric])
            current_batch_size = batch_labels_tensor.shape[0]
            metrics = get_metrics_task_specific(task_dev_metric, torch.as_tensor(batch_outputs[task_dev_metric]),
                                                batch_labels_tensor, current_batch_size, classes=classes, batch_idx=idx, log=log)

            if log:
                logger.debug("[train:batch#%d] Loss: %f", idx + 1, loss_value)
                logger.debug("[train:batch#%d] Processed tokens (without padding): %d (%d)", idx + 1, total_train_tokens_with_padding,
                             total_train_tokens)

                if len(auxiliary_tasks) > 0:
                    for t, v in tasks_loss_value.items():
                        # Log loss of all tasks
                        logger.debug("[train:batch#%d] Loss task '%s': %f", idx + 1, t, v)

                    for aux_task in auxiliary_tasks:
                        if aux_task in ("mlm",):
                            continue

                        _outputs = torch.as_tensor(batch_outputs[aux_task])
                        _labels = torch.as_tensor(batch_labels[aux_task])

                        # We don't want the metrics, just the log
                        get_metrics_task_specific(aux_task, _outputs, _labels, current_batch_size,
                                                  classes=classes, batch_idx=idx, log=log)

            show_statistics = (epoch == 0 and idx == 0) or (idx + 1) % show_statistics_every_batches == 0
            epoch_loss += loss_value
            epoch_acc += metrics["acc"]
            epoch_acc_per_class += metrics["acc_per_class"]
            epoch_f1_per_class += metrics["f1"]
            epoch_macro_f1 += metrics["macro_f1"]

            if plot and show_statistics:
                utils.append_from_tuple((batch_loss, epoch_loss / (idx + 1)),
                                        (batch_acc, epoch_acc * 100.0 / (idx + 1)),
                                        (batch_acc_classes[0], epoch_f1_per_class[0] * 100.0 / (idx + 1)),
                                        (batch_acc_classes[1], epoch_f1_per_class[1] * 100.0 / (idx + 1)),
                                        (batch_macro_f1, epoch_macro_f1 * 100.0 / (idx + 1)))

                if epoch != 0 or idx != 0:
                    plot_args = {
                        "show_statistics_every_batches": show_statistics_every_batches,
                        "batch_loss": batch_loss,
                        "batch_acc": batch_acc,
                        "batch_acc_classes": batch_acc_classes,
                        "batch_macro_f1": batch_macro_f1,
                        "epoch": epoch,
                        "epoch_train_loss": epoch_train_loss,
                        "epoch_train_acc": epoch_train_acc,
                        "epoch_train_acc_classes": epoch_train_acc_classes,
                        "epoch_train_macro_f1": epoch_train_macro_f1,
                        "epoch_dev_loss": epoch_dev_loss,
                        "epoch_dev_acc": epoch_dev_acc,
                        "epoch_dev_acc_classes": epoch_dev_acc_classes,
                        "epoch_dev_macro_f1": epoch_dev_macro_f1,
                        "final_dev_acc": None,
                        "final_dev_macro_f1": None,
                        "final_test_acc": None,
                        "final_test_macro_f1": None,
                    }

                    plot_statistics(plot_args, path=args.plot_path)

            if scheduler and show_statistics:
                # LRs statistics
                all_lrs = scheduler.get_last_lr()
                current_lr = all_lrs[0]
                len_lrs = len(all_lrs)

                if len_lrs != 1:
                    logger.debug("[batch#%d] LR scheduler: First and last LRs: %s", idx + 1, f"{str(all_lrs[0:10])[:-1]} ... {str(all_lrs[10:])[1:]}")
                else:
                    logger.debug("[batch#%d] LR scheduler: Current LR: %.8f", idx + 1, current_lr)

            if cuda_amp:
                amp_grad_scaler.step(optimizer)
                amp_grad_scaler.update()
            else:
                if optimizer is not None:
                    optimizer.step()

            if scheduler:
                scheduler.step()

        if total_train_tokens != dataset_train.total_tokens:
            if imbalanced_strategy in ("none", "weighted-loss"):
                logger.error("Total processed tokens are different from the initial total tokens: %d vs %d",
                             total_train_tokens, dataset_train.total_tokens)
            else:
                # The selected imbalanced_strategy modifies the number of samples, so we can't compare if it's what we expect
                pass

        current_last_layer_output = utils.get_layer_from_model(model.get_base_model().base_model.encoder.layer[-1], name="output.dense.weight")
        layer_updated = (current_last_layer_output != last_layer_output).any().cpu().detach().numpy()

        logger.debug("Has the model layer been updated? %s", 'yes' if layer_updated else 'no')

        dev_inference_metrics = inference(model, block_size, batch_size, all_tasks, tokenizer, criteria, dataset_dev,
                                          device, amp_context_manager, classes=classes, max_tokens=max_tokens)

        for aux_task in all_tasks:
            if aux_task in ("mlm",):
                continue

            all_outputs[aux_task] = torch.as_tensor(all_outputs[aux_task])
            all_labels[aux_task] = torch.as_tensor(all_labels[aux_task])
            metrics = get_metrics_task_specific(aux_task, all_outputs[aux_task], all_labels[aux_task],
                                                len(all_labels[aux_task]), classes=classes)
            epoch_acc = metrics["acc"]
            epoch_acc_per_class = metrics["acc_per_class"]
            epoch_f1_per_class = metrics["f1"]
            epoch_macro_f1 = metrics["macro_f1"]
            epoch_mcc = metrics["mcc"]

            if aux_task == task_dev_metric:
                epoch_loss /= idx + 1
                final_loss += epoch_loss
                final_acc += epoch_acc
                final_acc_per_class += epoch_acc_per_class
                final_f1_per_class += epoch_f1_per_class
                final_macro_f1 += epoch_macro_f1
                final_mcc += epoch_mcc

            logger.info("[train:epoch#%d] Avg. loss: %f", epoch + 1, epoch_loss)
            logger.info("[train:epoch#%d] Acc (task '%s'): %.2f %% (%s)",
                        epoch + 1, aux_task, epoch_acc * 100.0, "; ".join([f"{_lang}: {epoch_acc_per_class[_id] * 100.0:.2f} %" for _lang, _id in _lang2id.items()]))
            logger.info("[train:epoch#%d] Values per class (task '%s'; f1): (%s)",
                        epoch + 1, aux_task, "; ".join([f"{_lang}: {epoch_f1_per_class[_id] * 100.0:.2f} %" for _lang, _id in _lang2id.items()]))
            logger.info("[train:epoch#%d] Macro F1 (task '%s'): %.2f %%", epoch + 1, aux_task, epoch_macro_f1 * 100.0)
            logger.info("[train:epoch#%d] MCC (task '%s'): %.2f %%", epoch + 1, aux_task, epoch_mcc * 100.0)

            # Dev metrics
            dev_loss = dev_inference_metrics["loss"]
            dev_acc = dev_inference_metrics["metrics"][aux_task]["acc"]
            dev_acc_per_class = dev_inference_metrics["metrics"][aux_task]["acc_per_class"]
            dev_precision_per_class = dev_inference_metrics["metrics"][aux_task]["precision"]
            dev_recall_per_class = dev_inference_metrics["metrics"][aux_task]["recall"]
            dev_f1_per_class = dev_inference_metrics["metrics"][aux_task]["f1"]
            dev_macro_f1 = dev_inference_metrics["metrics"][aux_task]["macro_f1"]
            dev_mcc = dev_inference_metrics["metrics"][aux_task]["mcc"]

            logger.info("[dev:epoch#%d] Avg. loss: %f", epoch + 1, dev_loss)
            logger.info("[dev:epoch#%d] Acc (task '%s'): %.2f %% (%s)",
                        epoch + 1, aux_task, dev_acc * 100.0, "; ".join([f"{_lang}: {dev_acc_per_class[_id] * 100.0:.2f} %" for _lang, _id in _lang2id.items()]))
            logger.info("[dev:epoch#%d] Values per class (task '%s'; precision|recall|f1): "
                        "(%s | %s | %s)", epoch + 1, aux_task,
                        "; ".join([f"{_lang}: {dev_precision_per_class[_id] * 100.0:.2f} %" for _lang, _id in _lang2id.items()]),
                        "; ".join([f"{_lang}: {dev_recall_per_class[_id] * 100.0:.2f} %" for _lang, _id in _lang2id.items()]),
                        "; ".join([f"{_lang}: {dev_f1_per_class[_id] * 100.0:.2f} %" for _lang, _id in _lang2id.items()]))
            logger.info("[dev:epoch#%d] Macro F1 (task '%s'): %.2f %%", epoch + 1, aux_task, dev_macro_f1 * 100.0)
            logger.info("[dev:epoch#%d] MCC (task '%s'): %.2f %%", epoch + 1, aux_task, dev_mcc * 100.0)

            # TODO TBD get best model using multiple metrics from different tasks? E.g. macro F1 of main task and language identification task
            if aux_task == task_dev_metric:
                # Get best dev and train result (check out best_values_minimize and best_values_maximize if you modify these values)
                if best_dev_metric == "loss":
                    dev_target = dev_loss
                    train_target = epoch_loss
                elif best_dev_metric == "Macro-F1":
                    dev_target = dev_macro_f1 # Might be acc, loss, ...
                                            # We prefer macro over micro F1:
                                            #  https://datascience.stackexchange.com/questions/15989/micro-average-vs-macro-average-performance-in-a-multiclass-classification-settin#comment42550_24051
                    train_target = epoch_macro_f1 # It should be the same metric that dev_target
                elif best_dev_metric == "MCC":
                    dev_target = dev_mcc
                    train_target = epoch_mcc
                else:
                    raise Exception(f"Unknown best dev metric: {best_dev_metric}")

                if best_values_binary_func_comp(best_dev, dev_target) or (best_dev == dev_target and best_values_binary_func_comp(best_train, train_target)):
                    if best_dev == dev_target:
                        logger.debug("Dev is equal but train has been improved from %s to %s: checkpoint", str(best_train), str(train_target))
                    else:
                        logger.debug("Dev has been improved from %s to %s: checkpoint", str(best_dev), str(dev_target))

                    best_dev = dev_target

                    if best_values_binary_func_comp(best_train, train_target):
                        best_train = train_target

                    # Store model
                    if model_output:
                        model.save_model(model_output)

                    current_patience = 0
                else:
                    logger.debug("Dev has not been improved (best and current value): %s and %s", str(best_dev), str(dev_target))

                    current_patience += 1

                if plot:
                    utils.append_from_tuple((epoch_train_loss, epoch_loss),
                                            (epoch_train_acc, epoch_acc * 100.0),
                                            (epoch_train_acc_classes[0], epoch_f1_per_class[0] * 100.0),
                                            (epoch_train_acc_classes[1], epoch_f1_per_class[1] * 100.0),
                                            (epoch_train_macro_f1, epoch_macro_f1 * 100.0))
                    utils.append_from_tuple((epoch_dev_loss, dev_loss),
                                            (epoch_dev_acc, dev_acc * 100.0),
                                            (epoch_dev_acc_classes[0], dev_f1_per_class[0] * 100.0),
                                            (epoch_dev_acc_classes[1], dev_f1_per_class[1] * 100.0),
                                            (epoch_dev_macro_f1, dev_macro_f1 * 100.0))

                    plot_args = {
                        "show_statistics_every_batches": show_statistics_every_batches,
                        "batch_loss": batch_loss,
                        "batch_acc": batch_acc,
                        "batch_acc_classes": batch_acc_classes,
                        "batch_macro_f1": batch_macro_f1,
                        "epoch": epoch + 1,
                        "epoch_train_loss": epoch_train_loss,
                        "epoch_train_acc": epoch_train_acc,
                        "epoch_train_acc_classes": epoch_train_acc_classes,
                        "epoch_train_macro_f1": epoch_train_macro_f1,
                        "epoch_dev_loss": epoch_dev_loss,
                        "epoch_dev_acc": epoch_dev_acc,
                        "epoch_dev_acc_classes": epoch_dev_acc_classes,
                        "epoch_dev_macro_f1": epoch_dev_macro_f1,
                        "final_dev_acc": None,
                        "final_dev_macro_f1": None,
                        "final_test_acc": None,
                        "final_test_macro_f1": None,
                    }

                    plot_statistics(plot_args, path=args.plot_path)

        epoch += 1

        # Stop training?
        if patience > 0 and current_patience >= patience:
            # End of patience

            stop_training = True
        elif not train_until_patience:
            stop_training = epoch >= epochs

        # Load next shard, if any
        if multiple_shards and not stop_training:
            shard_id = epoch % len(filename_dataset_train)

            logger.info("Loading next shard: %d", shard_id)

            dataset_train, dataloader_train = \
                load_dataset(filename_dataset_train, "train", shard_id, **dataset_static_args)

    final_loss /= epoch
    final_acc /= epoch
    final_acc_per_class /= epoch
    final_f1_per_class /= epoch
    final_macro_f1 /= epoch
    final_mcc /= epoch

    logger.info("[train] Avg. loss: %f", final_loss)
    logger.info("[train] Avg. acc: %.2f %% (%s)",
                final_acc * 100.0, "; ".join([f"{_lang}: {final_acc_per_class[_id] * 100.0:.2f} %" for _lang, _id in _lang2id.items()]))
    logger.info("[train] Avg. values per class (f1): (%s)",
                "; ".join([f"{_lang}: {final_f1_per_class[_id] * 100.0:.2f} %" for _lang, _id in _lang2id.items()]))
    logger.info("[train] Avg. macro F1: %.2f %%", final_macro_f1 * 100.0)
    logger.info("[train] Avg. MCC: %.2f %%", final_mcc * 100.0)

    if do_not_load_best_model or not model_output:
        logger.warning("Using last model for dev and test evaluation")
    else:
        # Evaluate dev and test with best model

        logger.info("Loading best model (dev score): %s", str(best_dev))

        model = load_model(all_tasks, all_tasks_kwargs, model_input=model_output, pretrained_model=pretrained_model, device=device)

    dev_inference_metrics = inference(model, block_size, batch_size, all_tasks, tokenizer, criteria, dataset_dev,
                                      device, amp_context_manager, classes=classes, max_tokens=max_tokens)

    metrics_auxiliary_tasks = copy.deepcopy(all_tasks)

    metrics_auxiliary_tasks.remove(task_dev_metric) # Remove in order to be added as last element later

    # Dev metrics
    dev_loss = dev_inference_metrics["loss"]

    logger.info("[dev] Avg. loss: %f", dev_loss)

    for aux_task in metrics_auxiliary_tasks + [task_dev_metric]:
        dev_acc = dev_inference_metrics["metrics"][aux_task]["acc"]
        dev_acc_per_class = dev_inference_metrics["metrics"][aux_task]["acc_per_class"]
        dev_precision_per_class = dev_inference_metrics["metrics"][aux_task]["precision"]
        dev_recall_per_class = dev_inference_metrics["metrics"][aux_task]["recall"]
        dev_f1_per_class = dev_inference_metrics["metrics"][aux_task]["f1"]
        dev_macro_f1 = dev_inference_metrics["metrics"][aux_task]["macro_f1"]
        dev_mcc = dev_inference_metrics["metrics"][aux_task]["mcc"]

        logger.info("[dev] Acc (task '%s'): %.2f %% (%s)", aux_task,
                    dev_acc * 100.0, "; ".join([f"{_lang}: {dev_acc_per_class[_id] * 100.0:.2f} %" for _lang, _id in _lang2id.items()]))
        logger.info("[dev] Values per class (task '%s'; precision|recall|f1): "
                    "(%s | %s | %s)", aux_task,
                    "; ".join([f"{_lang}: {dev_precision_per_class[_id] * 100.0:.2f} %" for _lang, _id in _lang2id.items()]),
                    "; ".join([f"{_lang}: {dev_recall_per_class[_id] * 100.0:.2f} %" for _lang, _id in _lang2id.items()]),
                    "; ".join([f"{_lang}: {dev_f1_per_class[_id] * 100.0:.2f} %" for _lang, _id in _lang2id.items()]))
        logger.info("[dev] Macro F1 (task '%s'): %.2f %%", aux_task, dev_macro_f1 * 100.0)
        logger.info("[dev] MCC (task '%s'): %.2f %%", aux_task, dev_mcc * 100.0)

    test_inference_metrics = inference(model, block_size, batch_size, all_tasks, tokenizer, criteria, dataset_test,
                                       device, amp_context_manager, classes=classes, max_tokens=max_tokens)

    # Test metrics
    test_loss = test_inference_metrics["loss"]

    logger.info("[test] Avg. loss: %f", test_loss)

    for aux_task in metrics_auxiliary_tasks + [task_dev_metric]:
        test_acc = test_inference_metrics["metrics"][aux_task]["acc"]
        test_acc_per_class = test_inference_metrics["metrics"][aux_task]["acc_per_class"]
        test_precision_per_class = test_inference_metrics["metrics"][aux_task]["precision"]
        test_recall_per_class = test_inference_metrics["metrics"][aux_task]["recall"]
        test_f1_per_class = test_inference_metrics["metrics"][aux_task]["f1"]
        test_macro_f1 = test_inference_metrics["metrics"][aux_task]["macro_f1"]
        test_mcc = test_inference_metrics["metrics"][aux_task]["mcc"]

        logger.info("[test] Acc (task '%s'): %.2f %% (%s)", aux_task,
                    test_acc * 100.0, "; ".join([f"{_lang}: {test_acc_per_class[_id] * 100.0:.2f} %" for _lang, _id in _lang2id.items()]))
        logger.info("[test] Values per class (task '%s'; precision|recall|f1): "
                    "(%s | %s | %s)", aux_task,
                    "; ".join([f"{_lang}: {test_precision_per_class[_id] * 100.0:.2f} %" for _lang, _id in _lang2id.items()]),
                    "; ".join([f"{_lang}: {test_recall_per_class[_id] * 100.0:.2f} %" for _lang, _id in _lang2id.items()]),
                    "; ".join([f"{_lang}: {test_f1_per_class[_id] * 100.0:.2f} %" for _lang, _id in _lang2id.items()]))
        logger.info("[test] Macro F1 (task '%s'): %.2f %%", aux_task, test_macro_f1 * 100.0)
        logger.info("[test] MCC (task '%s'): %.2f %%", aux_task, test_mcc * 100.0)

    if plot:
        plot_args = {
            "show_statistics_every_batches": show_statistics_every_batches,
            "batch_loss": batch_loss,
            "batch_acc": batch_acc,
            "batch_acc_classes": batch_acc_classes,
            "batch_macro_f1": batch_macro_f1,
            # '"epoch": epoch' and not '"epoch": epoch + 1' because we have not added new values
            "epoch": epoch,
            "epoch_train_loss": epoch_train_loss,
            "epoch_train_acc": epoch_train_acc,
            "epoch_train_acc_classes": epoch_train_acc_classes,
            "epoch_train_macro_f1": epoch_train_macro_f1,
            "epoch_dev_loss": epoch_dev_loss,
            "epoch_dev_acc": epoch_dev_acc,
            "epoch_dev_acc_classes": epoch_dev_acc_classes,
            "epoch_dev_macro_f1": epoch_dev_macro_f1,
            "final_dev_acc": dev_acc,
            "final_dev_macro_f1": dev_macro_f1,
            "final_test_acc": test_acc,
            "final_test_macro_f1": test_macro_f1,
        }

        plot_statistics(plot_args, path=args.plot_path, freeze=True) # Let the user finish the execution if necessary

    if lock_file:
        # Create lock file since the training finished
        from pathlib import Path

        Path(lock_file).touch()

        logger.debug("Lock file created: %s", lock_file)

    logger.info("Done!")

def get_options_from_argv(argv_flag, default_value, dict_with_options):
    choices = list(dict_with_options.keys())
    args_options = dict_with_options[default_value]

    if argv_flag in sys.argv:
        idx = sys.argv.index(argv_flag)

        if len(sys.argv) > idx + 1:
            value = sys.argv[idx + 1]

            if value in choices:
                args_options = dict_with_options[value]

    result = {
        "default": default_value,
        "choices": choices,
        "options": args_options,
    }

    return result

def initialization():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="URL2Lang")
    inference = "--inference" in sys.argv
    lr_scheduler_conf = get_options_from_argv("--lr-scheduler", "inverse_sqrt", _lr_scheduler_args)
    optimizer_conf = get_options_from_argv("--optimizer", "adamw", _optimizer_args)

    if not inference:
        parser.add_argument('dataset_train_filename', type=str,
                            help="Filename with train data (TSV format). You can provide multiple files separated using ':' and "
                                 "each of them will be used one for each epoch using round-robin strategy")
        parser.add_argument('dataset_dev_filename', type=str, help="Filename with dev data (TSV format)")
        parser.add_argument('dataset_test_filename', type=str, help="Filename with test data (TSV format)")

    parser.add_argument('--batch-size', type=int, default=16,
                        help="Batch size. Elements which will be processed before proceed to train, but the whole batch will "
                             "be processed in blocks in order to avoid OOM errors")
    parser.add_argument('--block-size', type=int, help="Block size. Elements which will be provided to the model at once")
    parser.add_argument('--max-tokens', type=int, default=-1,
                        help="Process batches in groups tokens size (fairseq style). "
                             "Batch size is still relevant since the value is used when batches are needed (e.g. sampler from dataset)")
    parser.add_argument('--epochs', type=int, default=3, help="Epochs")
    parser.add_argument('--do-not-fine-tune', action="store_true", help="Do not apply fine-tuning to the base model (default weights)")
    parser.add_argument('--freeze-whole-model', action="store_true", help="Do not apply fine-tuning to the whole model, not only the base model")
    parser.add_argument('--dataset-workers', type=int, default=-1,
                        help="No. workers when loading the data in the dataset. When negative, all available CPUs will be used")
    parser.add_argument('--pretrained-model', default="xlm-roberta-base", help="Pretrained model")
    parser.add_argument('--max-length-tokens', type=int, default=256, help="Max. length for the generated tokens")
    parser.add_argument('--model-input', help="Model input path which will be loaded")
    parser.add_argument('--model-output', help="Model output path where the model will be stored")
    parser.add_argument('--inference', action="store_true",
                        help="Do not train, just apply inference (flag --model-input is recommended). "
                             "If this option is set, it will not be necessary to provide the input dataset")
    parser.add_argument('--inference-from-stdin', action="store_true", help="Read inference from stdin")
    parser.add_argument('--parallel-likelihood', action="store_true",
                        help="Print parallel likelihood instead of classification string (inference)")
    parser.add_argument('--threshold', type=float, default=-np.inf,
                        help="Only print URLs which have a parallel likelihood greater than the provided threshold (inference)")
    parser.add_argument('--imbalanced-strategy', type=str, choices=["none", "over-sampling", "weighted-loss"], default="none",
                        help="Strategy for dealing with imbalanced data")
    parser.add_argument('--patience', type=int, default=0, help="Patience before stopping the training")
    parser.add_argument('--train-until-patience', action="store_true",
                        help="Train until patience value is reached (--epochs will be ignored in order to stop, but will still be "
                             "used for other actions like LR scheduler)")
    parser.add_argument('--do-not-load-best-model', action="store_true",
                        help="Do not load best model for final dev and test evaluation (--model-output is necessary)")
    parser.add_argument('--overwrite-output-model', action="store_true", help="Overwrite output model if it exists (initial loading)")
    parser.add_argument('--remove-authority', action="store_true", help="Remove protocol and authority from provided URLs")
    parser.add_argument('--remove-positional-data-from-resource', action="store_true",
                        help="Remove content after '#' in the resorce "
                             "(e.g. https://www.example.com/resource#position -> https://www.example.com/resource)")
    parser.add_argument('--force-cpu', action="store_true", help="Run on CPU (i.e. do not check if GPU is possible)")
    parser.add_argument('--log-directory', help="Directory where different log files will be stored")
    parser.add_argument('--regression', action="store_true",
                        help="Apply regression instead of multiclass classification. Language will not be detected but checked out")
    parser.add_argument('--url-separator', default='/', help="Separator to use when URLs are stringified")
    parser.add_argument('--url-separator-new-token', action="store_true", help="Add special token for URL separator")
    parser.add_argument('--learning-rate', type=float, default=1e-5, help="Learning rate")
    parser.add_argument('--optimizer', choices=optimizer_conf["choices"], default=optimizer_conf["default"], help="Optimizer")
    parser.add_argument('--optimizer-args', **optimizer_conf["options"],
                        help="Args. for the optimizer (in order to see the specific configuration for a optimizer, use -h and set --optimizer)")
    parser.add_argument('--lr-scheduler', choices=lr_scheduler_conf["choices"], default=lr_scheduler_conf["default"], help="LR scheduler")
    parser.add_argument('--lr-scheduler-args', **lr_scheduler_conf["options"],
                        help="Args. for LR scheduler (in order to see the specific configuration for a LR scheduler, "
                             "use -h and set --lr-scheduler)")
    parser.add_argument('--re-initialize-last-n-layers', type=int, default=1,
                        help="Re-initialize last N layers from pretained model (will be applied only when fine-tuning the model)")
    parser.add_argument('--cuda-amp', action="store_true", help="Use CUDA AMP (Automatic Mixed Precision)")
    parser.add_argument('--llrd', action="store_true", help="Apply LLRD (Layer-wise Learning Rate Decay)")
    parser.add_argument('--stringify-instead-of-tokenization', action="store_true",
                        help="Preprocess URLs applying custom stringify instead of tokenization")
    parser.add_argument('--lowercase', action="store_true", help="Lowercase URLs while preprocessing")
    parser.add_argument('--auxiliary-tasks', type=str, nargs='*', choices=["mlm"],
                        help="Tasks which will try to help to the main task (multitasking)")
    parser.add_argument('--auxiliary-tasks-weights', type=float, nargs='*',
                        help="Weights for the loss of the auxiliary tasks. If none is provided, the weights will be 1, "
                             "but if any is provided, as many weights as auxiliary tasks will have to be provided")
    parser.add_argument('--freeze-embeddings-layer', action="store_true", help="Freeze embeddings layer")
    parser.add_argument('--remove-instead-of-truncate', action="store_true",
                        help="Remove pairs of URLs which would need to be truncated (if not enabled, truncation will be applied). "
                             "This option will be only applied to the training set")
    parser.add_argument('--best-dev-metric', default="Macro-F1", choices=["loss", "Macro-F1", "MCC"],
                        help="Which metric should be maximized or minimized when dev is being evaluated in order to save the best model")
    parser.add_argument('--task-dev-metric', default="language-identification",
                        choices=["language-identification"],
                        help="Task which will be used in order to save the best model. It will also be used in order to replace the main "
                             "task if --do-not-train-main-task is set")
    parser.add_argument('--auxiliary-tasks-flags', type=str, nargs='*',
                        choices=[],
                        help="Set of options which will set up some aspects of the auxiliary tasks")
    parser.add_argument('--pre-load-shards', action="store_true",
                        help="Load all shards at beginning one by one in order to get some statistics needed for some features. This "
                             "option is optional, but if not set, some features might not work as expected (e.g. linear LR scheduler)")

    parser.add_argument('--seed', type=int, default=71213,
                        help="Seed in order to have deterministic results (not fully guaranteed). "
                             "Set a negative number in order to disable this feature")
    parser.add_argument('--plot', action="store_true", help="Plot statistics (matplotlib pyplot) in real time")
    parser.add_argument('--plot-path', help="If set, the plot will be stored instead of displayed")
    parser.add_argument('--lock-file',
                        help="If set, and the file does not exist, it will be created once the training finishes. "
                             "If does exist, the training will not be executed")
    parser.add_argument('--waiting-time', type=int, default=20, help="Waiting time, if needed for letting the user react")


    parser.add_argument('-v', '--verbose', action="store_true", help="Verbose logging mode")

    args = parser.parse_args()

    return args

def fix_log_directory(args):
    log_directory = args.log_directory
    waiting_time = args.waiting_time

    if not log_directory:
        log_directory = tempfile.mkdtemp(prefix=f"u2l_{datetime.now().strftime('%Y%m%d%H%M%S')}_")

    if not utils.exists(log_directory, f=os.path.isdir):
        raise Exception(f"Provided log directory does not exist: '{log_directory}'")
    else:
        logger.info("Log directory: %s", log_directory)

        log_directory_files = os.listdir(utils.resolve_path(log_directory))

        if len(log_directory_files) != 0:
            logger.warning("Log directory contain %d files: waiting %d seconds before proceed", len(log_directory_files), waiting_time)

            time.sleep(waiting_time)

    args.log_directory = log_directory

def cli():
    global logger
    global logger_verbose

    # https://stackoverflow.com/questions/16549332/python-3-how-to-specify-stdin-encoding
    sys.stdin.reconfigure(encoding='utf-8', errors="backslashreplace")

    args = initialization()

    # Logging
    logger = utils.set_up_logging_logger(logger, level=logging.DEBUG if args.verbose else logging.INFO)

    logger.debug("Arguments processed: {}".format(str(args))) # First logging message should be the processed arguments

    # Verbose loggers
    logger_verbose["tokens"] = logging.getLogger("url2lang.tokens")
    logger_verbose["tokens"].propagate = False

    fix_log_directory(args) # We are going to use args.log_directory, so fix it if needed

    logger_verbose["tokens"] = utils.set_up_logging_logger(logger_verbose["tokens"], level=logging.DEBUG if args.verbose else logging.INFO,
                                                           filename=f"{args.log_directory}/tokens", format="%(asctime)s\t%(levelname)s\t%(message)s")

    main(args)

if __name__ == "__main__":
    cli()

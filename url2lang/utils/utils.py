
import os
import copy
import psutil
import logging
import gzip
import lzma
from contextlib import contextmanager
import argparse

import torch

logger = logging.getLogger("url2lang")

def wc_l(fd, do_not_count_empty=True):
    no_lines = 0
    tell = fd.tell()

    for line in fd:
        if do_not_count_empty and line.strip() == '':
            continue

        no_lines += 1

    # Restore initial status of the fd
    fd.seek(tell)

    return no_lines

def get_layer_from_model(layer, name=None, deepcopy=True):
    could_get_layer = False

    # Get layer from model (we need to do it with a for loop since it is a generator which cannot be accessed with idx)
    for last_layer_name, last_layer_param in layer.named_parameters():
        if last_layer_name == name:
            could_get_layer = True

            break

    if name is not None:
        assert could_get_layer, f"Could not get the layer '{name}'"

    last_layer_param_data = last_layer_param.data

    if deepcopy:
        # Return a deepcopy instead of the value itself to avoid affect the model if modified

        return copy.deepcopy(last_layer_param_data)

    return last_layer_param_data

def encode(tokenizer, text, max_length=512, add_special_tokens=True, padding="do_not_pad", return_attention_mask=False,
           return_tensors="pt", truncation=True):
    encoder = tokenizer.batch_encode_plus if isinstance(text, list) else tokenizer.encode_plus

    return encoder(text, add_special_tokens=add_special_tokens, truncation=truncation, padding=padding,
                   return_attention_mask=return_attention_mask, return_tensors=return_tensors, max_length=max_length)

def apply_model(model, tokenizer, tokens, encode=False):
    if encode:
        tokens = encode(tokenizer, tokens)

        output = model(**tokens)
    else:
        output = model(tokens)

    #input_ids = tokenized["input_ids"]
    #token_type_ids = tokenized["token_type_ids"]
    #attention_mask = tokenized["attention_mask"]

    #sentence_length = torch.count_nonzero(attention_mask).to("cpu").numpy()
    #tokens = input_ids[0,:sentence_length]

    return output

# TODO take into account --regression flag in order to add the language to the input model
#  Pro using lang+url in input: we can apply 0 shot learning
def tokenize_batch_from_iterator(iterator, tokenizer, batch_size, f=None, return_urls=False,
                                 auxiliary_tasks=[], inference=False):
    import url2lang.url2lang as url2lang

    def reset():
        urls = {
            "urls": [],
            "labels_task_language_identification": [],
        }
        initial_urls = []

        return urls, initial_urls

    urls, initial_urls = reset()

    for idx, url in enumerate(iterator, 1):
        url = url.strip().split('\t')
        initial_url = url[0]

        if inference:
            # Format:
            #  2: url true_lang[ignored]
            if len(url) not in (1, 2):
                raise Exception(f"Expected lengths are 1 or 2 but got {len(url)}")
        else:
            if len(url) != 2:
                raise Exception(f"It was expected 2 values per line (url, true_lang), but got {len(url)} values")

        if f:
            src_url = f(url[0])

            if isinstance(src_url, list):
                if len(src_url) != 1:
                    raise Exception(f"Unexpected size of list after applying function to URL: {len(src_url)}")

                src_url = src_url[0]
        else:
            src_url = url[0]

        if inference and len(url) in (1,):
            target_output = -1 # We don't know the result since inference=True
        else:
            true_lang = url[1]

            if true_lang in url2lang._lang2id.keys():
                target_output = url2lang._lang2id[true_lang]
            else:
                logger.warning("URL skipped since its language (%s) is not supported: %s", true_lang, initial_url)

                continue

        if tokenizer.sep_token in src_url:
            logger.warning("URL skipped since they contain the separator token: %s", initial_url)

            continue

        initial_urls.append([initial_url])
        urls["urls"].append(f"{src_url}")
        urls["labels_task_language_identification"].append(target_output)

        if len(urls["urls"]) >= batch_size:
            if return_urls:
                yield urls, initial_urls
            else:
                yield urls

            urls, initial_urls = reset()

    if len(urls["urls"]) != 0:
        if return_urls:
            yield urls, initial_urls
        else:
            yield urls

        urls, initial_urls = reset()

def get_current_allocated_memory_size():
    process = psutil.Process(os.getpid())
    size_in_bytes = process.memory_info().rss

    return size_in_bytes

def set_up_logging_logger(logger, filename=None, level=logging.INFO, format="[%(asctime)s] [%(name)s] [%(levelname)s] [%(module)s:%(lineno)d] %(message)s",
                          display_when_file=False):
    handlers = [
        logging.StreamHandler()
    ]

    if filename is not None:
        if display_when_file:
            # Logging messages will be stored and displayed
            handlers.append(logging.FileHandler(filename))
        else:
            # Logging messages will be stored and not displayed
            handlers[0] = logging.FileHandler(filename)

    formatter = logging.Formatter(format)

    for h in handlers:
        h.setFormatter(formatter)
        logger.addHandler(h)

    logger.setLevel(level)

    logger.propagate = False # We don't want to see the messages multiple times

    return logger

def append_from_tuple(*tuples):
    """We expect tuples where:
        - First element: list
        - Second element: value to be inserted in the list of the first component of the tuple
    """
    for l, v in tuples:
        l.append(v)

@contextmanager
def open_xz_or_gzip_or_plain(file_path, mode='rt'):
    f = None
    try:
        if file_path[-3:] == ".gz":
            f = gzip.open(file_path, mode)
        elif file_path[-3:] == ".xz":
            f = lzma.open(file_path, mode)
        else:
            f = open(file_path, mode)
        yield f

    except Exception:
        raise Exception("Error occurred while loading a file!")

    finally:
        if f:
            f.close()

def resolve_path(p):
    result = os.path.realpath(os.path.expanduser(p)) if isinstance(p, str) else p

    return result.rstrip('/') if result else result

def exists(p, res_path=False, f=os.path.isfile):
    return f(resolve_path(p) if res_path else p) if isinstance(p, str) else False

def replace_multiple(s, replace_list, replace_char=' '):
    for original_char in replace_list:
        s = s.replace(original_char, replace_char)
    return s

def set_up_logging(filename=None, level=logging.INFO, format="[%(asctime)s] [%(levelname)s] [%(module)s:%(lineno)d] %(message)s",
                   display_when_file=False):
    handlers = [
        logging.StreamHandler()
    ]

    if filename is not None:
        if display_when_file:
            # Logging messages will be stored and displayed
            handlers.append(logging.FileHandler(filename))
        else:
            # Logging messages will be stored and not displayed
            handlers[0] = logging.FileHandler(filename)

    logging.basicConfig(handlers=handlers, level=level,
                        format=format)

def update_defined_variables_from_dict(d, provided_locals, smash=False):
    for v, _ in d.items():
        if v in provided_locals and not smash:
            raise Exception(f"Variable '{v}' is already defined and smash=False")

    provided_locals.update(d)

def init_weight_and_bias(model, module):
    if isinstance(module, torch.nn.Linear):
        module.weight.data.normal_(mean=0.0, std=model.config.initializer_range)

        if module.bias is not None:
            module.bias.data.zero_()

    elif isinstance(module, torch.nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

def do_reinit(model, reinit_n_layers):
    try:
        # Re-init pooler.
        model.pooler.dense.weight.data.normal_(mean=0.0, std=model.config.initializer_range)
        model.pooler.dense.bias.data.zero_()

        for param in model.pooler.parameters():
            param.requires_grad = True
    except AttributeError:
        pass

    # Re-init last n layers.
    for n in range(reinit_n_layers):
        model.encoder.layer[-(n+1)].apply(lambda module: init_weight_and_bias(model, module))

def argparse_nargs_type(*types):
    def f(arg):
        t = types[f._invoked]
        choices = None

        if isinstance(t, dict):
            choices = t["choices"]
            t = t["type"]

        if not isinstance(arg, t):
            type_arg = type(arg)

            try:
                arg = t(arg) # Cast from str
            except:
                raise argparse.ArgumentTypeError(f"Arg. #{f._invoked + 1} is not instance of {str(t)}, but {str(type_arg)}")
        elif choices is not None:
            if arg not in choices:
                raise argparse.ArgumentTypeError(f"Arg. #{f._invoked + 1} invalid value: value not in {str(choices)}")

        f._invoked += 1
        return arg

    f._invoked = 0

    return f

def get_model_parameters_applying_llrd(model, learning_rate, weight_decay=0.01):
    # This function expects the optimizer to support weight_decay

    # Based on https://gist.github.com/peggy1502/5775b0b246ef5a64a9cf7b58cd722baf#file-readability_llrd-py from https://towardsdatascience.com/advanced-techniques-for-fine-tuning-transformers-82e4e61e16e#6196
    opt_parameters = []    # To be passed to the optimizer (only parameters of the layers you want to update).
    # Obtain parameters of heads and base model separately in order to don't update the base model multiple times
    named_parameters_base_model = list(model.get_base_model().named_parameters())
    named_parameters_heads = []

    for task in model.get_tasks_names():
        named_parameters_heads += list(model.get_head(task).named_parameters())

    # According to AAAMLP book by A. Thakur, we generally do not use any decay
    # for bias and LayerNorm.weight layers.
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    lr = learning_rate

    # === Pooler and regressor ======================================================

    params_0 = [p for n,p in named_parameters_heads if ("pooler" in n or "regressor" in n or "classifier" in n)
                and any(nd in n for nd in no_decay)]
    params_1 = [p for n,p in named_parameters_heads if ("pooler" in n or "regressor" in n or "classifier" in n)
                and not any(nd in n for nd in no_decay)]

    head_params = {"params": params_0, "lr": lr, "weight_decay": 0.0}
    opt_parameters.append(head_params)

    head_params = {"params": params_1, "lr": lr, "weight_decay": weight_decay}
    opt_parameters.append(head_params)

    lr *= 0.95

    # === 12 Hidden layers ==========================================================

    for layer in range(11,-1,-1):
        params_0 = [p for n,p in named_parameters_base_model if f"encoder.layer.{layer}." in n
                    and any(nd in n for nd in no_decay)]
        params_1 = [p for n,p in named_parameters_base_model if f"encoder.layer.{layer}." in n
                    and not any(nd in n for nd in no_decay)]

        layer_params = {"params": params_0, "lr": lr, "weight_decay": 0.0}
        opt_parameters.append(layer_params)

        layer_params = {"params": params_1, "lr": lr, "weight_decay": weight_decay}
        opt_parameters.append(layer_params)

        lr *= 0.95 # Based on https://arxiv.org/pdf/1905.05583.pdf

    # === Embeddings layer ==========================================================

    params_0 = [p for n,p in named_parameters_base_model if "embeddings" in n
                and any(nd in n for nd in no_decay)]
    params_1 = [p for n,p in named_parameters_base_model if "embeddings" in n
                and not any(nd in n for nd in no_decay)]

    embed_params = {"params": params_0, "lr": lr, "weight_decay": 0.0}
    opt_parameters.append(embed_params)

    embed_params = {"params": params_1, "lr": lr, "weight_decay": weight_decay}
    opt_parameters.append(embed_params)

    return opt_parameters

def get_tuple_if_is_not_tuple(obj, check_not_list=True):
    if not isinstance(obj, tuple):
        if check_not_list:
            if not isinstance(obj, list):
                return (obj,)
            else:
                return tuple(obj)
        else:
            return (obj,)

    return obj

def get_idx_after_protocol(url):
    idx = 0

    if url.startswith("http://"):
        idx += 7
    elif url.startswith("https://"):
        idx += 8
    else:
        # Other protocols

        idx = url.find("://")

        if idx == -1:
            # No "after" protocol found
            logger.warning("Protocol not found for the provided URL: %s", url)

            return 0

        idx += 3 # Sum len("://")

    return idx

def get_idx_resource(url, url_has_protocol=True):
    if url_has_protocol:
        idx = get_idx_after_protocol(url)

    idx = url.find('/', idx)

    if idx == -1:
        return len(url) # There is no resource (likely is just the authority, i.e., main resource of the website)

    return idx + 1

def get_data_from_batch(batch, block_size, device):
    urls = batch["url_tokens"]
    attention_mask = batch["url_attention_mask"]
    labels = batch["labels_task_language_identification"]

    # Split in batch_size batches
    start = 0
    current_batch_size = labels.reshape(-1).shape[0]
    end = start + (block_size if block_size else current_batch_size) # Return the whole batch if the block size was not provided

    while True:
        if start < end:
            _urls = urls[start:end].to(device)
            _attention_mask = attention_mask[start:end].to(device)
            _labels = labels[start:end].to(device)

            # Create dictionary with inputs and outputs
            inputs_and_outputs = {
                "labels_task_language_identification": _labels,
                "urls": _urls,
                "attention_mask": _attention_mask,
            }

            yield inputs_and_outputs

            start = end
            end = min(start + (block_size if block_size else current_batch_size), current_batch_size)
        else:
            break

def get_pytorch_version():
    try:
        torch_version = list(map(int, torch.__version__.split('+')[0].split('.')))
    except Exception as e:
        logger.error("%s", str(e))
        logger.error("Unexpected exception: returning -1.-1.-1 as torch version")

        return -1, -1, -1

    assert len(torch_version) == 3, f"Torch version is expected to be X.Y.Z, but got {'.'.join(torch_version)}"

    torch_version_major, torch_version_minor, torch_version_patch = torch_version

    return torch_version_major, torch_version_minor, torch_version_patch

def use_cuda(force_cpu=False):
    use_cuda = torch.cuda.is_available()

    return True if use_cuda and not force_cpu else False

def check_nltk_model(model_path, model, download=True, quiet=False):
    import nltk

    try:
        nltk.data.find(model_path)
    except LookupError:
        logger.info("NLTK model not available: %s", model)

        if download:
            logger.info("Downloading model: %s", model)

            nltk.download(model, quiet=quiet)

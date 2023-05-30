
import sys
import random
import logging

import url2lang.utils.utils as utils
from url2lang.metrics import (
    get_metrics,
    get_metrics_task_specific,
)
import url2lang.preprocess as preprocess

import numpy as np
import torch
#import torch.nn.functional as F
import transformers

logger = logging.getLogger("url2lang")
logger_tokens = logging.getLogger("url2lang.tokens")

def inference_with_heads(model, tasks, tokenizer, inputs_and_outputs, amp_context_manager,
                         tasks_weights=None, criteria=None, device=None):
    results = {"_internal": {"total_loss": None}}

    # Inputs and outputs
    urls = inputs_and_outputs["urls"]
    attention_mask = inputs_and_outputs["attention_mask"]
    labels = {}

    def apply_mlm():
        # "Mask" labels and mask URLs
        #  https://discuss.huggingface.co/t/bertformaskedlm-s-loss-and-scores-how-the-loss-is-computed/607/2
        _urls, _labels = transformers.DataCollatorForLanguageModeling(tokenizer, mlm_probability=0.15)\
                            .torch_mask_tokens(urls.clone().cpu().detach())

        return _urls, _labels

    with amp_context_manager:
        model_outputs = None

        if "language-identification" in tasks:
            if criteria:
                labels["language-identification"] = inputs_and_outputs["labels_task_language_identification"]
        if "mlm" in tasks:
            # MLM is applied at the same time with all the other tasks

            _urls, _labels = apply_mlm()
            urls = _urls # Replace since now there are masked tokens

            if criteria:
                labels["mlm"] = _labels

        # Move to device
        urls = urls.to(device)
        attention_mask = attention_mask.to(device)
        encoder_output = None

        for head_task in tasks:
            if criteria:
                # Move to device
                labels[head_task] = labels[head_task].to(device)

            # Inference
            model_outputs = model(head_task, urls, attention_mask, encoder_output=encoder_output)
            outputs = model_outputs["logits"] # Get head result
            encoder_output = model_outputs["encoder_output"]
            criterion = criteria[head_task] if criteria else None
            loss_weight = tasks_weights[head_task] if tasks_weights else 1.0

            # Calcule loss
            if head_task in ("language-identification",):
                regression = outputs.cpu().detach().numpy().shape[-1] == 1

                if regression:
                    # Regression
                    #outputs = torch.sigmoid(outputs).squeeze(1)
                    outputs = outputs.squeeze(1)
                    # TODO use threshold instead of torch.round
                    outputs_classification = torch.round(torch.sigmoid(outputs)).type(torch.int64).cpu() # Workaround for https://github.com/pytorch/pytorch/issues/54774
                else:
                    # Binary classification
                    #outputs = F.softmax(outputs, dim=1)
                    outputs_classification = torch.argmax(outputs.cpu(), dim=1)

                if criterion:
                    loss = criterion(outputs, labels[head_task])
                    loss *= loss_weight

                results[head_task] = {
                    "outputs": outputs,
                    "outputs_classification": outputs_classification,
                    "loss_detach": loss.cpu().detach() if criterion else None,
                    "regression": regression,
                }
            elif head_task == "mlm":
                if criterion:
                    loss = criterion(outputs.view(-1, tokenizer.vocab_size), labels[head_task].view(-1))
                    loss *= loss_weight

                results["mlm"] = {
                    "outputs": outputs,
                    "loss_detach": loss.cpu().detach() if criterion else None,
                }
            else:
                raise Exception(f"Unknown head task: {head_task}")

            # Sum loss (we don't want to define multiple losses at the same time in order to avoid high memory allocation)
            if criterion:
                if results["_internal"]["total_loss"] is None:
                    results["_internal"]["total_loss"] = loss
                else:
                    results["_internal"]["total_loss"] += loss
            else:
                results["_internal"]["total_loss"] = None

            if criteria and len(tasks) > 1:
                # Move data to CPU (it will free up memory if device is cuda)
                labels[head_task] = labels[head_task].cpu()

                #torch.cuda.empty_cache() # https://discuss.pytorch.org/t/how-to-delete-a-tensor-in-gpu-to-free-up-memory/48879
                                          # Inference is slower and it is not needed in order to decrease the total GPU memory used

        if len(tasks) > 1:
            # Move data to CPU (it will free up memory if device is cuda)
            urls = urls.cpu()
            attention_mask = attention_mask.cpu()

            #torch.cuda.empty_cache() # https://discuss.pytorch.org/t/how-to-delete-a-tensor-in-gpu-to-free-up-memory/48879

    return results

@torch.no_grad()
def inference(model, block_size, batch_size, tasks, tokenizer, criteria, dataset, device,
              amp_context_manager, classes=2, max_tokens=None):
    model.eval()

    total_loss = 0.0
    all_outputs = {task: [] for task in tasks}
    all_labels = {task: [] for task in tasks}
    total_blocks_per_batch = 1 if max_tokens else max(int(np.ceil(batch_size / block_size)), 1)
    total_tokens = 0
    total_tokens_with_padding = 0
    dataloader = dataset.dataloader

    for idx, batch in enumerate(dataloader):
        if max_tokens and batch is None:
            # Batch is under construction using max_tokens...
            continue

        for inputs_and_outputs in utils.get_data_from_batch(batch, None if max_tokens else block_size, None):
            total_tokens += sum([len(urls[urls != tokenizer.pad_token_id]) for urls in inputs_and_outputs["urls"]])
            total_tokens_with_padding += sum([len(urls) for urls in inputs_and_outputs["urls"]])

            # Inference
            results = inference_with_heads(model, tasks, tokenizer, inputs_and_outputs, amp_context_manager, criteria=criteria,
                                           device=device)

            # Tasks
            for task in tasks:
                if task in ("language-identification",):
                    loss_task = results[task]["loss_detach"] # TODO propagate somehow? Statistics?
                    outputs_classification = results[task]["outputs_classification"].cpu()
                    regression = results[task]["regression"]

                    if task == "language-identification":
                        labels = inputs_and_outputs["labels_task_language_identification"]
                    else:
                        raise Exception(f"Unknown/not supported task: {task}")

                    labels = labels.cpu()

                    if regression:
                        labels = torch.round(labels).type(torch.long)

                    all_outputs[task].extend(outputs_classification.tolist())
                    all_labels[task].extend(labels.tolist())
                elif task == "mlm":
                    # TODO propagate somehow? Statistics?
                    loss_mlm = results["mlm"]["loss_detach"]
                    outputs_mlm = results["mlm"]["outputs"].cpu()
                else:
                    raise Exception(f"Unknown task: {task}")

            loss = results["_internal"]["total_loss"].cpu()
            loss = loss.cpu() / total_blocks_per_batch

            total_loss += loss

    if total_tokens != dataset.total_tokens:
        logger.error("Total processed tokens are different from the initial total tokens: %d vs %d",
                     total_tokens, dataset.total_tokens)

    for task in tasks:
        all_outputs[task] = torch.as_tensor(all_outputs[task])
        all_labels[task] = torch.as_tensor(all_labels[task])

    metrics = {task: {} for task in tasks}
    total_loss /= idx + 1

    for task in tasks:
        if task == "mlm":
            continue

        metrics[task] = get_metrics_task_specific(task, all_outputs[task], all_labels[task], len(all_labels[task]), classes=classes)

    return {
        "loss": total_loss,
        "metrics": metrics,
    }

@torch.no_grad()
def interactive_inference(model, tokenizer, batch_size, max_length_tokens, device, amp_context_manager,
                          inference_from_stdin=False, remove_authority=False, remove_positional_data_from_resource=False,
                          parallel_likelihood=False, threshold=-np.inf, url_separator=' ', lower=True,
                          auxiliary_tasks=[], auxiliary_tasks_flags=[]):
    logger.info("Inference mode enabled")

    for aux_task in auxiliary_tasks:
        if aux_task in (): # Supported auxiliary tasks
            pass
        else:
            raise Exception(f"Not supported or unknown task: {aux_task}")

    if not inference_from_stdin:
        logger.info("Insert blank lines in order to end")

    logger_tokens.debug("preprocessed_urls\tmodel_input\ttokens\ttokens2str\tunk_chars\t"
                        "initial_tokens_vs_detokenized\tinitial_tokens_vs_detokenized_len_1")

    model.eval()

    all_tasks = ["language-identification"] + auxiliary_tasks

    while True:
        if inference_from_stdin:
            try:
                target_urls, initial_urls = \
                    next(utils.tokenize_batch_from_iterator(sys.stdin, tokenizer, batch_size,
                            f=lambda u: preprocess.preprocess_url(u, remove_protocol_and_authority=remove_authority,
                                                                  remove_positional_data=remove_positional_data_from_resource,
                                                                  separator=url_separator, lower=lower),
                            return_urls=True, auxiliary_tasks=auxiliary_tasks, inference=True))

            except StopIteration:
                break

            initial_src_urls = [u[0] for u in initial_urls]
        else:
            initial_src_urls = [input("url: ").strip()]
            src_url_lang = input("url lang: ").strip()

            if not initial_src_urls[0]:
                break

            src_url = initial_src_urls[0]
            data = f"{src_url}\t{src_url_lang}"
            data = [data]
            target_urls = next(utils.tokenize_batch_from_iterator(data, tokenizer, batch_size,
                               f=lambda u: preprocess.preprocess_url(u, remove_protocol_and_authority=remove_authority,
                                                                     remove_positional_data=remove_positional_data_from_resource,
                                                                     separator=url_separator, lower=lower),
                               auxiliary_tasks=auxiliary_tasks, inference=True))

        target_urls = target_urls["urls"]

        # Tokens
        tokens = utils.encode(tokenizer, target_urls, max_length_tokens, padding="longest", return_attention_mask=True)
        urls = tokens["input_ids"].to(device)
        attention_mask = tokens["attention_mask"].to(device)

        # Debug info
        ## Tokens
        urls_tokens = urls.cpu()

        for idx, ut in enumerate(urls_tokens):
            url_tokens = ut[ut != tokenizer.pad_token_id]
            original_str_from_tokens = tokenizer.decode(url_tokens) # Detokenize
            str_from_tokens = '<tok_sep>'.join([tokenizer.decode(t) for t in url_tokens]) # Detokenize adding a mark between tokens
            ## Unk
            unk = torch.sum((url_tokens == tokenizer.unk_token_id).int()) # Unk tokens (this should happen just with very strange chars)
            sp_unk_vs_tokens_len = f"{len(original_str_from_tokens.split(url_separator))} vs " \
                                   f"{len(str_from_tokens.split(url_separator))}"
            sp_unk_vs_one_len_tokens = f"{sum(map(lambda u: 1 if len(u) == 1 else 0, original_str_from_tokens.split(url_separator)))} vs " \
                                       f"{sum(map(lambda u: 1 if len(u) == 1 else 0, str_from_tokens.split(url_separator)))}"

            logger_tokens.debug("%s\t%s\t%s\t%s\t%d\t%s\t%s", target_urls[idx], original_str_from_tokens,
                                                              str(url_tokens).replace('\n', ' '), str_from_tokens, unk,
                                                              sp_unk_vs_tokens_len, sp_unk_vs_one_len_tokens)

        # Inference
        results = inference_with_heads(model, all_tasks, tokenizer, {"urls": urls, "attention_mask": attention_mask},
                                       amp_context_manager)

        # Get results of each task
        for task in all_tasks:
            outputs = results[task]["outputs"].cpu()
            outputs_argmax = results[task]["outputs_classification"]

            if task in ("language-identification",):
                outputs = torch.sigmoid(outputs)

            #if len(outputs_argmax.shape) == 0:
            #    outputs_argmax = np.array([outputs_argmax])

            if outputs.numpy().shape[0] != len(initial_src_urls):
                raise Exception("Output samples does not match with the length of src URLs "
                                f"({outputs.numpy().shape[0]} vs {len(initial_src_urls)})")

            if parallel_likelihood:
                for data, initial_src_url in zip(outputs.numpy(), initial_src_urls):
                    if task in ("language-identification",):
                        regression = results[task]["regression"]
                        likelihood = data if regression else data[1] # parallel

                        if likelihood >= threshold:
                            print(f"{task}\t{likelihood:.4f}\t{initial_src_url}")
                    else:
                        print(f"{task}\t{data}\t{initial_src_url}")
            else:
                for argmax, initial_src_url in zip(outputs_argmax, initial_src_urls):
                    if task in ("language-identification",):
                        print(f"{task}\t{'positive' if argmax == 1 else 'negative'}\t{initial_src_url}")
                    else:
                        print(f"{task}\t{argmax}\t{initial_src_url}")

@torch.no_grad()
def non_interactive_inference(model, tokenizer, batch_size, max_length_tokens, device, amp_context_manager,
                              src_urls, src_urls_lang, remove_authority=False, remove_positional_data_from_resource=False,
                              parallel_likelihood=False, threshold=-np.inf, url_separator=' ', lower=False,
                              auxiliary_tasks=[], auxiliary_tasks_flags=[]):
    model.eval()
    all_results = {}

    for aux_task in auxiliary_tasks:
        if aux_task in (): # Supported auxiliary tasks
            pass
        else:
            raise Exception(f"Not supported or unknown task: {aux_task}")

    all_tasks = ["language-identification"] + auxiliary_tasks

    for task in all_tasks:
        all_results[task] = []

    if len(src_urls_lang) != len(src_urls):
        raise Exception(f"Unexpected different lengths: {len(src_urls_lang)} vs {len(src_urls)}")

    # Process URLs
    src_urls = [src_url.replace('\t', ' ') for src_url in src_urls]
    str_urls = [f"{src_url}\t{src_urls_lang}" for src_url, trg_url in zip(src_urls, src_urls_lang)]

    urls_generator = utils.tokenize_batch_from_iterator(str_urls, tokenizer, batch_size,
                            f=lambda u: preprocess.preprocess_url(u, remove_protocol_and_authority=remove_authority,
                                                                  remove_positional_data=remove_positional_data_from_resource,
                                                                  separator=url_separator, lower=lower),
                            auxiliary_tasks=auxiliary_tasks, inference=True)

    for target_urls in urls_generator:
        target_urls = target_urls["urls"]

        # Tokens
        tokens = utils.encode(tokenizer, target_urls, max_length=max_length_tokens, padding="longest", return_attention_mask=True)
        urls = tokens["input_ids"].to(device)
        attention_mask = tokens["attention_mask"].to(device)

        # Inference
        results = inference_with_heads(model, all_tasks, tokenizer, {"urls": urls, "attention_mask": attention_mask},
                                       amp_context_manager)

        # Get results
        for task in all_tasks:
            outputs = torch.sigmoid(results[task]["outputs"]).cpu()
            outputs_argmax = results[task]["outputs_classification"]
            regression = results[task]["regression"]

            if outputs.numpy().shape[0] != len(src_urls):
                raise Exception("Output samples does not match with the length of src URLs: "
                                f"{outputs.numpy().shape[0]} vs {len(src_urls)}")

            if parallel_likelihood:
                _results = [data if regression else data[1] for data in outputs.numpy()]
                _results = [likelihood for likelihood in _results if likelihood >= threshold]
            else:
                _results = ['positive' if argmax == 1 else 'negative' for argmax in outputs_argmax]

            all_results[task].extend(_results)

    return all_results


import logging
import argparse
import json
import base64

import url2lang.utils.utils as utils
import url2lang.url2lang as u2l
import url2lang.inference as u2l_inference

import torch
import numpy as np
from flask import (
    Flask,
    request,
    jsonify,
)
from service_streamer import ThreadedStreamer

app = Flask("url2lang-flask-server")

global_conf = {} # Empty since it will be filled once main is run
logger = logging.getLogger("url2lang")

# Disable (less verbose) 3rd party logging
logging.getLogger("werkzeug").setLevel(logging.WARNING)

@app.route('/', methods=['GET'])
def info():
    available_routes = json.dumps(
        {
            "/hello-world": ["GET"],
            "/inference": ["GET", "POST"],
        },
        indent=4).replace('\n', '<br/>').replace(' ', '&nbsp;')

    return f"Available routes:<br/>{available_routes}"

@app.route('/hello-world', methods=["GET"])
def hello_world():
    return jsonify({"ok": "hello world! server is working!", "err": "null"})

@app.route('/inference', methods=["GET", "POST"])
def inference():
    if request.method not in ("GET", "POST"):
        return jsonify({"ok": "null", "err": "method is not: GET, POST"})

    # Get parameters
    try:
        if request.method == "GET":
            # GET method should be used only for testing purposes since HTML encoding is not being handled
            urls = request.args.getlist("urls")
        elif request.method == "POST":
            urls = request.form.getlist("urls")
        else:
            logger.warning("Unknown method: %s", request.method)

            return jsonify({"ok": "null", "err": f"unknown method: {request.method}"})
    except KeyError as e:
        logger.warning("KeyError: %s", e)

        return jsonify({"ok": "null", "err": f"could not get some mandatory field: 'urls' are mandatory"})

    if not urls:
        logger.warning("Empty urls: %s", urls)

        return jsonify({"ok": "null", "err": "'urls' are mandatory fields and can't be empty"})

    if not isinstance(urls, list):
        logger.warning("Single URL was provided instead of a batch: this will slow the inference")

        if not isinstance(urls, list):
            urls = [urls]

    logger.debug("Got %d URLs", len(urls))

    base64_encoded = global_conf["expect_urls_base64"]

    if base64_encoded:
        try:
            urls = [base64.b64decode(f"{u.replace('_', '+')}==").decode("utf-8", errors="backslashreplace").replace('\n', ' ') for u in urls]
        except Exception as e:
            logger.error("Exception when decoding BASE64: %s", e)

            return jsonify({"ok": "null", "err": "error decoding BASE64 URLs"})

    for idx, url in enumerate(urls, 1):
        logger.debug("'URL #%d: %s", idx, url)

    # Inference

    disable_streamer = global_conf["disable_streamer"]
    get_results = global_conf["streamer"].predict if not disable_streamer else batch_prediction
    results = get_results(urls)

    # Return results
    if len(results) != len(urls):
        logger.warning("Results length mismatch with the provided URLs (task '%s'): %d vs %d: %s vs %s",
                        task, len(results), len(urls), results, urls)

        return jsonify({
            "ok": "null",
            "err": f"results length mismatch with the provided URLs (task '{task}'): {len(results)} vs {len(urls)}",
        })

    results = [str(r) for r in results]

    logger.debug("Results: %s", results)

    return jsonify({
        "ok": results,
        "err": "null",
    })

def batch_prediction(urls):
    logger.debug("URLs batch size: %d", len(urls))

    model = global_conf["model"]
    tokenizer = global_conf["tokenizer"]
    device = global_conf["device"]
    batch_size = global_conf["batch_size"]
    max_length_tokens = global_conf["max_length_tokens"]
    amp_context_manager = global_conf["amp_context_manager"]
    remove_authority = global_conf["remove_authority"]
    remove_positional_data_from_resource = global_conf["remove_positional_data_from_resource"]
    parallel_likelihood = global_conf["parallel_likelihood"]
    url_separator = global_conf["url_separator"]
    lower = global_conf["lower"]
    auxiliary_tasks = global_conf["auxiliary_tasks"]
    target_task = global_conf["target_task"]

    # Inference
    results = u2l_inference.non_interactive_inference(
        model, tokenizer, batch_size, max_length_tokens, device, amp_context_manager, urls,
        remove_authority=remove_authority, remove_positional_data_from_resource=remove_positional_data_from_resource,
        parallel_likelihood=parallel_likelihood, url_separator=url_separator, lower=lower,
        auxiliary_tasks=auxiliary_tasks,
    )

    return results[target_task] # TODO do we need a list if the streamer is used (it seems so)?
                                # https://github.com/ShannonAI/service-streamer/issues/97

def main(args):
    model_input = args.model_input
    force_cpu = args.force_cpu
    use_cuda = utils.use_cuda(force_cpu=force_cpu)
    device = torch.device("cuda:0" if use_cuda else "cpu")
    pretrained_model = args.pretrained_model
    flask_port = args.flask_port
    lower = args.lowercase
    auxiliary_tasks = args.auxiliary_tasks
    target_task = args.target_task
    regression = args.regression
    streamer_max_latency = args.streamer_max_latency
    run_flask_server = not args.do_not_run_flask_server
    disable_streamer = args.disable_streamer

    if not disable_streamer:
        logger.warning("Since streamer is enabled, you might get slightly different results: not recommended for production")
        # Related to https://discuss.pytorch.org/t/slightly-different-results-in-same-machine-and-gpu-but-different-order/173581

    if auxiliary_tasks is None:
        auxiliary_tasks = []

    logger.debug("Device: %s", device)

    if "model" not in global_conf:
        # model = load_model(all_tasks, all_tasks_kwargs, model_input=model_input, pretrained_model=pretrained_model, device=device)
        all_tasks = ["language-identification"] + auxiliary_tasks
        all_tasks_kwargs = u2l.load_tasks_kwargs(all_tasks, auxiliary_tasks, regression)
        global_conf["model"] = u2l.load_model(all_tasks, all_tasks_kwargs, model_input=model_input,
                                              pretrained_model=pretrained_model, device=device)
    else:
        # We apply this step in order to avoid loading the model multiple times due to flask debug mode
        pass

    global_conf["tokenizer"] = u2l.load_tokenizer(pretrained_model)
    global_conf["device"] = device
    global_conf["batch_size"] = args.batch_size
    global_conf["max_length_tokens"] = args.max_length_tokens
    global_conf["amp_context_manager"], _, _ = u2l.get_amp_context_manager(args.cuda_amp, use_cuda)
    global_conf["remove_authority"] = args.remove_authority
    global_conf["remove_positional_data_from_resource"] = args.remove_positional_data_from_resource
    global_conf["parallel_likelihood"] = args.parallel_likelihood
    global_conf["url_separator"] = args.url_separator
    global_conf["streamer"] = ThreadedStreamer(batch_prediction, batch_size=args.batch_size, max_latency=streamer_max_latency)
    global_conf["disable_streamer"] = disable_streamer
    global_conf["expect_urls_base64"] = args.expect_urls_base64
    global_conf["lower"] = lower
    global_conf["auxiliary_tasks"] = auxiliary_tasks
    global_conf["target_task"] = target_task

    # Some guidance
    logger.info("Example: curl http://127.0.0.1:%d/hello-world", flask_port)
    logger.debug("Example: curl http://127.0.0.1:%d/inference -X POST -d \"urls=https://domain/resource1\"", flask_port)

    if run_flask_server:
        # Run flask server
        app.run(debug=args.flask_debug, port=flask_port)

def initialization():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="url2lang: flask server")

    parser.add_argument('model_input', help="Model input path which will be loaded")

    parser.add_argument('--batch-size', type=int, default=16, help="Batch size")
    parser.add_argument('--pretrained-model', default="xlm-roberta-base", help="Pretrained model")
    parser.add_argument('--max-length-tokens', type=int, default=256, help="Max. length for the generated tokens")
    parser.add_argument('--parallel-likelihood', action="store_true", help="Print parallel likelihood instead of classification string (inference)")
    parser.add_argument('--threshold', type=float, default=-np.inf, help="Only print URLs which have a parallel likelihood greater than the provided threshold (inference)")
    parser.add_argument('--remove-authority', action="store_true", help="Remove protocol and authority from provided URLs")
    parser.add_argument('--remove-positional-data-from-resource', action="store_true", help="Remove content after '#' in the resorce (e.g. https://www.example.com/resource#position -> https://www.example.com/resource)")
    parser.add_argument('--force-cpu', action="store_true", help="Run on CPU (i.e. do not check if GPU is possible)")
    parser.add_argument('--url-separator', default='/', help="Separator to use when URLs are stringified")
    parser.add_argument('--cuda-amp', action="store_true", help="Use CUDA AMP (Automatic Mixed Precision)")
    parser.add_argument('--disable-streamer', action="store_true", help="Do not use streamer (it might lead to slower inference and OOM errors)")
    parser.add_argument('--expect-urls-base64', action="store_true", help="Decode BASE64 URLs")
    parser.add_argument('--flask-port', type=int, default=5000, help="Flask port")
    parser.add_argument('--lowercase', action="store_true", help="Lowercase URLs while preprocessing")
    parser.add_argument('--auxiliary-tasks', type=str, nargs='*', choices=["mlm"],
                        help="Tasks which will try to help to the main task (multitasking)")
    parser.add_argument('--target-task', type=str, default="language-identification",
                        help="Task which will be used as primary task and whose results will be used")
    parser.add_argument('--regression', action="store_true", help="Apply regression instead of binary classification")
    parser.add_argument('--streamer-max-latency', type=float, default=0.1,
                        help="Streamer max latency. You will need to modify this parameter if you want to increase the GPU usage")
    parser.add_argument('--do-not-run-flask-server', action="store_true", help="Do not run app.run")

    parser.add_argument('-v', '--verbose', action="store_true", help="Verbose logging mode")
    parser.add_argument('--flask-debug', action="store_true", help="Flask debug mode. Warning: this option might load the model multiple times")

    args = parser.parse_args()

    return args

def cli():
    global logger

    args = initialization()
    logger = utils.set_up_logging_logger(logging.getLogger("url2lang.flask_server"), level=logging.DEBUG if args.verbose else logging.INFO)

    logger.debug("Arguments processed: {}".format(str(args)))

    main(args)

    if not args.do_not_run_flask_server:
        logger.info("Bye!")
    else:
        logger.info("Execution has finished")

if __name__ == "__main__":
    cli()

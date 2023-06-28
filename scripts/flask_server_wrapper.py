
def init(model_input, batch_size=16, streamer_max_latency=0.1, target_task="language-identification"):
    import os
    import sys
    import url2lang.flask_server as flask_server

    if "CUDA_VISIBLE_DEVICES" in os.environ:
        devices = os.environ["CUDA_VISIBLE_DEVICES"].split(',')

        if len(devices) > 1:
            import logging

            cuda_device = devices[os.getpid() % len(devices)]
            os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)

            logging.warning("NOT a perfect approach for assigning GPUs: be aware that you might need to reset if GPUs are not allocated properly")
            logging.info("CUDA device (PID: %d): %d (available: %d)", os.getpid(), cuda_device, len(devices))

    sys.argv = [sys.argv[0]] # Remove all provided args

    # Inject args that will be used by the Flask server
    sys.argv.extend([
        "--batch-size", str(batch_size),
        #"--parallel-likelihood",
        "--target-task", target_task,
        #"--regression",
        "--streamer-max-latency", str(streamer_max_latency),
        "--do-not-run-flask-server", # Necessary for gunicorn in order to work properly
        "--expect-urls-base64",
        #"--verbose",
        #"--disable-streamer", # It should be enabled for crawls of multiple websites, but disabled for a few websites
        model_input
    ])

    flask_server.cli()

    return flask_server.app

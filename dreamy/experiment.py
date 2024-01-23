"""
Tools for running dreaming experiments on Modal.
"""
import os

import dataclasses
import pickle
from typing import Callable

import torch
import transformers
import modal
from dreamy.epo import epo


def download(param_str="12b"):
    model_name = f"EleutherAI/pythia-{param_str}-deduped"

    import transformers

    transformers.AutoTokenizer.from_pretrained(
        model_name,
        token=os.environ.get("HUGGINGFACE_TOKEN", None),
    )

    transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        token=os.environ["HUGGINGFACE_TOKEN"],
    )


image = (
    modal.Image.from_registry("pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel")
    .pip_install(
        "numpy",
        "transformers>=4.34.0",
        "typer>=0.9",
        "accelerate",
        "tqdm",
        "boto3",
        "ninja",
        "packaging",
        "pandas",
        "s3fs",
        "mosaicml-streaming",
        "datasets",
    )
    .apt_install("git", "wget")
    .run_commands("pip3 install flash-attn --no-build-isolation")
    .run_function(download, secrets=[modal.Secret.from_name("huggingface")])
)

params = dict(
    retries=1,
    timeout=60 * 60 * 24,
    cpu=8,
    memory=64 * 1024,
    # 20, 40, 80 are options
    gpu=modal.gpu.A100(memory=int(os.environ.get("MODAL_A100_MEMORY", 40))),
    secrets=[
        modal.Secret.from_name("s3-access"),
        modal.Secret.from_name("huggingface"),
    ],
    mounts=[
        modal.Mount.from_local_dir(
            os.path.dirname(os.path.realpath(__file__)),
            remote_path="/root",
            condition=lambda fn: fn.endswith(".py"),
            recursive=True,
        )
    ],
    concurrency_limit=10,
)

stub = modal.Stub("dreamy", image=image)
stub_function = stub.function(**params)
stub_cls = stub.cls(**params)


def chunk_list(lst, N):
    avg = len(lst) // N
    rem = len(lst) % N
    start = 0

    for i in range(N):
        end = start + avg + (i < rem)
        yield lst[start:end]
        start = end


@stub_cls
class RemoteDream:
    def __init__(self, model_size):
        self.model_size = model_size

    def __enter__(self):
        self.model, self.tokenizer = load_model(model_size=self.model_size)

    @modal.method()
    def _dream(self, cfgs):
        output = []
        for c in cfgs:
            assert c.model_size == self.model_size
            output.append(dream(c, model=self.model, tokenizer=self.tokenizer))
        return output

    def dream(self, cfgs, n_workers=None, local=False):
        if n_workers is None:
            n_workers = min(10, len(cfgs))
        chunks = list(chunk_list(cfgs, n_workers))
        if local:
            return list(map(self._dream.local, chunks))
        else:
            return list(self._dream.map(chunks, return_exceptions=True))


def retrieve_files(cfgs):
    for c in cfgs:
        import s3fs

        fs = s3fs.S3FileSystem()
        s3_full_path = os.path.join(c.s3_bucket, c.s3_path)
        print("downloading", s3_full_path, "to", c.output_path)
        try:
            fs.download(s3_full_path, c.output_path)
        except FileNotFoundError:
            print("file not found", s3_full_path)


def check_file_exists(s3, bucket, key):
    from botocore.exceptions import ClientError

    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        # If a client error is thrown, check if it was a 404 error (file not found)
        if e.response["Error"]["Code"] == "404":
            return False
        else:
            # Re-raise the exception if it was a different kind of client error.
            raise


@dataclasses.dataclass
class DreamConfig:
    """ """

    runner_builder: Callable
    model_size: str = "12b"
    x_penalty_min: float = 1.0 / 10.0
    x_penalty_max: float = 10.0
    iters: int = 300
    seq_len: int = 12
    batch_size: int = 256
    population_size: int = 8
    explore_per_pop: int = 32
    restart_frequency: int = 30
    restart_xentropy: float = 2.0
    restart_xentropy_max_mult: float = 3.0
    initial_ids: torch.Tensor = None
    initial_str: str = ""  # overrides initial_ids
    topk: int = 512
    attribution_frequency: int = None
    output_path: str = ""
    s3_bucket: str = "caiplay"
    s3_path: str = ""
    seed: int = 0
    payload: dict = None
    gcg: float = (
        None  # sets x_penalty and overrides population_size and explore_per_pop
    )


def dream(c: DreamConfig, model=None, tokenizer=None):
    if len(c.s3_path) > 0:
        import boto3

        s3 = boto3.client("s3")
        if check_file_exists(s3, c.s3_bucket, c.s3_path):
            print("run already done, skipping", c.s3_path)
            return False

    if model is None:
        model, tokenizer = load_model(model_size=c.model_size)

    if c.gcg is not None:
        c.population_size = 1
        c.explore_per_pop = c.batch_size
        c.x_penalty_min = c.gcg
        c.x_penalty_max = c.gcg

    if len(c.initial_str) > 0:
        c.initial_ids = tokenizer.encode(c.initial_str, return_tensors="pt").to(
            model.device
        )

    history = epo(
        c.runner_builder(model, tokenizer),
        model,
        tokenizer,
        initial_ids=c.initial_ids,
        seed=c.seed,
        x_penalty_min=c.x_penalty_min,
        x_penalty_max=c.x_penalty_max,
        iters=c.iters,
        seq_len=c.seq_len,
        batch_size=c.batch_size,
        population_size=c.population_size,
        explore_per_pop=c.explore_per_pop,
        restart_frequency=c.restart_frequency,
        restart_xentropy=c.restart_xentropy,
        restart_xentropy_max_mult=c.restart_xentropy_max_mult,
        topk=c.topk,
        attribution_frequency=c.attribution_frequency,
    )

    # can't pickle the runner_builder (could use cloudpickle)
    c.runner_builder = None
    output = (c, history)
    if len(c.output_path) > 0:
        folder_path = os.path.dirname(c.output_path)
        os.makedirs(folder_path, exist_ok=True)
        with open(c.output_path, "wb") as f:
            pickle.dump(output, f)

    if len(c.s3_path) > 0:
        import boto3

        s3 = boto3.resource("s3")
        s3.Bucket(c.s3_bucket).upload_file(c.output_path, c.s3_path)
    return output

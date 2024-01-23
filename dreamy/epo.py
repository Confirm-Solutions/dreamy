"""
This file implements the EPO algorithm. See the `epo` function for the main entrypoint.
"""
import dataclasses
import time
import contextlib
from typing import Callable, Dict, List, Union, Tuple

import numpy as np
import torch
import torch.distributions
import torch.nn.functional as F
import transformers


@contextlib.contextmanager
def add_fwd_hooks(module_hooks: List[Tuple[torch.nn.Module, Callable]]):
    """
    Context manager for temporarily adding forward hooks to a model.

    Parameters
    ----------
    module_hooks
        A list of pairs: (module, fnc) The function will be registered as a
            forward hook on the module
    """
    try:
        handles = []
        for mod, hk in module_hooks:
            handles.append(mod.register_forward_hook(hk))
        yield
    finally:
        for h in handles:
            h.remove()


@dataclasses.dataclass
class History:
    """
    The `epo` function returns a History object that contains the full history
    of the population members at each iteration.
    """

    # The token ids for each population member at each iteration.
    ids: List = dataclasses.field(default_factory=lambda: [])
    # The cross-entropy loss for each population member at each iteration.
    xentropy: List = dataclasses.field(default_factory=lambda: [])
    # The target objective for each population member at each iteration.
    target: List = dataclasses.field(default_factory=lambda: [])
    # The indices of the population members that were retained at each iteration.
    keep: List = dataclasses.field(default_factory=lambda: [])
    # The runtime for each iteration.
    runtime: List = dataclasses.field(default_factory=lambda: [])

    def subset(self, slc):
        """
        Return a History object sliced along the iterations dimension.
        """
        return History(
            self.ids[slc],
            self.xentropy[slc],
            self.target[slc],
            self.keep[slc],
            self.runtime[slc],
        )

    def _insert(self, new_ids, target, xentropy, keep, runtime):
        self.ids.append(new_ids.cpu().numpy())
        self.target.append(target.cpu().numpy())
        self.xentropy.append(xentropy.cpu().numpy())
        self.keep.append(keep.cpu().numpy())
        self.runtime.append(runtime)

    def _finalize(self):
        self.ids = np.stack(self.ids, axis=0)
        self.target = np.stack(self.target, axis=0)
        self.xentropy = np.stack(self.xentropy, axis=0)
        self.keep = np.stack(self.keep, axis=0)
        self.runtime = np.array(self.runtime)


@torch.no_grad()
def epo(
    cache_run: Callable,
    model: torch.nn.Module,
    tokenizer: transformers.PreTrainedTokenizer,
    seq_len: int = 12,
    population_size: int = 8,
    iters: int = 300,
    explore_per_pop: int = 32,
    batch_size: int = 256,
    topk: int = 512,
    mutation_method: str = "gradient",
    x_penalty_min: float = 1.0 / 10.0,
    x_penalty_max: float = 10.0,
    restart_frequency: int = 50,
    restart_xentropy: float = 2.0,
    restart_xentropy_max_mult: float = 3.0,
    seed: int = 0,
    initial_ids: torch.Tensor = None,
    history: History = None,
    catch_keyboard_interrupt: bool = False,
    callback: Union[Callable, bool] = None,
    always_recompute_gradients: bool = False,
) -> History:
    """
    Run the EPO algorithm. See the paper for details.

    Parameters
    ----------
    cache_run
        A callable that accepts either input_ids or inputs_embeds and returns a
        dictionary containing the `target` and the logits for each token
        position.
    model
    tokenizer
    seq_len, optional
        The number of tokens in the optimized prompt, by default 16
    population_size, optional
        The population to keep at each iteration, by default 32
    iters, optional
        Number of iterations to run EPO, by default 1000
    explore_per_pop, optional
        Number of children per population member per iteration, by default 4
    batch_size, optional
        GPU batch size, by default 8
    topk, optional
        When selecting token replacements, we select the `topk` tokens by
        gradient magnitude and choose uniformly at random between those, by
        default 32.
    mutation_method, optional
        research, ignore, by default "gradient"
    x_penalty_min, optional
        The minimum cross-entropy penalty, by default 1.0/16.0
    x_penalty_max, optional
        The maximum cross-entropy penalty, by default 16.0
    restart_frequency, optional
        How often do we reset the Pareto frontier, by default 50
    restart_xentropy, optional
        When we reset the Pareto frontier, we select a population member that
        is optimal according to a cross-entropy penalty that is selected
        uniformly at random in the domain
        [restart_xentropy / restart_xentropy_max_mult,
         restart_xentropy * restart_xentropy_max_mult],
        restart_xentropy is by default 2.0
    restart_xentropy_max_mult, optional
        See the explanation for restart_xentropy, by default 3.0
    seed, optional
        Random seed used for initialization, by default 0
    initial_ids, optional
        The initial token ids to begin optimizing from. If None, the initial
        token ids will be selected randomly, by default None
    history, optional
        The history of an EPO run that we want to continue, by default None
    catch_keyboard_interrupt, optional
        Should we catch keyboard interrupts and end the EPO loop?, by default False
    callback, optional
        A function called at the beginning of each iteration, by default None
    always_recompute_gradients, optional
        If a population member is retained across an iteration, we default to
        not recomputing that population member's token gradients. If your
        cache_run stores internal state that changes, you may want to override
        this behavior and recompute gradients every iteration.

    Returns
    -------
        A History object containing the full history of the

    Raises
    ------
    ValueError
        _description_
    ValueError
        _description_
    ValueError
        _description_
    """
    start = time.time()
    explore_size = population_size * explore_per_pop
    device = model.device

    if seed is not None:
        torch.manual_seed(seed)

    if x_penalty_min is None or x_penalty_max is None:
        X = torch.zeros(population_size, device=model.device)
    else:
        X = torch.exp(
            torch.linspace(
                np.log(x_penalty_min), np.log(x_penalty_max), population_size
            )
        ).to(model.device)

    if callback is None:
        callback = pareto_callback(
            cache_run,
            model,
            tokenizer,
            X.min().item(),
            X.max().item(),
        )
    elif callback is False:
        callback = lambda *x: True

    #### history and initial_ids ####
    if history is not None:
        if initial_ids is not None:
            raise ValueError("Cannot specify both history and initial_ids.")
        input_ids = history.ids[-1, history.keep[-1]]
    elif initial_ids is not None:
        history = History()
        input_ids = initial_ids.to(model.device)
        if initial_ids.shape[1] != seq_len:
            raise ValueError(f"initial_ids must have shape (*, {seq_len})")
    else:
        history = History()
        input_ids = torch.randint(
            0, tokenizer.vocab_size, (population_size, seq_len)
        ).to(model.device)

    #### choose a update selection method ####
    if mutation_method == "gradient":
        selector_type = GradientSelector
    else:
        raise ValueError(f"Unknown selection method: {mutation_method}")
    selector = selector_type(model, cache_run, X, batch_size)

    #### Run the EPO loop: ####
    if hasattr(cache_run, "setup"):
        cache_run.setup(input_ids)
    state = selector.setup(input_ids)

    # We use a try/except block so that we can catch keyboard interrupts and
    # still return results. This is useful for interactive use when it's nice
    # to launch with a large `iters` parameter and then just stop the run when
    # the results look good enough.
    try:
        for i in range(iters):
            ########################################
            # 1) Report!
            ########################################
            terminate_flag = callback(i, state, time.time() - start, history)
            if (
                (isinstance(terminate_flag, str) and terminate_flag == "terminate")
                or (isinstance(terminate_flag, torch.Tensor) and terminate_flag.item())
                or (isinstance(terminate_flag, bool) and terminate_flag)
            ):
                if i == 0:
                    history._insert(
                        state.ids,
                        state.target,
                        state.xentropy,
                        torch.arange(state.ids.shape[0]),
                        time.time() - start,
                    )
                break
            else:
                start = time.time()
            recompute_gradients = always_recompute_gradients or (
                terminate_flag == "recompute_gradients"
            )

            ########################################
            # 2) Birth children from parents
            # copy inputs to expand out to explore_size new candidates.
            ########################################
            source_idx = torch.cat(
                (
                    torch.arange(state.ids.shape[0], device=device).repeat(
                        explore_size // state.ids.shape[0]
                    ),
                    torch.arange(explore_size % state.ids.shape[0], device=device),
                )
            )
            assert source_idx.shape[0] == explore_size
            assert (source_idx < state.ids.shape[0]).all()

            new_ids = state.ids[source_idx, :].clone()

            ########################################
            # 3) Run the selector. This might be:
            #    - random
            #    - gradient-guided
            #    - cosine-similarity-guided
            ########################################
            selector.mutate(state, source_idx, new_ids, topk)

            ########################################
            # 5) Evaluate fitness
            ########################################
            new_state = evaluate_fitness(
                model, cache_run, new_ids, batch_size=batch_size
            )
            all_state = state.cat(new_state)

            # note that all_loss is a matrix with a row for each population
            # member because each population member slot uses a different
            # xentropy penalty.
            all_loss = (
                -all_state.target[None, :] + X[:, None] * all_state.xentropy[None, :]
            )
            keep = (-all_loss).argmax(dim=1).to(torch.int)

            if i % restart_frequency == 0:
                min_mult = 1.0 / restart_xentropy_max_mult
                max_mult = restart_xentropy_max_mult
                mult = min_mult + (max_mult - min_mult) * torch.rand(1).item()
                restart_X = restart_xentropy * mult
                restart_loss = -all_state.target + restart_xentropy * all_state.xentropy
                print(f"restarting with xentropy penalty of {restart_X:.2f}")
                keep[:] = restart_loss.argmin()

            history._insert(
                all_state.ids,
                all_state.target,
                all_state.xentropy,
                keep,
                time.time() - start,
            )

            ########################################
            # 6) Calculate gradients for the next iteration.
            ########################################
            if i != iters - 1:
                if selector.uses_gradient:
                    if recompute_gradients:
                        survived = torch.tensor([])
                        new = keep
                    else:
                        survived = keep[keep < state.ids.shape[0]]
                        new = keep[keep >= state.ids.shape[0]]
                    if new.shape[0] > 0:
                        state_new = selector.setup(all_state.ids[new])
                    if survived.shape[0] > 0:
                        state_survived = state.subset(survived)
                        if new.shape[0] > 0:
                            state = state_survived.cat(state_new)
                        else:
                            state = state_survived
                    else:
                        state = state_new
                else:
                    state = all_state.subset(keep)

    # it's handy to sometimes be able to interrupt the loop and still get
    # results!
    except KeyboardInterrupt:
        if catch_keyboard_interrupt:
            pass
        else:
            raise

    terminate_flag = callback(i, state, time.time() - start, history, final=True)

    history._finalize()

    return history


@dataclasses.dataclass
class ParetoFrontier:
    # the range of cross-entropy penalties used
    Xvs: np.ndarray
    # the target and xentropy values for each penalty level
    full_target: np.ndarray
    full_xentropy: np.ndarray
    # the unique indices in full_target/full_xentropy that make up the pareto frontier.
    unique: np.ndarray
    # the target and xentropy values for the unique entries
    target: np.ndarray
    xentropy: np.ndarray
    # the token ids for each unique point on the frontier.
    ids: np.ndarray
    # the detokenized text for each unique point on the frontier.
    text: List[str]


def build_pareto_frontier(tokenizer, histories, Xvs=None):
    """
    Construct a pareto frontier from the history of several EPO runs. We allow
    multiple histories to be passed so that we can construct the Pareto
    frontier across several different runs of EPO with different random
    initializations.

    Parameters
    ----------
    tokenizer
    histories
        A list of History objects returned by the EPO algorithm. We allow
        multiple independent histories to be combined
    Xvs, optional
        The range of cross-entropy penalties to use.
        By default Xvs = 1.0 / np.linspace(0, 50, 1000)[1:]

    Returns
    -------
        A ParetoFrontier object.
    """

    if Xvs is None:
        Xvs = 1.0 / np.linspace(0, 50, 1000)[1:]

    if not isinstance(histories, list):
        histories = [histories]
    x = []
    t = []
    ids = []
    for h in histories:
        x.append(h.xentropy.flatten())
        t.append(h.target.flatten())
        ids.append(h.ids.reshape((-1, h.ids.shape[-1])))

    history_x = np.concatenate(x)
    history_t = np.concatenate(t)
    history_ids = np.concatenate(ids, axis=0)
    pareto_t = np.empty(Xvs.shape[0])
    pareto_x = np.empty(Xvs.shape[0])
    pareto_idxs = []
    for i, Xv in enumerate(Xvs):
        loss = -history_t + Xv * history_x
        idx = loss.argmin()
        pareto_idxs.append(idx)
        pareto_t[i] = history_t[idx]
        pareto_x[i] = history_x[idx]
    pareto_unique = np.unique(pareto_idxs, return_index=True)[1]
    pareto_ids = [history_ids[pareto_idxs[i]] for i in pareto_unique]
    pareto_text = [tokenizer.decode(ids) for ids in pareto_ids]
    return ParetoFrontier(
        np.array(Xvs),
        pareto_t[pareto_unique],
        pareto_x[pareto_unique],
        pareto_ids,
        pareto_text,
        pareto_unique,
        pareto_t,
        pareto_x,
    )


def gcg(
    cache_run: Callable,
    model: torch.nn.Module,
    tokenizer: transformers.PreTrainedTokenizer,
    seq_len: int = 16,
    iters: int = 1000,
    batch_size: int = 8,
    topk: int = 32,
    x_penalty_min: float = 1.0 / 16.0,
    x_penalty_max: float = 16.0,
    seed: int = 0,
    initial_ids: torch.Tensor = None,
    history: History = None,
    catch_keyboard_interrupt: bool = False,
    callback: Union[Callable, bool] = None,
    always_recompute_gradients: bool = False,
):
    """GCG is a special case of EPO where the population size is 1."""
    epo(
        cache_run,
        model,
        tokenizer,
        seq_len=seq_len,
        population_size=1,
        iters=iters,
        explore_per_pop=batch_size,
        batch_size=batch_size,
        topk=topk,
        mutation_method="gradient",
        x_penalty_min=x_penalty_min,
        x_penalty_max=x_penalty_max,
        seed=seed,
        initial_ids=initial_ids,
        history=history,
        catch_keyboard_interrupt=catch_keyboard_interrupt,
        callback=callback,
        always_recompute_gradients=always_recompute_gradients,
    )


def cat_if_not_none(a, b):
    if a is None or b is None:
        return None
    else:
        return torch.cat((a, b), dim=0)


@dataclasses.dataclass
class State:
    ids: torch.Tensor
    target: torch.Tensor
    xentropy: torch.Tensor
    final_token: torch.Tensor
    token_grads: torch.Tensor
    extra: Dict[str, torch.Tensor]

    def cat(self, state2):
        return State(
            ids=torch.cat((self.ids, state2.ids), dim=0),
            target=torch.cat((self.target, state2.target), dim=0),
            xentropy=torch.cat((self.xentropy, state2.xentropy), dim=0),
            final_token=torch.cat((self.final_token, state2.final_token), dim=0),
            token_grads=cat_if_not_none(self.token_grads, state2.token_grads),
            extra={
                k: cat_if_not_none(self.extra[k], state2.extra[k]) for k in self.extra
            },
        )

    def subset(self, keep):
        return State(
            ids=self.ids[keep],
            target=self.target[keep],
            xentropy=self.xentropy[keep],
            final_token=self.final_token[keep],
            token_grads=self.token_grads[keep.to("cpu")]
            if self.token_grads is not None
            else None,
            extra={k: self.extra[k][keep] for k in self.extra},
        )


# based on https://github.com/llm-attacks/llm-attacks/blob/main/llm_attacks/gcg/gcg_attack.py
def token_grads(
    model: torch.nn.Module,
    cache_run: Callable,
    input_ids: torch.Tensor,
    x_penalty: torch.Tensor,
    batch_size: int,
):
    """
    Compute gradients with respect to one-hot encoded input tokens. This is a
    infinitesimal approximation to the token influence on the loss so it's a
    very noisy indicator of which tokens might reduce loss.
    """
    embed = model.get_input_embeddings()

    token_grads = torch.empty(
        (input_ids.shape[0], input_ids.shape[1], embed.num_embeddings),
        dtype=torch.float,
    )
    loss = torch.empty(input_ids.shape[0], device=model.device)
    xentropy = torch.empty(input_ids.shape[0], device=model.device)
    target = torch.empty(input_ids.shape[0], device=model.device)
    final_token = torch.empty(input_ids.shape[0], device=model.device, dtype=torch.long)
    extra = dict()

    with torch.enable_grad():
        model.zero_grad()

        for i in range(0, input_ids.shape[0], batch_size):
            imax = min(i + batch_size, input_ids.shape[0])

            # using a one hot matrix as input to the model gives us gradients with
            # respect to potential input tokens.
            one_hot = F.one_hot(
                input_ids[i:imax].clone(), num_classes=embed.num_embeddings
            ).to(embed.weight.dtype)
            one_hot.requires_grad = True

            cache = cache_run(inputs_embeds=torch.matmul(one_hot, embed.weight))

            logits_offset = cache["logits"][:, :-1]
            this_xentropy = (
                -(torch.log_softmax(logits_offset, dim=-1) * one_hot[:, 1:])
                .sum(dim=-1)
                .mean(dim=-1)
            )

            this_loss = -cache["target"] + this_xentropy * x_penalty[i:imax]
            this_loss.sum().backward()

            loss[i:imax] = this_loss
            target[i:imax] = cache["target"]
            xentropy[i:imax] = this_xentropy
            final_token[i:imax] = cache["logits"][:, -1, :].argmax(dim=-1)
            token_grads[i:imax] = one_hot.grad

            for k in cache:
                if k not in ["target", "logits"]:
                    e = cache[k]
                    if k not in extra:
                        extra[k] = torch.empty(
                            (input_ids.shape[0], *e.shape[1:]),
                            dtype=e.dtype,
                            device=e.device,
                        )
                    extra[k][i:imax] = e

            # important to zero out gradients here to release memory
            model.zero_grad()

    return State(input_ids, target, xentropy, final_token, token_grads, extra)


def calc_xentropy(logits, input_ids):
    logits_offset = logits[:, :-1]
    return (
        torch.nn.CrossEntropyLoss(reduction="none")(
            logits_offset.reshape(-1, logits_offset.shape[-1]),
            input_ids[:, 1:].reshape(-1),
        )
        .view(*logits_offset.shape[:2])
        .mean(dim=-1)
    )


def evaluate_fitness(
    model: torch.nn.Module,
    cache_run: Callable,
    input_ids: torch.Tensor,
    batch_size: int,
):
    target = torch.empty(input_ids.shape[0], dtype=torch.float, device=input_ids.device)
    xentropy = torch.empty(
        input_ids.shape[0], dtype=torch.float, device=input_ids.device
    )
    final_token = torch.empty(
        input_ids.shape[0], dtype=torch.long, device=input_ids.device
    )
    extra = dict()
    for i in range(0, input_ids.shape[0], batch_size):
        imax = min(i + batch_size, input_ids.shape[0])
        mini_batch = cache_run(input_ids=input_ids[i:imax])
        target[i:imax] = mini_batch["target"]
        xentropy[i:imax] = calc_xentropy(mini_batch["logits"], input_ids[i:imax])
        final_token[i:imax] = mini_batch["logits"][:, -1, :].argmax(dim=-1)

        for k in mini_batch:
            if k not in ["target", "logits"]:
                e = mini_batch[k]
                if k not in extra:
                    extra[k] = torch.empty(
                        (input_ids.shape[0], *e.shape[1:]),
                        dtype=e.dtype,
                        device=e.device,
                    )
                extra[k][i:imax] = e

    return State(input_ids, target, xentropy, final_token, None, extra)


class Selector:
    def __init__(
        self,
        model: torch.nn.Module,
        cache_run: Callable,
        X: torch.Tensor,
        batch_size: int,
    ):
        self.model = model
        self.cache_run = cache_run
        self.X = X
        self.batch_size = batch_size


class GradientSelector(Selector):
    uses_gradient = True

    def setup(self, input_ids: torch.Tensor):
        return token_grads(
            self.model,
            self.cache_run,
            input_ids,
            x_penalty=self.X[: input_ids.shape[0]],
            batch_size=self.batch_size,
        )

    def mutate(self, state, source_idx, input_ids, topk):
        # when just flipping, the current token gradient falls out of the
        # topk operation, so we can just use the negative new token grad
        topk_grad = (-state.token_grads).topk(k=topk, dim=-1)
        pos = torch.randint(
            low=0,
            high=input_ids.shape[1],
            size=(input_ids.shape[0],),
            device=input_ids.device,
        )
        token_idx = torch.randint(
            low=0,
            high=topk,
            size=(input_ids.shape[0],),
            device=input_ids.device,
        )
        input_ids[torch.arange(input_ids.shape[0]), pos] = topk_grad.indices.to(
            input_ids.device
        )[source_idx, pos, token_idx]


def pareto_callback(cache_run, model, tokenizer, x_penalty_min, x_penalty_max):
    def f(i, state, last_runtime, history, final=False):
        if last_runtime is not None:
            print("runtime: {:.2f} seconds".format(last_runtime))
        print(f"\nbeginning step {i}, current pareto frontier inputs:")
        last_idx = None

        Xvs = torch.exp(
            torch.linspace(
                np.log(x_penalty_min / 10.0), np.log(x_penalty_max * 10.0), 200
            )
        ).to(model.device)
        loss = -state.target[None] + Xvs[:, None] * state.xentropy[None]
        idxs = loss.argmin(dim=1)
        for i in range(len(Xvs)):
            idx = idxs[i]
            if idx == last_idx:
                continue
            text = tokenizer.decode(state.ids[idx])
            last_token = tokenizer.decode(state.final_token[idx])
            print(
                f"{Xvs[i]} xentropy={state.xentropy[idx]:.2f} target={state.target[idx]:.2f} {repr(text + '[' + last_token + ']')}"
            )
            last_idx = idx

    return f

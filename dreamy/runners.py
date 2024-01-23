"""
_summary_

Returns
-------
    _description_
"""
from typing import Optional, Tuple

import torch
from transformers.models.gpt_neox.modeling_gpt_neox import apply_rotary_pos_emb

from dreamy.epo import add_fwd_hooks


def does_retokenize(model, tokenizer, input_ids):
    good = torch.empty(input_ids.shape[0], dtype=bool).to(model.device)
    input_strs = tokenizer.batch_decode(input_ids)
    for i, s in enumerate(input_strs):
        retokenized = tokenizer.encode(s, return_tensors="pt").to(model.device)
        if retokenized.shape[1] != input_ids.shape[1]:
            good[i] = False
        else:
            good[i] = (retokenized[0] == input_ids[i]).all()
        if not good[i]:
            print(f"bad input {i}: {s}")
    return good


def logit_diff_runner(
    model, tokenizer, token_id, banned_text, check_retokenization=False
):
    def run(input_ids=None, inputs_embeds=None):
        if input_ids is not None:
            if check_retokenization:
                good = does_retokenize(model, tokenizer, input_ids)
            else:
                good = torch.ones(input_ids.shape[0], dtype=bool).to(model.device)
            input_text = tokenizer.batch_decode(input_ids)
            good &= torch.tensor(
                [banned_text.lower() not in s.lower() for s in input_text], dtype=bool
            ).to(good.device)
        else:
            good = torch.ones(inputs_embeds.shape[0], dtype=bool).to(model.device)

        if input_ids is not None:
            output = model(input_ids)
        else:
            output = model(inputs_embeds=inputs_embeds)

        out = dict()
        out["logits"] = output.logits
        out["target"] = torch.where(
            good,
            output.logits[:, -1, token_id]
            - torch.where(
                output.logits[:, -1].argmax(dim=-1) == token_id,
                output.logits[:, -1].topk(dim=-1, k=2).values[:, 1],
                output.logits[:, -1].max(dim=-1).values,
            ),
            -torch.finfo(output.logits.dtype).max,
        )
        # probs = torch.log_softmax(last_logits, -1, :], dim=-1)
        # out["target"] = torch.where(
        #     good, probs[:, token_id], -torch.finfo(probs.dtype).max
        # )
        return out

    return run


def neuron_runner(model, tokenizer, layer, neuron, check_retokenization=False):
    def run(input_ids=None, inputs_embeds=None):
        if input_ids is not None:
            if check_retokenization:
                good = does_retokenize(model, tokenizer, input_ids)
            else:
                good = torch.ones(input_ids.shape[0], dtype=bool).to(model.device)
        else:
            good = torch.ones(inputs_embeds.shape[0], dtype=bool).to(model.device)

        out = {}

        def get_target(module, input, output):
            out["target"] = input[0][:, -1, neuron]

        hooks = [
            (model.gpt_neox.layers[layer].mlp.dense_4h_to_h, get_target),
        ]

        with add_fwd_hooks(hooks):
            if input_ids is not None:
                output = model(input_ids)
            else:
                output = model(inputs_embeds=inputs_embeds)

        out["logits"] = output.logits
        out["target"][~good] = -torch.finfo(out["target"].dtype).max
        return out

    return run


def residual_runner(model, tokenizer, layer, vector, check_retokenization=False):
    def run(input_ids=None, inputs_embeds=None):
        if input_ids is not None:
            if check_retokenization:
                good = does_retokenize(model, tokenizer, input_ids)
            else:
                good = torch.ones(input_ids.shape[0], dtype=bool).to(model.device)
        else:
            good = torch.ones(inputs_embeds.shape[0], dtype=bool).to(model.device)

        out = {}

        def get_target(module, input, output):
            resid = input[0][:, -1]
            std_resid = (resid - resid.mean(dim=-1, keepdim=True)) / resid.std(
                dim=-1, keepdim=True
            )
            out["target"] = std_resid @ vector

        hooks = [
            (model.gpt_neox.layers[layer], get_target),
        ]

        with add_fwd_hooks(hooks):
            if input_ids is not None:
                output = model(input_ids)
            else:
                output = model(inputs_embeds=inputs_embeds)

        out["logits"] = output.logits
        out["target"][~good] = -torch.finfo(out["target"].dtype).max
        return out

    return run


def attention_forward(
    self,
    hidden_states: torch.FloatTensor,
    attention_mask: torch.FloatTensor = None,
    position_ids: torch.LongTensor = None,
    head_mask: Optional[torch.FloatTensor] = None,
    layer_past: Optional[Tuple[torch.Tensor]] = None,
    use_cache: Optional[bool] = False,
    output_attentions: Optional[bool] = False,
):
    has_layer_past = layer_past is not None

    # Compute QKV
    # Attention heads [batch, seq_len, hidden_size]
    #   --> [batch, seq_len, (np * 3 * head_size)]
    qkv = self.query_key_value(hidden_states)

    # [batch, seq_len, (num_heads * 3 * head_size)]
    #   --> [batch, seq_len, num_heads, 3 * head_size]
    new_qkv_shape = qkv.size()[:-1] + (self.num_attention_heads, 3 * self.head_size)
    qkv = qkv.view(*new_qkv_shape)

    # [batch, seq_len, num_attention_heads, 3 * head_size] --> 3 [batch, num_attention_heads, seq_len, head_size]
    query = qkv[..., : self.head_size].permute(0, 2, 1, 3)
    key = qkv[..., self.head_size : 2 * self.head_size].permute(0, 2, 1, 3)
    value = qkv[..., 2 * self.head_size :].permute(0, 2, 1, 3)

    # Compute rotary embeddings on rotary_ndims
    query_rot = query[..., : self.rotary_ndims]
    query_pass = query[..., self.rotary_ndims :]
    key_rot = key[..., : self.rotary_ndims]
    key_pass = key[..., self.rotary_ndims :]

    # Compute token offset for rotary embeddings (when decoding)
    seq_len = key.shape[-2]
    if has_layer_past:
        seq_len += layer_past[0].shape[-2]
    cos, sin = self.rotary_emb(value, seq_len=seq_len)
    query, key = apply_rotary_pos_emb(query_rot, key_rot, cos, sin, position_ids)
    query = torch.cat((query, query_pass), dim=-1)
    key = torch.cat((key, key_pass), dim=-1)

    # GPT-neo-X casts query and key in fp32 to apply rotary embedding in full precision
    target_dtype = value.dtype
    if query.dtype != target_dtype:
        query = query.to(target_dtype)
    if key.dtype != target_dtype:
        key = key.to(target_dtype)

    batch_size, num_attention_heads, query_length, attn_head_size = query.size()
    key_length = key.size(-2)

    # dynamically increase the causal mask with the key length, if needed.
    if key_length > self.bias.shape[-1]:
        self._init_bias(key_length, device=key.device)
    causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]

    query = query.view(batch_size * num_attention_heads, query_length, attn_head_size)
    key = key.view(batch_size * num_attention_heads, key_length, attn_head_size)
    attn_scores = torch.zeros(
        batch_size * num_attention_heads,
        query_length,
        key_length,
        dtype=query.dtype,
        device=key.device,
    )
    attn_scores = torch.baddbmm(
        attn_scores,
        query,
        key.transpose(1, 2),
        beta=1.0,
        alpha=self.norm_factor,
    )
    attn_scores = attn_scores.view(
        batch_size, num_attention_heads, query_length, key_length
    )

    mask_value = torch.finfo(attn_scores.dtype).min
    # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
    # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
    mask_value = torch.tensor(mask_value, dtype=attn_scores.dtype).to(
        attn_scores.device
    )
    attn_scores = torch.where(causal_mask, attn_scores, mask_value)

    if attention_mask is not None:
        # Apply the attention mask
        attn_scores = attn_scores + attention_mask

    attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)

    # Mask heads if we want to
    if head_mask is not None:
        attn_weights = attn_weights * head_mask

    attn_weights = self.attention_dropout(attn_weights)

    return (
        query.reshape((batch_size, -1, query_length, attn_head_size)),
        key.reshape((batch_size, -1, query_length, attn_head_size)),
        value.reshape((batch_size, -1, query_length, attn_head_size)),
        attn_weights,
    )


def attention_entry_runner(
    model,
    tokenizer,
    layer,
    head_idx,
    token_idx,
    check_retokenization=False,
):
    """
    Returns a model-runner

    Parameters
    ----------
    model
        _description_
    tokenizer
        _description_
    layer
        _description_
    head_idx
        _description_
    token_idx
        _description_
    check_retokenization, optional
        _description_, by default False
    """

    def run(input_ids=None, inputs_embeds=None):
        if input_ids is not None:
            if check_retokenization:
                good = does_retokenize(model, tokenizer, input_ids)
            else:
                good = torch.ones(input_ids.shape[0], dtype=bool).to(model.device)
        else:
            good = torch.ones(inputs_embeds.shape[0], dtype=bool).to(model.device)

        out = {}

        def get_attention_entry(module, input, output):
            _, _, _, attn_matrix = attention_forward(module, *input)
            out["target"] = torch.where(
                good,
                attn_matrix[:, head_idx, -1, token_idx],
                -torch.finfo(attn_matrix.dtype).max,
            )

        hooks = [
            (model.gpt_neox.layers[layer].attention, get_attention_entry),
        ]

        with add_fwd_hooks(hooks):
            if input_ids is not None:
                output = model(input_ids)
            else:
                output = model(inputs_embeds=inputs_embeds)

        out["logits"] = output.logits
        return out

    return run

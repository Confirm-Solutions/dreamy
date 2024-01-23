import html
from typing import Callable

import matplotlib
import torch


def viz_simple(tokens, attributions, normalize=True, colormap="viridis"):
    # Normalize the attributions to [0, 1]
    if normalize:
        attributions -= attributions.min()
        attributions /= attributions.max()

    # Create an HTML string with colored background for each token
    html_str = """
    <div style='
        background: white;
        padding: 10px;
        color: rgba(0,0,0,1);
        font-size: 15px;
        width: fit-content;'>"""
    cmap = matplotlib.colormaps[colormap]
    for token, attr in zip(tokens, attributions):
        rgba = [int(x * 255) for x in cmap(attr)[:3]]
        color = f"rgba({rgba[0]}, {rgba[1]}, {rgba[2]}, 1.0)"
        html_str += f'<span style="background: {color};">{token}</span>'
    html_str += "</div>"

    # Display the HTML string
    return html_str


def viz(
    tokenizer,
    tokens,
    baseline,
    xentropy,
    resamplings,
    targets,
    target_name="Target",
    colormap="Reds",
):
    resamplings = resamplings.cpu().numpy()
    targets_nonneg = targets.cpu().numpy()
    targets_nonneg[targets_nonneg < 0] = 0

    amax = (baseline - targets_nonneg).max(axis=-1)
    amax /= baseline

    amin = (baseline - targets_nonneg).min(axis=-1)
    amin[amin < 0] = 0
    if amin.max() > 0:
        amin -= amin.min()
        amin /= baseline

    max_pad_top = max((amin * 30).max(), 8)

    # setting font-size in child elements prevents weird extra space problems?
    space = "&nbsp;"
    html_str = f"""
    <div style='
        margin-bottom: 10px;
        padding: 3px;
        background: white;
        color: rgba(0,0,0,1);
        width: fit-content;
        font-size: 0;'>
    <div style='font-size: 15px;'>
    {target_name}: {baseline:.2f} {space * 10}Cross-entropy: {xentropy:.2f}\n
    </div>
    <div style='
        background: white;
        padding-right: 10px;
        padding-bottom: 5px;
        padding-top: {max_pad_top}px;
        color: rgba(0,0,0,1);
        width: fit-content;
        height: fit-content;'>
    """
    cmap = matplotlib.colormaps[colormap]

    def prep_token(t):
        # escape and show whitespace
        return (
            html.escape(t)
            .replace(" ", "&nbsp;")
            .replace("\n", "\\n")
            .replace("\t", "\\t")
            .replace("\r", "\\r")
        )

    for i, token in enumerate(tokens):
        if amax[i] > 0.7:
            font_color = "200,200,200"
        else:
            font_color = "0,0,0"
        rgba = [int(x * 255) for x in cmap(amax[i])[:3]]
        worst_idx = targets[i].argmin()
        worst_token = tokenizer.decode(resamplings[i, worst_idx, i])
        worst_target = targets[i, worst_idx]
        top3_idx = targets[i].topk(k=3).indices.numpy()
        top3_tokens = tokenizer.batch_decode(resamplings[i, top3_idx, i])
        top3_target = targets[i, top3_idx]
        tooltip = f"""
        Worst: <tt>{repr(prep_token(worst_token))}</tt>, {worst_target:.3f}<br>
        Top-3: (<tt>{repr(prep_token(top3_tokens[0]))}</tt>, {top3_target[0]:.3f}),
            (<tt>{repr(prep_token(top3_tokens[1]))}</tt>, {top3_target[1]:.3f}),
            (<tt>{repr(prep_token(top3_tokens[2]))}</tt>, {top3_target[2]:.3f})
        """

        # margin-left lets us overlap the border pixels so that there's only a
        # one pixel border.
        token_style = f"""
        padding-top: {amin[i] * 30}px;
        border: 1px solid #555555;
        margin-left: -1px;
        font-size: 15px;
        background: rgba({rgba[0]}, {rgba[1]}, {rgba[2]}, 1.0);
        color: rgba({font_color},1);
        """
        tooltip_style = """
        display: inline-block;
        visibility: hidden;
        position: absolute;
        background-color: #f9f9f9;
        border: 1px solid #dcdcdc;
        color: #333333;
        padding: 2px;
        border-radius: 3px;
        """
        tooltip_on = """
        onmouseover="this.style.visibility='visible'"
        onmouseout="this.style.visibility='hidden'"
        """
        tooltip_attr = f"""class="viztooltip" style="{tooltip_style}" {tooltip_on}"""
        tooltip = f"""<span {tooltip_attr}>{tooltip}</span>"""

        html_str += (
            f"""<span style="{token_style}">{prep_token(token)}{tooltip}</span>"""
        )
        html_str += """
        <script>
        document.querySelectorAll('.viztooltip').forEach(function(span) {
            span.parentElement.onmouseover = function() {
                span.style.visibility = 'visible';
            }
            span.parentElement.onmouseout = function() {
                span.style.visibility = 'hidden';
            }
        });
        </script>
        """
    html_str += "</div></div>"

    # Display the HTML string
    return html_str


@torch.no_grad()
def resample(model, cache_run: Callable, input_ids, k=range(64), batch_size=256):
    from dreamy.epo import _token_grads

    if len(input_ids.shape) > 1:
        raise ValueError("input_ids must be a 1D tensor of token IDs")

    x_penalty = torch.zeros((input_ids.shape[0],), device=input_ids.device)
    output = _token_grads(
        model,
        cache_run,
        input_ids.unsqueeze(0),
        x_penalty=x_penalty,
        batch_size=batch_size,
    )
    baseline = output.target[0].item()

    seq_len = input_ids.shape[0]
    topk = (-output.token_grads).topk(k=max(k) + 1, dim=-1).indices[..., k]
    targets = torch.empty(seq_len * len(k))
    resamplings = torch.empty((seq_len, len(k), seq_len), dtype=torch.long)
    resamplings[:, :, :] = input_ids.unsqueeze(0).unsqueeze(0)
    resamplings[
        torch.arange(seq_len, device=model.device)[None, :],
        :,
        torch.arange(seq_len, device=model.device)[None, :],
    ] = topk
    resamplings_flat = resamplings.view(-1, seq_len)
    for i in range(0, resamplings_flat.shape[0], batch_size):
        end_i = min(i + batch_size, resamplings_flat.shape[0])
        targets[i:end_i] = cache_run(resamplings_flat[i:end_i].to(model.device))[
            "target"
        ]
    targets = targets.reshape(seq_len, len(k))

    return dict(
        baseline=baseline,
        xentropy=output.xentropy[0].item(),
        targets=targets,
        resamplings=resamplings,
    )


def resample_viz(
    model,
    tokenizer,
    cache_run: Callable,
    input_ids,
    k=range(64),
    batch_size=256,
    colormap="Reds",
    target_name="Target",
):
    attrib = resample(model, cache_run, input_ids, k=k, batch_size=batch_size)
    viz_html = viz(
        tokenizer,
        tokenizer.batch_decode(input_ids),
        colormap=colormap,
        target_name=target_name,
        **attrib,
    )
    return attrib, viz_html

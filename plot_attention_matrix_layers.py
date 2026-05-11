import argparse
import os

import matplotlib.pyplot as plt
import torch

from mechanism_common import (
    DEFAULT_BASE_MODEL,
    DEFAULT_OUTPUT_DIR,
    build_fewshot_messages,
    load_model,
    load_task,
    load_tokenizer,
)


def parse_layers(text):
    layers = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            layers.extend(range(int(start), int(end) + 1))
        else:
            layers.append(int(part))
    return layers


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot full query-key attention matrices for selected transformer layers."
    )
    parser.add_argument("--base_model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--model", default="BASE", choices=["BASE", "GOMOKU_COT", "GOMOKU_NOCOT", "GO_COT", "GO_NOCOT"])
    parser.add_argument("--task", default="object_counting")
    parser.add_argument("--sample_index", type=int, default=3, help="Dataset index. Use 3 to skip 3-shot examples by default.")
    parser.add_argument("--n_shot", type=int, default=3)
    parser.add_argument("--layers", default="24-27", help="Comma list or range, e.g. 20,24-27.")
    parser.add_argument("--head_reduce", default="mean", choices=["mean", "max"], help="How to reduce attention heads.")
    parser.add_argument("--contrast_low", type=float, default=5.0, help="Lower percentile for color contrast.")
    parser.add_argument("--contrast_high", type=float, default=95.0, help="Upper percentile for color contrast.")
    parser.add_argument("--max_tokens", type=int, default=160, help="Crop to the last N tokens if the prompt is long.")
    parser.add_argument("--output_dir", default=f"{DEFAULT_OUTPUT_DIR}/attention_maps")
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def reduce_heads(attn, mode):
    # attn shape: [batch, heads, query, key]
    attn = attn[0]
    if mode == "max":
        return attn.max(dim=0).values
    return attn.mean(dim=0)


def plot_layers(attn_mats, layers, title, output_path, contrast_low, contrast_high, dpi):
    fig_h = max(2.6 * len(layers), 3.2)
    fig, axes = plt.subplots(len(layers), 1, figsize=(9, fig_h), squeeze=False)
    fig.suptitle(title, fontsize=13)

    for ax, layer, mat in zip(axes[:, 0], layers, attn_mats):
        values = mat.detach().float().cpu()
        vmax = torch.quantile(values.flatten(), contrast_high / 100).item()
        vmin = torch.quantile(values.flatten(), contrast_low / 100).item()
        image = ax.imshow(values.numpy(), cmap="plasma", aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_title(f"Layer {layer}", fontsize=11)
        ax.set_xlabel("Key position")
        ax.set_ylabel("Query position")
        fig.colorbar(image, ax=ax, fraction=0.025, pad=0.04)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    layers = parse_layers(args.layers)
    tokenizer = load_tokenizer(args.base_model)
    model = load_model(args.model, base_model=args.base_model)

    fewshot = build_fewshot_messages(args.task, n_shot=args.n_shot)
    ds = load_task(args.task)
    ex = ds[int(args.sample_index)]
    messages = fewshot + [{"role": "user", "content": ex["input"]}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    if inputs.input_ids.shape[-1] > args.max_tokens:
        inputs = {k: v[:, -args.max_tokens:] for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(
            **inputs,
            output_attentions=True,
            use_cache=False,
            return_dict=True,
        )

    n_layers = len(outputs.attentions)
    for layer in layers:
        if layer < 0 or layer >= n_layers:
            raise ValueError(f"Layer {layer} out of range. Model has {n_layers} layers.")

    attn_mats = [reduce_heads(outputs.attentions[layer], args.head_reduce) for layer in layers]
    layer_label = args.layers.replace(",", "_").replace("-", "_")
    output_path = os.path.join(
        args.output_dir,
        f"{args.model.lower()}_{args.task}_sample{args.sample_index}_layers_{layer_label}_{args.head_reduce}.png",
    )
    title = (
        f"{args.model} - {args.task} - Layers {args.layers} "
        f"(plasma colormap, {args.contrast_low:.0f}-{args.contrast_high:.0f}% contrast)"
    )
    plot_layers(attn_mats, layers, title, output_path, args.contrast_low, args.contrast_high, args.dpi)
    print("Saved:", output_path)


if __name__ == "__main__":
    main()

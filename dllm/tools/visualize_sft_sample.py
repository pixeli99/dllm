from __future__ import annotations

import argparse
import html
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any


IGNORE_LABEL = -100


@dataclass
class LaunchDefaults:
    dataset_args: str | None = None
    model_name_or_path: str | None = None
    max_length: int | None = None
    truncation: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load a cached SFT DatasetDict, decode one sample, and mark which "
            "tokens are supervised by labels."
        )
    )
    parser.add_argument(
        "--sh_path",
        type=str,
        default=None,
        help=(
            "Optional launch script to read defaults from, e.g. "
            "/lustre/projects/polyullm/lipengxiang_tmp/dllm/scripts/looped_llada/run.sh"
        ),
    )
    parser.add_argument(
        "--dataset_args",
        type=str,
        default=None,
        help=(
            "Cached dataset path or dataset_args. Overrides --dataset_args found "
            "inside --sh_path."
        ),
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help=(
            "Tokenizer/model path. Overrides --model_name_or_path found inside "
            "--sh_path. Defaults to GSAI-ML/LLaDA-8B-Base."
        ),
    )
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help=(
            "Sample index. With --match_training_view True, this is the index "
            "after the lightweight training-time keep/filter rule."
        ),
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=None,
        help="Training max_length. Defaults to --max_length in --sh_path, then 1024.",
    )
    parser.add_argument(
        "--truncation",
        type=str,
        default=None,
        choices=("right", "filter"),
        help="Training truncation mode. Defaults to --truncation in --sh_path, then right.",
    )
    parser.add_argument(
        "--match_training_view",
        type=str_to_bool,
        default=True,
        help=(
            "Whether to mimic SFT post_process_dataset for this one sample. "
            "Keeps prompt_len <= max_length, then right-truncates."
        ),
    )
    parser.add_argument(
        "--text_char_limit",
        type=int,
        default=12000,
        help="Maximum characters printed for each decoded text block. Use 0 for no limit.",
    )
    parser.add_argument(
        "--token_table_limit",
        type=int,
        default=160,
        help="Maximum token rows printed in the terminal table. Use 0 to skip.",
    )
    parser.add_argument(
        "--html_out",
        type=str,
        default=None,
        help=(
            "Optional absolute HTML output path with ignored tokens greyed out "
            "and supervised tokens highlighted."
        ),
    )
    return parser.parse_args()


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}.")


def read_launch_defaults(sh_path: str | None) -> LaunchDefaults:
    if not sh_path:
        return LaunchDefaults()

    path = Path(sh_path).expanduser().resolve()
    text = path.read_text(encoding="utf-8")
    tokens = shell_tokens_without_comments(text)

    return LaunchDefaults(
        dataset_args=find_flag_value(tokens, "dataset_args"),
        model_name_or_path=find_flag_value(tokens, "model_name_or_path"),
        max_length=parse_optional_int(find_flag_value(tokens, "max_length")),
        truncation=find_flag_value(tokens, "truncation"),
    )


def shell_tokens_without_comments(text: str) -> list[str]:
    code_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        code_lines.append(line)

    joined = "\n".join(code_lines).replace("\\\n", " ")
    return shlex.split(joined, comments=True, posix=True)


def find_flag_value(tokens: list[str], name: str) -> str | None:
    flag = f"--{name}"
    prefix = f"{flag}="
    for idx, token in enumerate(tokens):
        if token == flag and idx + 1 < len(tokens):
            return tokens[idx + 1]
        if token.startswith(prefix):
            return token[len(prefix) :]
    return None


def parse_optional_int(value: str | None) -> int | None:
    return int(value) if value is not None else None


def infer_repo_root(sh_path: str | None) -> Path:
    if sh_path:
        start = Path(sh_path).expanduser().resolve().parent
    else:
        start = Path.cwd().resolve()

    for candidate in (start, *start.parents):
        if (candidate / "dllm").is_dir() and (candidate / "examples").is_dir():
            return candidate
    return Path.cwd().resolve()


def resolve_dataset_args(dataset_args: str, repo_root: Path) -> str:
    pieces = re.split(r"(\s*[|+]\s*)", dataset_args)
    resolved: list[str] = []
    for piece in pieces:
        if not piece.strip() or piece.strip() in {"|", "+"}:
            resolved.append(piece)
            continue
        resolved.append(resolve_one_dataset_spec(piece.strip(), repo_root))
    return "".join(resolved)


def resolve_one_dataset_spec(spec: str, repo_root: Path) -> str:
    dllm = import_dllm()
    name, _ = dllm.utils.parse_spec(spec)
    if not name or not is_relative_local_path(name):
        return spec

    suffix = spec[len(name) :]
    absolute = (repo_root / name).resolve()
    return f"{absolute}{suffix}"


def is_relative_local_path(path: str) -> bool:
    return path.startswith(".") or path.startswith("..")


def choose_sample(
    split_dataset: Any,
    requested_index: int,
    match_training_view: bool,
    truncation: str,
    max_length: int,
) -> tuple[dict[str, Any], int]:
    if requested_index < 0:
        raise ValueError("--index must be >= 0.")

    if not match_training_view:
        return dict(split_dataset[requested_index]), requested_index

    has_prompt_len = "prompt_len" in split_dataset.column_names

    kept = -1
    for raw_idx in range(len(split_dataset)):
        row = dict(split_dataset[raw_idx])
        if should_keep_row(row, truncation, max_length, has_prompt_len):
            kept += 1
        else:
            continue

        if kept == requested_index:
            return clip_row_like_training(row, truncation, max_length), raw_idx

    raise IndexError(
        f"Could not find kept sample index {requested_index} in split after "
        f"applying truncation={truncation!r}, max_length={max_length}."
    )


def should_keep_row(
    row: dict[str, Any], truncation: str, max_length: int, has_prompt_len: bool
) -> bool:
    if truncation == "filter":
        return len(row["input_ids"]) <= max_length
    if truncation == "right" and has_prompt_len:
        return row["prompt_len"] <= max_length
    return True


def clip_row_like_training(
    row: dict[str, Any], truncation: str, max_length: int
) -> dict[str, Any]:
    if truncation == "filter":
        return row
    if truncation != "right":
        raise NotImplementedError(f"Unsupported truncation mode: {truncation}")

    for key in ("input_ids", "labels", "attention_mask"):
        if key in row and isinstance(row[key], list):
            row[key] = row[key][:max_length]
    return row


def import_dllm():
    import dllm

    return dllm


def load_tokenizer(dllm: Any, model_name_or_path: str):
    model_args = dllm.utils.ModelArguments(model_name_or_path=model_name_or_path)
    return dllm.utils.get_tokenizer(model_args=model_args)


def supervised_mask(labels: list[int]) -> list[bool]:
    return [label != IGNORE_LABEL for label in labels]


def decode_ids(tokenizer: Any, token_ids: list[int]) -> str:
    return tokenizer.decode(
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def truncate_text(text: str, char_limit: int) -> str:
    if char_limit <= 0 or len(text) <= char_limit:
        return text
    omitted = len(text) - char_limit
    return f"{text[:char_limit]}\n\n...[truncated {omitted} chars]"


def print_summary(
    *,
    dataset_args: str,
    split: str,
    requested_index: int,
    raw_index: int,
    row: dict[str, Any],
    tokenizer: Any,
    text_char_limit: int,
    token_table_limit: int,
) -> None:
    input_ids = row["input_ids"]
    labels = row.get("labels", [])
    mask = supervised_mask(labels) if labels else []
    loss_token_count = sum(mask)
    ignored_token_count = len(mask) - loss_token_count

    prompt_ids = [tid for tid, is_loss in zip(input_ids, mask) if not is_loss]
    response_ids = [tid for tid, is_loss in zip(input_ids, mask) if is_loss]

    print("=== SFT sample summary ===")
    print(f"dataset_args: {dataset_args}")
    print(f"split: {split}")
    print(f"requested_index: {requested_index}")
    print(f"raw_index: {raw_index}")
    print(f"columns: {sorted(row.keys())}")
    print(f"input_ids_len: {len(input_ids)}")
    if labels:
        print(f"labels_len: {len(labels)}")
        print(f"ignored_label_tokens: {ignored_token_count}")
        print(f"supervised_label_tokens: {loss_token_count}")
    if "prompt_len" in row:
        print(f"prompt_len: {row['prompt_len']}")

    print("\n=== decoded full sample ===")
    print(truncate_text(decode_ids(tokenizer, input_ids), text_char_limit))

    if labels:
        print("\n=== ignored/prompt region ===")
        print(truncate_text(decode_ids(tokenizer, prompt_ids), text_char_limit))
        print("\n=== supervised/loss region ===")
        print(truncate_text(decode_ids(tokenizer, response_ids), text_char_limit))

    if token_table_limit > 0:
        print_token_table(
            tokenizer=tokenizer,
            input_ids=input_ids,
            labels=labels,
            limit=token_table_limit,
        )


def print_token_table(
    tokenizer: Any, input_ids: list[int], labels: list[int], limit: int
) -> None:
    print(f"\n=== token table: first {min(limit, len(input_ids))} tokens ===")
    print("idx\tlabel\tinput_id\tlabel_id\ttoken")
    for idx, token_id in enumerate(input_ids[:limit]):
        label_id = labels[idx] if idx < len(labels) else None
        label_state = "loss" if label_id != IGNORE_LABEL else "ignore"
        token_text = tokenizer.convert_ids_to_tokens(token_id)
        token_text = token_text.replace("\n", "\\n").replace("\t", "\\t")
        print(f"{idx}\t{label_state}\t{token_id}\t{label_id}\t{token_text!r}")

    if len(input_ids) > limit:
        print(f"... omitted {len(input_ids) - limit} tokens")


def write_html(
    *,
    html_out: str,
    row: dict[str, Any],
    tokenizer: Any,
    dataset_args: str,
    split: str,
    requested_index: int,
    raw_index: int,
) -> None:
    path = Path(html_out).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    input_ids = row["input_ids"]
    labels = row.get("labels", [IGNORE_LABEL] * len(input_ids))
    mask = supervised_mask(labels)

    spans = []
    for token_id, is_loss in zip(input_ids, mask):
        piece = decode_ids(tokenizer, [token_id])
        css_class = "loss" if is_loss else "ignore"
        spans.append(
            f'<span class="{css_class}" title="id={token_id}">'
            f"{html.escape(piece)}</span>"
        )

    full_text = html.escape(decode_ids(tokenizer, input_ids))
    supervised_text = html.escape(
        decode_ids(tokenizer, [tid for tid, keep in zip(input_ids, mask) if keep])
    )
    ignored_text = html.escape(
        decode_ids(tokenizer, [tid for tid, keep in zip(input_ids, mask) if not keep])
    )

    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>SFT sample {split}/{requested_index}</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.45;
      margin: 24px;
      color: #17202a;
      background: #fbfbf8;
    }}
    .meta {{
      color: #4f5b67;
      font-size: 14px;
      margin-bottom: 16px;
    }}
    .legend span {{
      display: inline-block;
      margin-right: 12px;
      padding: 2px 8px;
      border-radius: 4px;
      border: 1px solid #ccd2d8;
      font-size: 13px;
    }}
    .sample, pre {{
      white-space: pre-wrap;
      word-break: break-word;
      border: 1px solid #d9dee3;
      border-radius: 6px;
      padding: 14px;
      background: #ffffff;
    }}
    .ignore {{
      color: #7c8793;
      background: #edf0f2;
    }}
    .loss {{
      color: #123524;
      background: #ccebd8;
    }}
    h2 {{
      font-size: 18px;
      margin-top: 24px;
    }}
  </style>
</head>
<body>
  <h1>SFT Sample Visualization</h1>
  <div class="meta">
    dataset_args: {html.escape(dataset_args)}<br>
    split: {html.escape(split)}<br>
    requested_index: {requested_index}<br>
    raw_index: {raw_index}<br>
    input_ids_len: {len(input_ids)}<br>
    supervised_label_tokens: {sum(mask)}<br>
    ignored_label_tokens: {len(mask) - sum(mask)}
  </div>
  <div class="legend">
    <span class="ignore">ignored label / prompt</span>
    <span class="loss">supervised label / loss</span>
  </div>
  <h2>Token-Level View</h2>
  <div class="sample">{''.join(spans)}</div>
  <h2>Full Decoded Text</h2>
  <pre>{full_text}</pre>
  <h2>Ignored / Prompt Region</h2>
  <pre>{ignored_text}</pre>
  <h2>Supervised / Loss Region</h2>
  <pre>{supervised_text}</pre>
</body>
</html>
"""
    path.write_text(document, encoding="utf-8")
    print(f"\nWrote HTML visualization to: {path}")


def main() -> None:
    args = parse_args()
    launch_defaults = read_launch_defaults(args.sh_path)
    repo_root = infer_repo_root(args.sh_path)
    dllm = import_dllm()

    dataset_args = args.dataset_args or launch_defaults.dataset_args
    if not dataset_args:
        raise ValueError("Provide --dataset_args or a --sh_path containing --dataset_args.")
    dataset_args = resolve_dataset_args(dataset_args, repo_root)

    model_name_or_path = (
        args.model_name_or_path
        or launch_defaults.model_name_or_path
        or "GSAI-ML/LLaDA-8B-Base"
    )
    max_length = args.max_length or launch_defaults.max_length or 1024
    truncation = args.truncation or launch_defaults.truncation or "right"

    tokenizer = load_tokenizer(dllm, model_name_or_path)
    dataset = dllm.data.load_sft_dataset(
        dataset_args,
        load_preprocessed_data=True,
    )

    if args.split not in dataset:
        raise KeyError(f"Split {args.split!r} not found. Available splits: {list(dataset)}")

    row, raw_index = choose_sample(
        split_dataset=dataset[args.split],
        requested_index=args.index,
        match_training_view=args.match_training_view,
        truncation=truncation,
        max_length=max_length,
    )

    print_summary(
        dataset_args=dataset_args,
        split=args.split,
        requested_index=args.index,
        raw_index=raw_index,
        row=row,
        tokenizer=tokenizer,
        text_char_limit=args.text_char_limit,
        token_table_limit=args.token_table_limit,
    )

    if args.html_out:
        write_html(
            html_out=args.html_out,
            row=row,
            tokenizer=tokenizer,
            dataset_args=dataset_args,
            split=args.split,
            requested_index=args.index,
            raw_index=raw_index,
        )


if __name__ == "__main__":
    main()

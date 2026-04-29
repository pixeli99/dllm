"""
Loader for nvidia/OpenMathInstruct-2.

Source columns: `problem`, `generated_solution`, `expected_answer`,
`problem_source`, `generation_model`.

We map each row into the repo's standard two-turn `messages` chat:
    user      <- problem
    assistant <- generated_solution   (typically ends with \\boxed{<answer>})

The full split has ~14M rows, so this loader supports *early slicing* via
`train_limit` / `test_limit`. When limits are given we use HF split
slicing (`split="train[:N]"`) so we never map/filter the full 14M rows
just to throw most of it away.

Run:
    # Loader self-test:
    python /Users/pixeli/dllm/dllm/data/openmathinstruct2.py
"""

from datasets import DatasetDict, load_dataset


def load_dataset_openmathinstruct2(
    dataset_name_or_path: str,
    train_limit: int | None = None,
    test_limit: int | None = None,
    num_proc: int = 8,
) -> DatasetDict:
    """
    Load nvidia/OpenMathInstruct-2 and return a DatasetDict with a
    `messages` column and a train/test split.

    Args:
        dataset_name_or_path: "nvidia/OpenMathInstruct-2" or a local path.
        train_limit:  If set, cap train split at this many rows.
        test_limit:   If set, cap test split at this many rows. When neither
                      limit is given, defaults to a 1%% test split.
        num_proc: Worker count for `map` / `filter`.

    Returns:
        DatasetDict({"train": Dataset, "test": Dataset}) where each row has
        a single column `messages: List[Dict[str, str]]`.
    """
    # Early slice at load time so 14M rows don't get mapped/filtered for nothing.
    if train_limit is not None or test_limit is not None:
        total_needed = (train_limit or 0) + (test_limit or 0)
        # Headroom for rows dropped by the filter + a small absolute buffer.
        total_with_buffer = int(total_needed * 1.02) + 1000
        split_spec = f"train[:{total_with_buffer}]"
    else:
        split_spec = "train"

    ds = load_dataset(dataset_name_or_path, split=split_spec)

    def map_fn(ex):
        return {
            "messages": [
                {"role": "user", "content": (ex.get("problem") or "").strip()},
                {
                    "role": "assistant",
                    "content": (ex.get("generated_solution") or "").strip(),
                },
            ]
        }

    ds = ds.map(map_fn, remove_columns=ds.column_names, num_proc=num_proc)
    ds = ds.filter(
        lambda r: bool(r["messages"][0]["content"])
        and bool(r["messages"][1]["content"]),
        num_proc=num_proc,
    )

    # Split
    total_rows = len(ds)
    if train_limit is None and test_limit is None:
        return DatasetDict(ds.train_test_split(test_size=0.01, seed=42))

    if train_limit is not None and test_limit is not None:
        n_train = min(int(train_limit), total_rows)
        n_test = min(int(test_limit), max(0, total_rows - n_train))
    elif train_limit is not None:
        n_test = max(100, int(total_rows * 0.01))
        n_train = min(int(train_limit), max(1, total_rows - n_test))
    else:
        n_test = min(int(test_limit), total_rows)
        n_train = max(1, total_rows - n_test)

    n_total_used = n_train + n_test
    ds = ds.select(range(n_total_used))
    return DatasetDict(ds.train_test_split(test_size=n_test, seed=42))


if __name__ == "__main__":
    from dllm.utils import resolve_with_base_env

    path = resolve_with_base_env("nvidia/OpenMathInstruct-2", "BASE_DATASETS_DIR")
    ds = load_dataset_openmathinstruct2(path, train_limit=20, test_limit=5)
    print(ds)
    print("sample:", ds["train"][0])

from pathlib import Path
from typing import Union, Optional, List

from datasets import load_dataset
import transformers


def get_data(
    tokenizer,
    data_dir: Union[str, Path] = '/home/test/test07/data/slimpj_6b',
    streaming: bool = True,
    n_workers: int = 8,
    overwrite_cache: bool = False,
    token_ids_only: bool = True,
    max_len: int = 1024,
    eos_token_id: Optional[int] = None,
):
    '''
    Returns an iterable of batches of token IDs.

    This will use `load_dataset` from the HuggingFace Datasets library to load the
    data from `data_dir`, tokenize each example, concatenate the input IDs, add an
    EOS token ID at the end of each sequence, then split into chunks of `max_len`
    tokens, and return a tensor of (batch_size, max_len).
    '''
    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    
    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id
    
    data_dir = Path(data_dir)
    data_files = {
        "train": sorted(map(str, data_dir.glob("*.jsonl"))),
        'validation': [str(data_dir / 'slimpajama-0.jsonl')],
        'test': [str(data_dir / 'slimpajama-1.jsonl')],
    }
    print(f">> Loading data from {str(data_dir)}, {streaming = }")
    raw_dataset = load_dataset(
        str(data_dir), data_files=data_files, streaming=streaming
    )

    text_column_name = 'text'

    # Tokenize in streaming mode
    def tokenize_function(examples: dict) -> List[List[int]]:
        texts = examples[text_column_name]
        encodings = tokenizer(texts)
        batch_ids: List[List[int]] = encodings['input_ids']
        for ids in batch_ids:
            ids += [eos_token_id]
        concat_ids = sum(batch_ids, [])  # Concatenate into one long ids
        total_len = len(concat_ids)

        chunked_ids: List[List[int]] = []
        chunk_len = max_len
        print(total_len, chunk_len)
        for i in range(0, total_len, chunk_len):
            this_chunk: List[int] = concat_ids[i:i + chunk_len]
            chunked_ids.append(this_chunk)
        batch = {
            'input_ids': chunked_ids,
            'labels': chunked_ids.copy(),
        }
        return batch

    if streaming:
        print(">> Tokenizing data on the fly...")
        tokenized_dataset = raw_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=[text_column_name] if token_ids_only else []
        )
    else:
        print(f"Tokenizing data...")
        tokenized_dataset = raw_dataset.map(
            tokenize_function,
            batched=True,
            num_proc=n_workers,
            remove_columns=[text_column_name] if token_ids_only else [],
            load_from_cache_file=not overwrite_cache,
            desc="Running tokenizer on dataset",
        )
    return tokenized_dataset

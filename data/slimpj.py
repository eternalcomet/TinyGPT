from pathlib import Path
# import logging

from datasets import load_dataset
import transformers
# from transformers.testing_utils import CaptureLogger

from args import DataArguments, ModelArguments


# logger = logging.getLogger(__name__)


def get_data(data_path='/home/test/test07/data/slimpj_6b', streaming=True):
    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    data_dir = Path(data_dir)
    data_files = {
        "train": sorted(map(str, data_dir.glob("*.jsonl"))),
        'validation': [str(data_dir / 'slimpajama-0.jsonl')],
        'test': [str(data_dir / 'slimpajama-1.jsonl')],
    }
    print(f">> Loading data from {str(data_dir)}, {streaming = }")
    raw_datasets = load_dataset(
        str(data_dir), data_files=data_files, streaming=streaming
    )

    text_column_name = 'text'

    # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    # tok_logger = transformers.utils.logging.get_logger(
    #     "transformers.tokenization_utils_base"
    # )

    def tokenize_function(examples):
        # with CaptureLogger(tok_logger) as cl:
        output = tokenizer(examples[text_column_name])
        # # clm input could be much much longer than block_size
        # if "Token indices sequence length is longer than the" in cl.out:
        #     tok_logger.warning(
        #         "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
        #         " before being passed to the model."
        #     )
        return output

    with training_args.main_process_first(desc="dataset map tokenization"):
        if not data_args.streaming:
            print(f"Tokenizing data...")
            tokenized_dataset = raw_dataset.map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                # remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )
        else:
            print(">> Tokenizing data on the fly...")
            tokenized_dataset = raw_dataset.map(
                tokenize_function,
                batched=True,
                # remove_columns=column_names,
            )
    return tokenized_dataset

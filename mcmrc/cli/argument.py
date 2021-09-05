from dataclasses import dataclass, field
from typing import Optional

@dataclass
class BasicModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )


@dataclass
class BasicDataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file"}
    )
    data_dir: Optional[str] = field(default=None, metadata={"help": "the local path of input data"})
    dataset: str = field(
        default='race',
        metadata={"help": "name of the used dataset, race or dream. Default: race."}
    )
    dataload_script: Optional[str] = field(
        default=None,
        metadata={"help": "path to the dataset processing script with the dataset builder. Can be either:a local path "
                          "to processing script or the directory containing the script (if the script has the same "
                          "name as the directory),e.g. ``'./dataset/squad'`` or ``'./dataset/squad/squad.py'"}
    )
    dataload_split: Optional[str] = field(
        default=None,
        metadata={"help": "the type (or say 'category') needs to be loaded. For 'race' dataset, it can be chosen from "
                          "'middle', 'high' or 'all'. For 'dream' dataset, it should be 'plain_text'. May be more "
                          "dataset will be included."}
    )
    eval_dataset: Optional[str] = field(
        default="all",
        metadata={"help": "the eval dataset,'dev', 'test' or 'all' (means both 'dev' and 'test'). default: all"}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_qa_length: int = field(
        default=64,
        metadata={
            "help":     "The maximum total input sequence length after WordPiece tokenization. "
                        "Sequences longer than this will be truncated, and sequences shorter "
                        "than this will be padded."
        },
    )
    max_seq_length: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. If passed, sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to the maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )

    def __post_init__(self):
        if self.train_file is not None and self.train_file.lower() == 'none':
            self.train_file = None
        if self.validation_file is not None and self.validation_file.lower() == 'none':
            self.validation_file = None
        if self.test_file is not None and self.test_file.lower() == 'none':
            self.test_file = None


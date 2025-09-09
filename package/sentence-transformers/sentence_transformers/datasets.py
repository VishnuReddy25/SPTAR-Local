import torch
from torch.utils.data import Dataset
from typing import List, Union
import logging

logger = logging.getLogger(__name__)

class SentencesDataset(Dataset):
    """
    Dataset for smart batching, that is each batch is only padded to its longest sequence instead of padding all
    sequences to the max length.
    The SentenceBertEncoder.smart_batching_collate is required for this to work.
    SmartBatchingDataset does *not* work with DataLoader shuffle=True, set shuffle=False.

    This dataset should only be used with SentenceTransformer for inference or with SentenceTransformer and
    MultipleNegativesRankingLoss. For training with other losses, use InputExample and a label-aware data collator.

    :param examples: the examples to use. Can be of type `InputExample`, `str`, or `List[str]`
    :param model: the SentenceTransformer model. Can be None, but then a collation_function must be used that tokenizes the examples
    :param max_seq_length: the maximum sequence length
    :param convert_to_tensor: whether to convert the input to tensors
    :param show_progress_bar: whether to show a progress bar
    """

    def __init__(self, examples: List[Union[str, List[str], dict]], model=None, max_seq_length: int = 512, convert_to_tensor: bool = True, show_progress_bar: bool = False):
        self.model = model
        self.max_seq_length = max_seq_length
        self.convert_to_tensor = convert_to_tensor
        self.show_progress_bar = show_progress_bar

        if isinstance(examples[0], dict):
            # InputExample format
            self.examples = examples
        elif isinstance(examples[0], str):
            # Single sentences
            self.examples = [{"texts": [example]} for example in examples]
        elif isinstance(examples[0], list) and isinstance(examples[0][0], str):
            # List of sentences
            self.examples = [{"texts": example} for example in examples]
        else:
            raise ValueError("Invalid example format")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]

class ParallelSentencesDataset:
    """
    This dataset can be used for knowledge distillation with a teacher and student model.
    The dataset consists of sentence pairs. For each sentence pair, the teacher model computes
    an embedding. The student model is trained to match these teacher embeddings.

    :param student_model: Student model that should be trained to match the teacher embeddings
    :param teacher_model: Teacher model to compute the target embeddings
    :param batch_size: Batch size for computing the teacher embeddings
    :param use_embedding_cache: Whether to cache the teacher embeddings
    """

    def __init__(self, student_model, teacher_model, batch_size: int = 8, use_embedding_cache: bool = True):
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.batch_size = batch_size
        self.use_embedding_cache = use_embedding_cache
        self.datasets = []
        self.teacher_embeddings = []

    def add_dataset(self, sentences: List[List[str]], max_sentence_length: int = 256):
        """
        Add a dataset to the ParallelSentencesDataset

        :param sentences: List of sentence pairs
        :param max_sentence_length: Maximum sentence length
        """
        self.datasets.append({
            'sentences': sentences,
            'max_sentence_length': max_sentence_length
        })

    def __len__(self):
        return sum(len(dataset['sentences']) for dataset in self.datasets)

    def __getitem__(self, idx):
        # Find which dataset this index belongs to
        dataset_idx = 0
        local_idx = idx
        for i, dataset in enumerate(self.datasets):
            if local_idx < len(dataset['sentences']):
                dataset_idx = i
                break
            local_idx -= len(dataset['sentences'])

        sentences = self.datasets[dataset_idx]['sentences'][local_idx]
        return {
            'sentences': sentences,
            'teacher_embedding': None  # Will be computed in collate_fn
        }

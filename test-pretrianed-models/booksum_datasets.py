import csv
import json
import os

import datasets


_URLS = {
    "chapters": "",
}

_CITATION = """
"""

_DESCRIPTION = """
"""

_DOCUMENT = "text"
_SUMMARY = "summary"


class BookSumConfig(datasets.BuilderConfig):
    """BuilderConfig for Scientific Papers."""

    def __init__(self, filename=None, **kwargs):
        """BuilderConfig for ScientificPapers
        Args:
          filename: filename of different configs for the dataset.
          **kwargs: keyword arguments forwarded to super.
        """
        # 1.1.0 remove sentence breaker <S> and </S> in summary.
        super(BookSumConfig, self).__init__(version=datasets.Version("1.1.1"), **kwargs)
        self.filename = filename


class BookSum(datasets.GeneratorBasedBuilder):
    """Scientific Papers."""

    BUILDER_CONFIGS = [
        BookSumConfig(name="chapters", description="chapters"),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    _DOCUMENT: datasets.Value("string"),
                    _SUMMARY: datasets.Value("string"),
                }
            ),
            supervised_keys=None,
            homepage="https://github.com/armancohan/long-summarization",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        ## TODO: Put the path to the booksum dataset and rename the train.jsonl, 
        ##       val.jsonl and test.jsonl to the appropriate file paths. 
        path = "/Users/zkjzou/Desktop/Research/BookSum/BookSum Data 2"
        
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"path": os.path.join(path, "train.jsonl")},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"path": os.path.join(path, "val.jsonl")},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"path": os.path.join(path, "test.jsonl")},
            ),
        ]

    def _generate_examples(self, path=None):
        """Yields examples."""
        with open(path, encoding="utf-8") as f:
            for line in f:

                d = json.loads(line)
                document = "\n".join(d["text"])
                summary = "\n".join(d["summary"])


                yield {
                    _DOCUMENT: document,
                    _SUMMARY: summary,
                }

if __name__ == '__main__':
    ## TODO: to test if the file works, change the path to this file. 
    dataset = datasets.load_dataset('/Users/zkjzou/Desktop/Research/BookSum/Long-Document/test-pretrianed-models/booksum_datasets.py', 'chapters')

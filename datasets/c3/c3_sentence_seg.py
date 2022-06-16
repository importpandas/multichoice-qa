# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""C3 Parallel Corpora"""

import json
import spacy

import datasets
from mcmrc.data_utils.processors import process_text

_CITATION = """\
 @article{sun2019investigating,
   title={Investigating Prior Knowledge for Challenging Chinese Machine Reading Comprehension},
   author={Sun, Kai and Yu, Dian and Yu, Dong and Cardie, Claire},
   journal={Transactions of the Association for Computational Linguistics},
   year={2020},
   url={https://arxiv.org/abs/1904.09679v3}
 }
 """

_DESCRIPTION = """\
 Machine reading comprehension tasks require a machine reader to answer questions relevant to the given document. In this paper, we present the first free-form multiple-Choice Chinese machine reading Comprehension dataset (C^3), containing 13,369 documents (dialogues or more formally written mixed-genre texts) and their associated 19,577 multiple-choice free-form questions collected from Chinese-as-a-second-language examinations.
 We present a comprehensive analysis of the prior knowledge (i.e., linguistic, domain-specific, and general world knowledge) needed for these real-world problems. We implement rule-based and popular neural methods and find that there is still a significant performance gap between the best performing model (68.5%) and human readers (96.0%), especially on problems that require prior knowledge. We further study the effects of distractor plausibility and data augmentation based on translated relevant datasets for English on model performance. We expect C^3 to present great challenges to existing systems as answering 86.8% of questions requires both knowledge within and beyond the accompanying document, and we hope that C^3 can serve as a platform to study how to leverage various kinds of prior knowledge to better understand a given written or orally oriented text.
 """

_URL = "https://raw.githubusercontent.com/nlpdata/c3/master/data/"
nlp = spacy.load("zh_core_web_sm")

class C3Config(datasets.BuilderConfig):
    """BuilderConfig for NewDataset"""

    def __init__(self, type_, **kwargs):
        """
        Args:
            pair: the language pair to consider
            zip_file: The location of zip file containing original data
            **kwargs: keyword arguments forwarded to super.
        """
        self.type_ = type_
        super().__init__(**kwargs)


class C3(datasets.GeneratorBasedBuilder):
    """C3 is the first free-form multiple-Choice Chinese machine reading Comprehension dataset, containing 13,369 documents (dialogues or more formally written mixed-genre texts) and their associated 19,577 multiple-choice free-form questions collected from Chinese-as-a-second language examinations."""

    VERSION = datasets.Version("1.0.0")

    # This is an example of a dataset with multiple configurations.
    # If you don't want/need to define several sub-sets in your dataset,
    # just remove the BUILDER_CONFIG_CLASS and the BUILDER_CONFIGS attributes.
    BUILDER_CONFIG_CLASS = C3Config
    BUILDER_CONFIGS = [
        C3Config(
            name="mixed",
            description="Mixed genre questions",
            version=datasets.Version("1.0.0"),
            type_="mixed",
        ),
        C3Config(
            name="dialog",
            description="Dialog questions",
            version=datasets.Version("1.0.0"),
            type_="dialog",
        ),
        C3Config(
            name="all",
            description="all questions",
            version=datasets.Version("1.0.0"),
            type_="all",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # datasets.features.FeatureConnectors
            features=datasets.Features(
                {
                    "example_id": datasets.Value("string"),
                    "article": datasets.Value("string"),
                    "article_sent_start": datasets.Sequence(datasets.Value("int32")),
                    "answer": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "options": datasets.features.Sequence(datasets.Value("string"))
                    # These are the features of your dataset like images, labels ...
                }
            ),
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
            # Homepage of the dataset for documentation
            homepage="https://github.com/nlpdata/c3",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        # m or d
        T = self.config.type_[0]
        if T == "a":
            train_files = [_URL + f"c3-{t}-train.json" for t in ['m', 'd']]
            T = 'm'
        else:
            train_files = _URL + f"c3-{T}-train.json"
        eval_files = [_URL + f"c3-{T}-{split}.json" for split in ["test", "dev"]]
        train_dl_dir = dl_manager.download_and_extract(train_files)
        eval_dl_dir = dl_manager.download_and_extract(eval_files)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filename": train_dl_dir,
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filename": eval_dl_dir[0],
                    "split": "test",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filename": eval_dl_dir[1],
                    "split": "dev",
                },
            ),
        ]

    def _generate_examples(self, filename, split):
        """Yields examples."""
        if not isinstance(filename, list):
            filename_list = [filename]
        else:
            filename_list = filename
        counter = 0
        for filename in filename_list:
            with open(filename, "r", encoding="utf-8") as sf:
                data = json.load(sf)
                for id_, (documents, questions, document_id) in enumerate(data):
                    document = documents[0]
                    article = process_text(document)
                    doc = nlp(article)
                    article_sent_start = [sent.start_char for sent in doc.sents]
                    for i, sent_start in enumerate(article_sent_start):
                        if i > 0 and sent_start - article_sent_start[i - 1] <= 3:
                            article_sent_start.pop(i)

                    for i, que in enumerate(questions):
                        options = que["choice"]
                        options += [''] * (4 - len(options))
                        answer = que["answer"]
                        label = options.index(answer)
                        label = chr(label + ord('A'))
                        yield counter, {
                            "example_id": document_id + '-' + str(i),
                            "article": article,
                            "article_sent_start": article_sent_start,
                            "question": que["question"],
                            "answer": label,
                            "options": que["choice"],
                        }
                        counter += 1

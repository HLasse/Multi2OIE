from typing import Callable, Dict, Iterable, Iterator, List

import spacy
from spacy import Vocab
from spacy.language import Language
from spacy.pipeline import TrainablePipe
from spacy.tokens import Doc
from spacy.util import minibatch
from spacy_transformers.align import get_alignment
from transformers import AutoTokenizer

from .knowledge_triplets import KnowledgeTriplets
from .util import (
    install_extension,
    wp2tokid,
    wp_span_to_token,
    match_extraction_spans_to_wp,
)


@Language.factory(
    "relation_extractor",
    default_config={
        "confidence_threshold": 2.7,
        "labels": [],
        "model_args": {
            "batch_size": 64,
        },
    },
)
def make_relation_extractor(
    nlp: Language,
    name: str,
    confidence_threshold: float,
    labels: List,
    model_args: Dict,
):
    return SpacyRelationExtractor(
        nlp.vocab,
        name=name,
        confidence_threshold=confidence_threshold,
        labels=labels,
        model_args=model_args,
    )


class SpacyRelationExtractor(TrainablePipe):
    """spaCy pipeline component that adds a multilingual relation-extraction component.
    The extractions are saved in the doc._.relation_triplets, ._.relation_head,
    ._.relation_relation, and ._.relation_tail attributes.
    Args:
        vocab (Vocab): The Vocab object for the pipeline.
        confidence_threshold (float): A threshold for model confidence to filter uncertain relations by.
        model_args (Dict): Keyword arguments for KnowledgeTriplets (e.g. batch_size, path)
        name (str): spaCy internal
        labels (List(str)): Required for TrainablePipe but unused. Leave as empty list (or don't)
    """

    def __init__(
        self,
        vocab: Vocab,
        name: str,
        labels: List[str],
        confidence_threshold: float,
        model_args,
    ):
        self.name = name
        self.vocab = vocab
        self.model = KnowledgeTriplets(**model_args)
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
        self.confidence_threshold = confidence_threshold

        [
            install_extension(ext)
            for ext in [
                "relation_triplets",
                "relation_head",
                "relation_relation",
                "relation_tail",
                "relation_confidence",
            ]
        ]

    def set_annotations(self, doc: Iterable[Doc], predictions: Dict) -> None:
        """Assign the extracted features to the Doc objects. Extractions below the
        confidence threshold are filtered, wordpieces and spacy tokens
        are aligned and then attributes are set .


        Assign the extracted features to the Doc object. Extractions below the
        confidence threshold are filtered, wordpieces and spacy tokens
        are aligned and then attributes are set .
        Args:
            docs (Iterable[Doc]): The documents to modify.
            predictions: (Dict): A batch of outputs from KnowledgeTriplets.extract_relations().
        """
        # remove empty docs
        if not doc:
            return
        # get nested list of indices above confidence threshold
        filtered_indices = [
            [
                idx
                for idx, conf in enumerate(confidences)
                if conf > self.confidence_threshold
            ]
            for confidences in (predictions["confidence"])
        ]
        # only keep relations above the threshold
        for key in ["extraction", "extraction_span", "confidence"]:
            predictions[key] = [
                [values[filter_idx] for filter_idx in indices]
                for indices, values in zip(filtered_indices, predictions[key])
            ]

        # concatenate wordpieces and concatenate extraction span. Handle new extraction spans by adding the length of the previous
        # Join wordpieces to single list
        wordpieces = [j for i in predictions["wordpieces"] for j in i]
        # Concatenate extraction spans.

        matched_extractions = match_extraction_spans_to_wp(
            predictions["extraction_span"], predictions["wordpieces"]
        )

        align = get_alignment([doc], [wordpieces], self.tokenizer.all_special_tokens)
        wp2tokids = wp2tokid(align)
        aligned_extractions = wp_span_to_token(matched_extractions, wp2tokids, doc)

        # Set doc level attributes
        merged_confidence = [j for i in predictions["confidence"] for j in i]
        setattr(doc._, "relation_triplets", aligned_extractions["triplet"])
        setattr(doc._, "relation_confidence", merged_confidence)
        setattr(doc._, "relation_head", aligned_extractions["head"])
        setattr(doc._, "relation_relation", aligned_extractions["relation"])
        setattr(doc._, "relation_tail", aligned_extractions["tail"])

    def __call__(self, doc: Doc) -> Doc:
        """Apply the pipe to one document. The document is modified in place,
        and returned. This usually happens under the hood when the nlp object
        is called on a text and all components are applied to the Doc.
        docs (Doc): The Doc to process.
        RETURNS (Doc): The processed Doc.
        DOCS: https://spacy.io/api/transformer#call
        """
        outputs = self.predict([doc])
        self.set_annotations([doc], outputs)
        return doc

    def pipe(self, stream: Iterable[Doc], *, batch_size: int = 128) -> Iterator[Doc]:
        """Apply the pipe to a stream of documents. This usually happens under
        the hood when the nlp object is called on a text and all components are
        applied to the Doc. Batch size is controlled by `batch_size` when
        instatiating the nlp.pipe object.
        stream (Iterable[Doc]): A stream of documents.
        batch_size (int): The number of documents to buffer.
        YIELDS (Doc): Processed documents in order.
        DOCS: https://spacy.io/api/transformer#pipe
        """
        for doc in stream:
            sents = [sent.text for sent in doc.sents]
            self.set_annotations(doc, self.predict(sents))
            yield doc

    def predict(self, docs: Iterable[Doc]) -> Dict:
        """Apply the pipeline's model to a batch of docs, without modifying them.
        Returns the extracted features as the FullTransformerBatch dataclass.
        docs (Iterable[Doc]): The documents to predict.
        RETURNS (Dict): The extracted features.
        DOCS: https://spacy.io/api/transformer#predict
        """
        return self.model.extract_relations(docs)


if __name__ == "__main__":
    nlp = spacy.load("da_core_news_sm")

    test_sents = [
        "Pernille Blume vinder delt EM-sølv i Ungarn.",
        "Pernille Blume blev nummer to ved EM på langbane i disciplinen 50 meter fri.",
        "Hurtigst var til gengæld hollænderen Ranomi Kromowidjojo, der sikrede sig guldet i tiden 23,97 sekunder.",
        "Og at formen er til en EM-sølvmedalje tegner godt, siger Pernille Blume med tanke på, at hun få uger siden var smittet med corona.",
        "Ved EM tirsdag blev det ikke til medalje for den danske medley for mixede hold i 4 x 200 meter fri.",
        "In a phone call on Monday, Mr. Biden warned Mr. Netanyahu that he could fend off criticism of the Gaza strikes for only so long, according to two people familiar with the call",
        "That phone call and others since the fighting started last week reflect Mr. Biden and Mr. Netanyahu’s complicated 40-year relationship.",
        "Politiet skal etterforske Siv Jensen etter mulig smittevernsbrudd.",
        "En av Belgiens mest framträdande virusexperter har flyttats med sin familj till skyddat boende efter hot från en beväpnad högerextremist.",
    ]

    config = {"confidence_threshold": 1.0, "model_args": {"batch_size": 10}}

    # model = KnowledgeTriplets()

    nlp.add_pipe("relation_extractor", config=config)

    pipe = nlp.pipe(test_sents)

    for d in pipe:
        print(d.text, "\n", d._.relation_triplets)

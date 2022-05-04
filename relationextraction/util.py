from typing import Dict, List, Optional, Tuple

import numpy as np
import spacy
import torch
from spacy.language import Language
from spacy.tokens import Doc, Span
from spacy_transformers.align import get_alignment, get_token_positions
from spacy_transformers.data_classes import TransformerData
from thinc.api import torch2xp
from thinc.types import Ragged
from transformers import AutoTokenizer
from transformers.file_utils import ModelOutput

from relationextraction import KnowledgeTriplets


def split_by_doc(self) -> List[TransformerData]:
    """
    Split a TransformerData that represents a batch into a list with
    one TransformerData per Doc.
    This is essentially the same as in:
    https://github.com/explosion/spacy-transformers/blob/5a36943fccb66b5e7c7c2079b1b90ff9b2f9d020/spacy_transformers/data_classes.py
    """
    flat_spans = []
    for doc_spans in self.spans:
        flat_spans.extend(doc_spans)
    token_positions = get_token_positions(flat_spans)
    outputs = []
    start = 0
    prev_tokens = 0
    for doc_spans in self.spans:
        if len(doc_spans) == 0 or len(doc_spans[0]) == 0:
            outputs.append(TransformerData.empty())
            continue
        start_i = token_positions[doc_spans[0][0]]
        end_i = token_positions[doc_spans[-1][-1]] + 1
        end = start + len(doc_spans)
        doc_tokens = self.wordpieces[start:end]
        doc_align = self.align[start_i:end_i]
        doc_align.data = doc_align.data - prev_tokens
        model_output = ModelOutput()
        logits = self.model_output.logits
        for key, output in self.model_output.items():
            if isinstance(output, torch.Tensor):
                model_output[key] = torch2xp(output[start:end])
            elif (
                isinstance(output, tuple)
                and all(isinstance(t, torch.Tensor) for t in output)
                and all(t.shape[0] == logits.shape[0] for t in output)
            ):
                model_output[key] = [torch2xp(t[start:end]) for t in output]
        outputs.append(
            TransformerData(
                wordpieces=doc_tokens,
                model_output=model_output,
                align=doc_align,
            )
        )
        prev_tokens += doc_tokens.input_ids.size
        start += len(doc_spans)
    return outputs


#### Wordpiece <-> spacy alignment functions
def wp2tokid(align: Ragged) -> Dict[int, int]:
    """Gets the wordpiece to token id mapping"""
    wp2tokid = []
    for i, length in enumerate(align.lengths):
        wp2tokid += length * [i]

    return {int(wp_id): t_id for wp_id, t_id in zip(align.data, wp2tokid)}


def wp_span_to_token(
    relation_span: List[List[int]], wp_tokenid_mapping: Dict, doc: Doc
) -> List[Dict[str, List[Span]]]:
    """
    Converts the wp span for each relation to spans.
    Assumes that relations are contiguous"""
    relations = {"triplet": [], "head": [], "relation": [], "tail": []}
    for triplet in relation_span:
        # convert list of wordpieces in the extraction to a tuple of the span (start, end)
        head = get_wp_span_tuple(triplet[0])
        relation = get_wp_span_tuple(triplet[1])
        tail = get_wp_span_tuple(triplet[2])

        # convert the wp span to token span
        head = wp_to_token_id_mapping(head, wp_tokenid_mapping)
        relation = wp_to_token_id_mapping(relation, wp_tokenid_mapping)
        tail = wp_to_token_id_mapping(tail, wp_tokenid_mapping)

        # convert token span to spacy span
        head = token_span_to_spacy_span(head, doc)
        relation = token_span_to_spacy_span(relation, doc)
        tail = token_span_to_spacy_span(tail, doc)

        relations["head"].append(head)
        relations["relation"].append(relation)
        relations["tail"].append(tail)
        relations["triplet"].append((head, relation, tail))
    return relations


def get_wp_span_tuple(span: List[int]) -> Tuple[int, int]:
    """Converts the relation span to a tuple, assumes that extractions are contiguous"""
    if not span:
        return ""
    if len(span) == 1:
        return (span[0], span[0])
    else:
        return (span[0], span[-1])


def wp_to_token_id_mapping(
    span: Tuple[int, int], wp_tokenid_mapping: Dict[int, int]
) -> Tuple[int, int]:
    """converts wordpiece spans to token ids"""
    if span:
        return (wp_tokenid_mapping[span[0]], wp_tokenid_mapping[span[1]])
    else:
        return ""


def token_span_to_spacy_span(span: Tuple[int, int], doc: Doc):
    """converts token id span to span"""
    if not span:
        return ""
    else:
        return doc[span[0] : span[1] + 1]


def install_extension(doc_attr) -> None:
    if not Doc.has_extension(doc_attr):
        Doc.set_extension(doc_attr, default=None)


def match_extraction_spans_to_wp(
    extraction_spans: List[List[List[List[str]]]], wordpieces: List[str]
):
    """Correct the extracted spans from the model to match wordpieces in sentences that are not the first.

    Args:
        extraction_spans (List[List[List[List[str]]]]): The extracted spans
        wordpieces (List[str]): Wordpieces

    """
    max_wp_idx = 0
    matched_extractions = []
    for i, sent_span in enumerate(extraction_spans):
        new_spans = []
        if i > 0:
            max_wp_idx += len(wordpieces[i - 1])
        for triplet in sent_span:
            trip = []
            for relation_type in triplet:
                rel = []
                for val in relation_type:
                    rel.append(val + max_wp_idx)
                trip.append(rel)
            new_spans.append(trip)

        matched_extractions += new_spans
    return matched_extractions

# RelationExtraction

A Python library for extracting knowledge triplets from a text document.

## :wrench: Installation

### Prerequisites

- Python 3.7 or above

- CUDA 10.0 or above

##### using  'pip' command
~~~~
pip install git+https://github.com/HLasse/Multi2OIE
~~~~

## :woman_technologist: Usage

```python

from relationextraction import KnowledgeTriplets


test_sents = [
    "Lasse er en dreng på 26 år.",
    "Jeg arbejder som tømrer",
    "Albert var videnskabsmand og døde i 1921",
    "Lasse lives in Denmark and owns two cats",
]


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


# initialize a class object
# call the class method for extracting triplets from a given list of sentences

relations = KnowledgeTriplets()
final_result = relations.extract_relations(test_sents)


print(final_result ["sentence"])
print(final_result ["extraction_3"])
```

## With spaCy
```py
from relationextraction import SpacyRelationExtractor
import spacy


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


# change these to your purposes. 2.7 is the default confidence threshold(the bulk of bad relations not kept and the majority of correct ones kept)
# batch_size should be changed according to your device. Can most likely be bumped up a fair bit
config = {"confidence_threshold": 2.7, "model_args": {"batch_size": 10}}
nlp.add_pipe("relation_extractor", config=config)

pipe = nlp.pipe(test_sents)

for d in pipe:
    print(d.text, "\n", d._.relation_triplets)

```



## ISSUE:
- Run the model sentence wise




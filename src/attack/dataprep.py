from collections import defaultdict
from typing import Dict, Tuple

import datasets
import torch
from transformers import AutoModel

from . import e5_utils


def create_sample_dataset() -> Tuple[torch.Tensor, Dict]:
    """Creates a sample dataset for testing.

    Returns:
        Tuple[torch.Tensor, Dict]: A tuple containing the embeddings and batch dict,
        which are the output of the tokenizer (token_id, padding, etc).
    """
    # input_texts = [
    #     'query: how much protein should a female eat',
    #     'query: summit define',
    #     ("passage: As a general guideline, the CDC's average requirement of "
    #      "protein for women ages 19 to 70 is 46 grams per day. But, as you "
    #      "can see from this chart, you'll need to increase that if you're "
    #      "expecting or training for a marathon. Check out the chart below to "
    #      "see how much protein you should be eating each day."),
    #     ("passage: Definition of summit for English Language Learners. : "
    #      "1  the highest point of a mountain : the top of a mountain. : "
    #      "2  the highest level. : 3  a meeting or series of meetings between "
    #      "the leaders of two or more governments."),
    # ]

    input_texts = [
        "query: What is interesting about the city of Seattle?",
        "query: What are some of the elements of the periodic table?",
        "query: What wars were fought in the 20th century?",
        "passage: SAP is a software company.",
        "passage: ABAP is a programming language.",
        "passage: SuccessFactors is a human capital management company owned by SAP.",
    ]

    model = AutoModel.from_pretrained("intfloat/e5-small")

    embeddings, batch_dict = e5_utils.run_e5(model, input_texts)  # BSV

    assert isinstance(embeddings, torch.Tensor)
    assert embeddings.dim() == 2, embeddings.dim()
    assert embeddings.shape[1] <= model.config.max_position_embeddings

    return (embeddings, batch_dict)


# Example of element within HotPotQA:
# {'answer': 'yes',
#  'context': {'sentences': [['Ed Wood is a 1994 American biographical period '
#                             'comedy-drama film directed and produced by Tim '
#                             'Burton, and starring Johnny Depp as cult '
#                             'filmmaker Ed Wood.',
#                             " The film concerns the period in Wood's life when "
#                             'he made his best-known films as well as his '
#                             'relationship with actor Bela Lugosi, played by '
#                             'Martin Landau.',
#                             ' Sarah Jessica Parker, Patricia Arquette, Jeffrey '
#                             'Jones, Lisa Marie, and Bill Murray are among the '
#                             'supporting cast.'],
#                            ['Scott Derrickson (born July 16, 1966) is an '
#                             'American director, screenwriter and producer.',
#                             ' He lives in Los Angeles, California.',
#                             ' He is best known for directing horror films such '
#                             'as "Sinister", "The Exorcism of Emily Rose", and '
#                             '"Deliver Us From Evil", as well as the 2016 '
#                             'Marvel Cinematic Universe installment, "Doctor '
#                             'Strange."'],
#                            ['Woodson is a census-designated place (CDP) in '
#                             'Pulaski County, Arkansas, in the United States.',
#                             ' Its population was 403 at the 2010 census.',
#                             ' It is part of the Little Rock–North Little '
#                             'Rock–Conway Metropolitan Statistical Area.',
#                             ' Woodson and its accompanying Woodson Lake and '
#                             'Wood Hollow are the namesake for Ed Wood Sr., a '
#                             'prominent plantation owner, trader, and '
#                             'businessman at the turn of the 20th century.',
#                             ' Woodson is adjacent to the Wood Plantation, the '
#                             'largest of the plantations own by Ed Wood Sr.'],
#                            ['Tyler Bates (born June 5, 1965) is an American '
#                             'musician, music producer, and composer for films, '
#                             'television, and video games.',
#                             ' Much of his work is in the action and horror '
#                             'film genres, with films like "Dawn of the Dead, '
#                             '300, Sucker Punch," and "John Wick."',
#                             ' He has collaborated with directors like Zack '
#                             'Snyder, Rob Zombie, Neil Marshall, William '
#                             'Friedkin, Scott Derrickson, and James Gunn.',
#                             ' With Gunn, he has scored every one of the '
#                             'director\'s films; including "Guardians of the '
#                             'Galaxy", which became one of the highest grossing '
#                             'domestic movies of 2014, and its 2017 sequel.',
#                             ' In addition, he is also the lead guitarist of '
#                             'the American rock band Marilyn Manson, and '
#                             'produced its albums "The Pale Emperor" and '
#                             '"Heaven Upside Down".'],
#                            ['Edward Davis Wood Jr. (October 10, 1924 – '
#                             'December 10, 1978) was an American filmmaker, '
#                             'actor, writer, producer, and director.'],
#                            ['Deliver Us from Evil is a 2014 American '
#                             'supernatural horror film directed by Scott '
#                             'Derrickson and produced by Jerry Bruckheimer.',
#                             ' The film is officially based on a 2001 '
#                             'non-fiction book entitled "Beware the Night" by '
#                             'Ralph Sarchie and Lisa Collier Cool, and its '
#                             'marketing campaign highlighted that it was '
#                             '"inspired by actual accounts".',
#                             ' The film stars Eric Bana, Édgar Ramírez, Sean '
#                             'Harris, Olivia Munn, and Joel McHale in the main '
#                             'roles and was released on July 2, 2014.'],
#                            ['Adam Collis is an American filmmaker and actor.',
#                             ' He attended the Duke University from 1986 to '
#                             '1990 and the University of California, Los '
#                             'Angeles from 2007 to 2010.',
#                             ' He also studied cinema at the University of '
#                             'Southern California from 1991 to 1997.',
#                             ' Collis first work was the assistant director for '
#                             'the Scott Derrickson\'s short "Love in the Ruins" '
#                             '(1995).',
#                             ' In 1998, he played "Crankshaft" in Eric '
#                             'Koyanagi\'s "Hundred Percent".'],
#                            ['Sinister is a 2012 supernatural horror film '
#                             'directed by Scott Derrickson and written by '
#                             'Derrickson and C. Robert Cargill.',
#                             ' It stars Ethan Hawke as fictional true-crime '
#                             'writer Ellison Oswalt who discovers a box of home '
#                             'movies in his attic that puts his family in '
#                             'danger.'],
#                            ['Conrad Brooks (born Conrad Biedrzycki on January '
#                             '3, 1931 in Baltimore, Maryland) is an American '
#                             'actor.',
#                             ' He moved to Hollywood, California in 1948 to '
#                             'pursue a career in acting.',
#                             ' He got his start in movies appearing in Ed Wood '
#                             'films such as "Plan 9 from Outer Space", "Glen or '
#                             'Glenda", and "Jail Bait."',
#                             ' He took a break from acting during the 1960s and '
#                             '1970s but due to the ongoing interest in the '
#                             'films of Ed Wood, he reemerged in the 1980s and '
#                             'has become a prolific actor.',
#                             ' He also has since gone on to write, produce and '
#                             'direct several films.'],
#                            ['Doctor Strange is a 2016 American superhero film '
#                             'based on the Marvel Comics character of the same '
#                             'name, produced by Marvel Studios and distributed '
#                             'by Walt Disney Studios Motion Pictures.',
#                             ' It is the fourteenth film of the Marvel '
#                             'Cinematic Universe (MCU).',
#                             ' The film was directed by Scott Derrickson, who '
#                             'wrote it with Jon Spaihts and C. Robert Cargill, '
#                             'and stars Benedict Cumberbatch as Stephen '
#                             'Strange, along with Chiwetel Ejiofor, Rachel '
#                             'McAdams, Benedict Wong, Michael Stuhlbarg, '
#                             'Benjamin Bratt, Scott Adkins, Mads Mikkelsen, and '
#                             'Tilda Swinton.',
#                             ' In "Doctor Strange", surgeon Strange learns the '
#                             'mystic arts after a career-ending car accident.']],
#              'title': ['Ed Wood (film)',
#                        'Scott Derrickson',
#                        'Woodson, Arkansas',
#                        'Tyler Bates',
#                        'Ed Wood',
#                        'Deliver Us from Evil (2014 film)',
#                        'Adam Collis',
#                        'Sinister (film)',
#                        'Conrad Brooks',
#                        'Doctor Strange (2016 film)']},
#  'id': '5a8b57f25542995d1e6f1371',
#  'level': 'hard',
#  'question': 'Were Scott Derrickson and Ed Wood of the same nationality?',
#  'supporting_facts': {'sent_id': [0, 0],
#                       'title': ['Scott Derrickson', 'Ed Wood']},
#  'type': 'comparison'}

_VAL_ROWS = 7405


def check_hotpotqa(ds: datasets.Dataset):
    """Checks the HotPotQA dataset."""
    # Gather stats on sentences.
    lengths = defaultdict(lambda: 0)
    for example in ds:
        assert "question" in example
        lengths[len(example["context"]["sentences"])] += 1

    assert sum(lengths.values()) == _VAL_ROWS  # From the paper Table 1: https://arxiv.org/pdf/1809.09600


def create_hotpotqa_dataset_for_e5(
    max_examples: int, max_length: int
) -> Tuple[Tuple[int], Tuple[str], Dict, torch.Tensor]:
    """Creates an evaluation dataset based on HotPotQA.

    Unfortunately, there is no easy way to tell if the tokenizer has clipped a string, or,
    if a string was exactly 512 tokens. So we remove all strings that are 512 tokens or more tokens.
    Note that a string has 6 tokens which always be present: [CLS], "Query", ":", "sep", ..., "sep", and "pad".
    So the maximum length of the input sequences is 512 - 6 = 506.

    Args:
        max_examples (int): The maximum number of examples to use.
        max_length (int): The maximum length of the input sequences (in total tokens).

    Returns:
        texts (sequence): The texts to embed. "query: " and "passage: " prefixes are used.
        reconstituted_texts (sequence): The texts are passed through a tokenizer and then
            decoded back to strings.
        batch_dict (dict): The id, mask, and type tensors.
        embeddings (torch.Tensor): The embeddings.
    """
    if max_examples <= 0 or max_examples > _VAL_ROWS * 10:
        raise ValueError(f"max_examples must be between 1 and {_VAL_ROWS * 10}.")

    ds: datasets.Dataset = datasets.load_dataset("hotpotqa/hotpot_qa", name="distractor", split="validation")
    check_hotpotqa(ds)

    texts = []

    for example in ds:
        texts.append(f"query: {example['question']}")
        if len(texts) >= max_examples:
            break

        for sentences in example["context"]["sentences"]:
            texts.append(f"passage: {''.join(sentences)}")
            if len(texts) >= max_examples:
                break
        else:  # No break, inner loop.
            continue  # continue outer loop.
        break  # If inner loop breaks, break outer loop too.

    else:  # No break
        assert _VAL_ROWS * 8 < len(texts) < _VAL_ROWS * 10, len(texts)

    model = AutoModel.from_pretrained("intfloat/e5-small")

    batch_dict, reconstructed_texts, embeddings = e5_utils.run_e5(model, texts)

    assert isinstance(embeddings, torch.Tensor)
    assert embeddings.dim() == 2, embeddings.dim()
    assert embeddings.shape[1] <= model.config.max_position_embeddings

    return (tuple(texts), tuple(reconstructed_texts), batch_dict, embeddings)

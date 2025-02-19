#!/usr/bin/env python3

# @yaml
# signature: literature_search <search_term>
# docstring: Search for research papers using the Semantic Scholar API for a given search_term. Returns list of (paperId, title, abstract, pdf_url)
# arguments:
#   search_term:
#       type: string
#       description: The query to use to look up research papers
#       required: true
#   num_results:
#       type: int
#       description: how many results to return
#       required: false
#       default: 20

import argparse
import json
from dataclasses import asdict, dataclass
from typing import List, Optional

import requests


@dataclass
class LiteratureSearchResult:
    paperId: str
    title: str
    abstract: str
    pdf_url: Optional[str]


def literature_search(query: str, num_papers: int = 20) -> List[LiteratureSearchResult]:
    fields = "paperId,title,abstract,openAccessPdf"
    url = f"http://api.semanticscholar.org/graph/v1/paper/search/bulk?query={query}&fields={fields}&limit={num_papers}&openAccessPdf"

    r = requests.get(url).json()
    retrieved = 0

    output_papers = []
    while retrieved < num_papers:
        if "data" in r:
            retrieved += len(r["data"])
            for paper in r["data"]:
                output_papers.append(
                    LiteratureSearchResult(
                        paperId=paper['paperId'],
                        title=paper['title'],
                        abstract=paper['abstract'],
                        pdf_url=paper['openAccessPdf'].get('url'),
                    )
                )
        if "token" not in r:
            break
        r = requests.get(f"{url}&token={r['token']}").json()
    return output_papers



# TODO; add params for fields of study
parser = argparse.ArgumentParser(
    description="Search for research papers using the Semantic Scholar API for a given search_term. Returns list of paper titles, abstracts, and a link to the pdf"
)
parser.add_argument("search_term", type=str, help="The query to use to look up research papers")
parser.add_argument("num_results", type=str, default=20, help="How many results to return", nargs='?')
args = parser.parse_args()

literature_search_results = literature_search(args.search_term, args.num_results)
for result in literature_search_results:
    print(json.dumps(asdict(result)))

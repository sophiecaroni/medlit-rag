import os
from dotenv import load_dotenv
from Bio import Entrez
from Bio.Entrez import esearch, efetch, read
from Bio.Entrez.Parser import StringElement

load_dotenv()
Entrez.email = os.environ.get("ENTREZ_EMAIL")  # Set the Entrez email parameter
_TOPICS = (
    topic.strip()
    for topic in os.getenv("TOPICS", "").split(",")
    if topic.strip()
)


def load_articles(
        topics: list[str] | tuple[str] = _TOPICS,
        fast: bool = False
) -> list[str]:
    """
    Loads articles from PubMed database and return article IDs for the configured topics.

    :param topics: Override default topics with a custom list of search terms.
    :param fast: If True, limits results to 4 IDs per topic (for fast iteration).
    :return: List of PubMed article IDs.
    """
    # Define publication age of articles to include
    pub_age_y = 10
    retmax = 4 if fast else 500  # number of abstracts per topic
    all_ids = []
    for topic in topics:
        results = Entrez.read(esearch(
            term=topic, db='pubmed', datetype='pdat',
            retmax=retmax,
            reldate=pub_age_y*365,  # reldate needs days
            # add restirction to english published articles
        ))
        all_ids.extend(results['IdList'])
    return all_ids


def _merge_abstract_sections(sections: list[StringElement]) -> str:
    """Merge structured abstract sections (e.g. Background, Methods) into a single string."""
    sections_txt = [str(section) for section in sections]
    return ' '.join(sections_txt)


def fetch_articles(abstracts_ids: list[str]) -> list[dict]:
    """
    Fetch and parse articles metadata from PubMed for the given artucle IDs.

    Skips articles that have no abstract or are missing required fields (year, title).

    :param abstracts_ids: List of PubMed IDs to fetch.
    :return: List of dicts with keys: pmid, year, title, text.
    """
    results = read(efetch(db='pubmed', id=abstracts_ids))

    # Order in a dict
    art_dict_lst = []
    for entry in results['PubmedArticle']:
        article = entry['MedlineCitation']['Article']  # article contents live under MedlineCitation and Article

        # Skip articles without abstract
        if 'Abstract' not in article.keys():
            continue

        # Abstract are presented by section (Background, methods, ...). Merge them into a single string
        abstract_text = _merge_abstract_sections(article['Abstract']['AbstractText'])

        try:
            # Retrieve other article information and dict-entry to the list
            art_dict_lst.append(dict(
                pmid=str(entry['MedlineCitation']['PMID']),
                year=str(article['ArticleDate'][0]['Year']),
                title=str(article['ArticleTitle']),
                text=abstract_text,
            ))

        except (KeyError, IndexError):
            # Skip articles missing some of the required information
            continue

    return art_dict_lst


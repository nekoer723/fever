import argparse
import prettytable
import logging
from drqa import retriever

# parser = argparse.ArgumentParser()
# parser.add_argument('--model', type=str, default=None)
# args = parser.parse_args()

ranker = retriever.get_class('tfidf')(tfidf_path=None)
db = retriever.get_class('sqlite')(db_path=None)


def process(query, k=1):
    doc_names, doc_scores = ranker.closest_docs(query, k)
    doc_contexts = []
    doc_scores2 = []
    table = prettytable.PrettyTable(
        ['Rank', 'Doc Id', 'Doc Score']
    )
    for i in range(len(doc_names)):
        doc_contexts.append(getText(doc_names[i]))
        doc_scores2.append(doc_scores[i])
        table.add_row([i + 1, doc_names[i], '%.5g' % doc_scores[i]])
    print(table)
    return {'doc_names': doc_names, 'doc_scores': doc_scores2, 'doc_contexts': doc_contexts}


def getText(doc_name):
    doc_context = db.get_doc_text(doc_name)
    return doc_context

from gensim.models import KeyedVectors
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.preprocessing.text import text_to_word_sequence

DOCUMENT_LIMIT = 20


def load_data():
    origin = pd.read_csv("./shows.csv")
    # null data to  empty string
    origin.fillna("", inplace=True)
    origin.info()

    # make one string for data analysis
    origin["concated"] = (
        origin["title"]
        + "\n"
        + origin["subtitle"]
        + "\n"
        + origin["description"]
        + "\n"
        + origin["summary"]
    )

    # text_embeding
    origin["token"] = [text_to_word_sequence(item) for item in origin["concated"]]
    return origin


def get_document_vectors(model, document_list):
    document_embedding_list = []
    for line in document_list:
        doc2vec = None
        count = 0
        # region gen avg of vectors in document
        for word in line.split():
            # sum all vectors of word in this document
            if word in model.key_to_index:
                count += 1
                doc2vec = model[word] if doc2vec is None else doc2vec + model[word]
        if doc2vec is not None:
            # divide sum of vectors to length.
            doc2vec = doc2vec / count
            document_embedding_list.append(doc2vec)
        # endregion

    return document_embedding_list


def recommendations(dataset, title, similarities, finding_size=5):
    # find index of title in dataset
    indices = pd.Series(dataset.index, index=dataset["title"]).drop_duplicates()
    idx = indices[title]

    # get simirarites ranking
    sim_scores = list(enumerate(similarities[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    recommended = []
    # find n most simirar datasets
    for item in sim_scores[1 : finding_size + 1]:
        cim = {"idx": item[0], "sim": item[1], "title": dataset["title"][item[0]]}
        recommended.append(cim)
    return recommended


if __name__ == "main":

    origin = load_data()

    model = KeyedVectors.load_word2vec_format(
        "./GoogleNews-vectors-negative300.bin", binary=True
    )
    document_embedding_list = get_document_vectors(
        model, origin["concated"][:DOCUMENT_LIMIT]
    )
    print("num of document_vectors:", len(document_embedding_list))

    cosine_similarities = cosine_similarity(
        document_embedding_list, document_embedding_list
    )
    print("cos matrix size :", cosine_similarities.shape)

    result = recommendations(
        origin,
        origin["title"][15],
        cosine_similarities,
        finding_size=5,
    )
    print("idx\tsim\t\t\ttitle")
    for item in result:
        print(f"{item['idx']}\t{item['sim']}\t{item['title']}")

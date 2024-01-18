from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):

    tfids_vector = TfidfVectorizer(vocabulary=vocab)
    embed = tfids_vector.fit_transform(sentences)
    features = tfids_vector.get_feature_names()
    return embed.toarray(), features
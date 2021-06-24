from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def calculate_cosine_similarity(str_list_1, str_list_2=None):
    """
    This method calculates the cosine similiraties and 
    return a two-dimension array, similarities between str_list_1 and str_list_2 
    If str_list_2 is None or empty, return similarities between str_list_1 and str_list_1
    Arguements:
        str_list_1 {list} -- List consisting entity or noun chunk keyowrds
    Default
        str_list_2 {list} -- `default = None` 

    """
    if not str_list_1:
        return np.array([[0.0]])
    if str_list_2 is None:
        vectors = get_vectors(str_list_1)
        return cosine_similarity(vectors[:len(str_list_1)])
    else:
        vectors = get_vectors(str_list_1 + str_list_2)
        len_x = len(str_list_1)
        return cosine_similarity(vectors[:len_x], Y=vectors[len_x:len(str_list_1 + str_list_2)])


def get_vectors(str_list):
    """
        This method is used to convert list of strings into vectors using CountVectorizer from sklearn package
        Arguements:
            str_list {list} -- List consisting entity or noun chunk keyowrds
        Returns:
            m {array} - array consisting of vector representation of a string
    """
    text = [t.replace("&", "_") for t in str_list]

    vectorizer = CountVectorizer(input=text, stop_words=None)
    try:
        vectorizer.fit(text)
    except ValueError:
        text += ["random_string_a_p_w"]
        vectorizer.fit(text)
    m = vectorizer.transform(text)
    return m.toarray()


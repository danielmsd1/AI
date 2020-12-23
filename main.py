import tensorflow as tf
from sklearn import preprocessing
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, Conv1D, GlobalMaxPool1D


# Get the dataset
def import_imdb():
    tf.keras.datasets.imdb.load_data(
        path="imdb.npz",
        num_words=None,
        skip_top=0,
        maxlen=None,
        seed=113,
        start_char=1,
        oov_char=2,
        index_from=3
    )

    tf.keras.datasets.imdb.get_word_index(path="imdb_word_index.json")


if __name__ == '__main__':
    import_imdb()

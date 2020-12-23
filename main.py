# Using naive bias to predict
# IMDB movie review sentiment classification dataset
#
import tensorflow as tf
# import label encoder
from sklearn import preprocessing


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


# creating the labelEncoder
le = preprocessing.LabelEncoder()
# convert string labels into numbers
imdb_encoded = le.fit_transform()

if __name__ == '__main__':
    import_imdb()

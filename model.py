from keras.models import Model
from keras.layers import Bidirectional, LSTM, Conv1D, Embedding, Input
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy
from keras.optimizers import Adam
from config import num_labels, embedding_dim, rnn_dim, learing_rate, cws_num


def load_model(vocab_size):
    input = Input(shape=(None, ))
    output = Embedding(vocab_size, embedding_dim)(input)
    output = Conv1D(embedding_dim, 3, padding='same', strides=1, activation='relu')(output)
    cws_crf = CRF(cws_num, sparse_target=True, name='cws_crf')
    cws_output = cws_crf(output)

    output = Bidirectional(LSTM(rnn_dim, return_sequences=True, dropout=0.5, recurrent_dropout=0.5))(output)
    ner_crf = CRF(num_labels, sparse_target=True, name='ner_crf')
    ner_output = ner_crf(output)
    model = Model(inputs=input, outputs=[cws_output, ner_output])
    model.summary()
    model.compile(optimizer=Adam(learing_rate), loss=crf_loss, metrics=[crf_viterbi_accuracy],
                  loss_weights={'cws_crf': 0.5, 'ner_crf': 0.5})
    return model


if __name__ == '__main__':
    model = load_model(21130)


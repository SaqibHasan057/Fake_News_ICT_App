from keras import backend as K
from keras import initializers
from keras.engine.topology import Layer
import pickle
from keras.preprocessing.sequence import pad_sequences


class HierarchicalAttentionNetwork(Layer):
    def __init__(self, attention_dim,**kwargs):
        self.init = initializers.get('normal')
        self.supports_masking = True
        self.attention_dim = attention_dim
        super(HierarchicalAttentionNetwork, self).__init__()

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = K.variable(self.init((input_shape[-1], self.attention_dim)))
        self.b = K.variable(self.init((self.attention_dim,)))
        self.u = K.variable(self.init((self.attention_dim, 1)))
        self.trainable_weights = [self.W, self.b, self.u]
        super(HierarchicalAttentionNetwork, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None):
        # size of x :[batch_size, sel_len, attention_dim]
        # size of u :[batch_size, attention_dim]
        # uit = tanh(xW+b)
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))

        ait = K.exp(K.squeeze(K.dot(uit, self.u), -1))

        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting
            ait *= K.cast(mask, K.floatx())
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        weighted_input = x * K.expand_dims(ait)
        output = K.sum(weighted_input, axis=1)

        return output

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]

    def get_config(self):
        config = super(HierarchicalAttentionNetwork, self).get_config()

        # Specify here all the values for the constructor's parameters
        config['attention_dim'] = self.attention_dim
        config['supports_masking'] = self.supports_masking
        return config


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def loadTokenizer(dir):
    f1 = open(dir+"Count","rb")
    f2 = open(dir+"Seq","rb")
    f3 = open(dir+"Tfidf","rb")

    countTokenizer = pickle.load(f1)
    seqTokenizer = pickle.load(f2)
    tfidfTokenizer = pickle.load(f3)

    return tfidfTokenizer,seqTokenizer


def tf_idf(x,tokenizer):
    # data vectorizer
    docarray = tokenizer.transform(x).toarray()
    return docarray


def tokenizedSequences(x,tokenizer):
    sequences = tokenizer.texts_to_sequences(x)
    data = pad_sequences(sequences,maxlen=5000,padding='pre',truncating='pre')

    return data

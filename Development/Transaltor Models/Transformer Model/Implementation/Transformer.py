import tensorflow as tf
import os
import string
import pickle
import re
import numpy as np

class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, seq_len, vocab_size, embed_dim, **kwargs):
        super(PositionalEmbedding, self).__init__(**kwargs)
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.token_embedding = tf.keras.layers.Embedding(
            input_dim = vocab_size, output_dim = embed_dim
        )
        self.position_embedding = tf.keras.layers.Embedding(
            input_dim = seq_len, output_dim = embed_dim
        )
        
    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start = 0, limit=length, delta=1 )
        embedded_tokens = self.token_embedding(inputs)
        embedded_positions = self.position_embedding(positions)
        return embedded_tokens + embedded_positions
    def compute_mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs, 0)
    
class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, embed_dim, latent_dim, num_heads, dropout,**kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.attention = tf.keras.layers.MultiHeadAttention(
        num_heads= num_heads, key_dim = embed_dim)
        self.layer_norm1=tf.keras.layers.LayerNormalization()
        self.layer_norm2=tf.keras.layers.LayerNormalization()        
        self.layer_ffn = tf.keras.Sequential(
            [tf.keras.layers.Dense(latent_dim, activation="relu"),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(embed_dim)]
        )
        self.supports_masking = True
        
        
        
    def call(self, inputs, mask=None):
        if mask is not None:
            padding_mask = tf.cast(mask[:, tf.newaxis, tf.newaxis, :], dtype="int32")
            
        attention_output = self.attention(
            query = inputs, 
            value = inputs,
            key = inputs,
            attention_mask = padding_mask
        )
        ffn_input = self.layer_norm1(inputs + attention_output)
        ffn_output = self.layer_ffn(ffn_input)
        return self.layer_norm2(ffn_input+ffn_output)
    

class TransformerDecoder(tf.keras.layers.Layer):
    def __init__(self, embed_dim, latent_dim, num_heads, dropout, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.attention1 = tf.keras.layers.MultiHeadAttention(
        num_heads= num_heads, key_dim = embed_dim)
        self.attention2 = tf.keras.layers.MultiHeadAttention(
        num_heads= num_heads, key_dim = embed_dim)
        self.layer_ffn = tf.keras.Sequential(
            [tf.keras.layers.Dense(latent_dim, activation="relu"),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(embed_dim),]
        )
        self.layer_norm1=tf.keras.layers.LayerNormalization()
        self.layer_norm2=tf.keras.layers.LayerNormalization()        
        self.layer_norm3=tf.keras.layers.LayerNormalization()
        self.supports_masking = True
        
    def call(self, inputs, encoder_outputs, mask=None):
        casual_mask = self.get_causal_attention_mask(inputs)
        if mask is not None:
            padding_mask = tf.cast(mask[:, tf.newaxis, :], dtype="int32")
            padding_mask = tf.minimum(padding_mask, casual_mask)
        attention_output1 = self.attention1(
            query=inputs, value=inputs, key=inputs, attention_mask = casual_mask
        )
        out1 = self.layer_norm1(inputs + attention_output1)
        attention_output2 = self.attention2(
            query = out1, value=encoder_outputs, key = encoder_outputs, attention_mask = padding_mask
        )
        out2 = self.layer_norm2(out1+attention_output2)
        ffn_output = self.layer_ffn(out2)
        return self.layer_norm3(out2 + ffn_output)
    
    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, seq_len = input_shape[0], input_shape[1]
        i=tf.range(seq_len)[:,tf.newaxis]
        j=tf.range(seq_len)
        mask = tf.cast(i>=j, dtype="int32")
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1), tf.constant([1,1], dtype=tf.int32)],
            axis=0,
        )
        return tf.tile(mask, mult)
    
# Hyper parameters
embed_dim = 128
num_heads = 10
latent_dim = 2048
vocab_size = 30000
seq_len = 20
dropout = 0.2

def createModel():
    encoder_inputs = tf.keras.Input(shape=(None,), dtype="int64", name="encoder_inputs")
    x = PositionalEmbedding(seq_len, vocab_size, embed_dim)(encoder_inputs)
    encoder_outputs = TransformerEncoder(embed_dim, latent_dim, num_heads, dropout,name="encoder_1")(x)
    encoder = tf.keras.Model(encoder_inputs, encoder_outputs)

    decoder_inputs = tf.keras.Input(shape=(None,), dtype="int64", name="decoder_inputs")
    encoded_seq_inputs = tf.keras.Input(shape=(None, embed_dim), name="decoder_state_inputs")
    x = PositionalEmbedding(seq_len, vocab_size, embed_dim)(decoder_inputs)
    x = TransformerDecoder(embed_dim, latent_dim, num_heads, dropout,name="decoder_1")(x, encoded_seq_inputs)
    x = tf.keras.layers.Dropout(0.4)(x)
    decoder_outputs = tf.keras.layers.Dense(vocab_size, activation="softmax")(x)
    decoder = tf.keras.Model([decoder_inputs, encoded_seq_inputs], decoder_outputs)

    decoder_outputs = decoder([decoder_inputs, encoder_outputs])
    transformer = tf.keras.Model(
        [encoder_inputs, decoder_inputs], decoder_outputs, name="transformer"
    )
    return transformer

def load_pretrained():
    transformer = createModel()
    transformer.load_weights(os.path.join(os.path.dirname(os.path.realpath(__file__)),"en_hi","eng to hindi.h5"))
    return transformer
 
def check(line):
    print(line)

check(os.path.join(os.path.dirname(os.path.realpath(__file__)),"en_hi","hin_vector.pkl"))

strip_chars = string.punctuation + "¿"
strip_chars = strip_chars.replace("[", "")
strip_chars = strip_chars.replace("]", "")

check("string ok")
def custom_standardization(input_string):
    lowercase = tf.strings.lower(input_string)
    return tf.strings.regex_replace(lowercase, "[%s]" % re.escape(strip_chars), "")

def textVectorization():
    # Tokenizing Sentence
    # strip_chars = string.punctuation + "¿"
    # strip_chars = strip_chars.replace("[", "")
    # strip_chars = strip_chars.replace("]", "")

    # check("string ok")
    # def custom_standardization(input_string):
    #     lowercase = tf.strings.lower(input_string)
    #     return tf.strings.regex_replace(lowercase, "[%s]" % re.escape(strip_chars), "")
    # check("custom_standardization ok")
    # eng_vector = tf.keras.layers.TextVectorization(
    #     max_tokens= 30000, output_mode = "int", output_sequence_length = 20 
    # )

    # hin_vector = tf.keras.layers.TextVectorization(
    #     max_tokens= 30000, output_mode = "int", output_sequence_length = 20+1, standardize=custom_standardization 
    # )
    #reading pickle file
    # return "hello"
    # custom_standardization()
    # with open(os.path.join(os.path.dirname(os.path.realpath(__file__)),"en_hi","eng_vector.pkl"), "rb") as eng:
    #     eng = pickle.load(eng)
    # check("opening file ok")
    # with open(os.path.join(os.path.dirname(os.path.realpath(__file__)),"en_hi","hin_vector.pkl"), "rb") as hin:
    #     hin = pickle.load(hin)
    file1 = open(os.path.join(os.path.dirname(os.path.realpath(__file__)),"en_hi","eng_vector.pkl"), "rb")
    eng= pickle.load(file1)
    file2 = open(os.path.join(os.path.dirname(os.path.realpath(__file__)),"en_hi","hin_vector.pkl"), "rb")
    hin = pickle.load(file2)
    check("open file 1 ok")
    new_hin = tf.keras.layers.TextVectorization.from_config(hin["config"])
    new_hin.adapt(tf.data.Dataset.from_tensor_slices(["xyz"]))
    new_hin.set_weights(hin['weights'])
    hin_vector = new_hin
    new_eng = tf.keras.layers.TextVectorization.from_config(eng["config"])
    new_eng.adapt(tf.data.Dataset.from_tensor_slices(["xyz"]))
    new_eng.set_weights(eng['weights'])
    eng_vector = new_eng
    return [eng_vector, hin_vector]
    
def decode_sequence(input_sentence, hin_vector, eng_vector, transformer):
    hindi_vocab = hin_vector.get_vocabulary()
    check("hindi_vocab")
    hindi_index_lookup = dict(zip(range(len(hindi_vocab)), hindi_vocab))
    check("lookup ok")
    max_decoded_sentence_length = 20
    
    tokenized_input_sentence = eng_vector([input_sentence])
    decoded_sentence = "[start]"
    check("decode sentence ok")
    for i in range(max_decoded_sentence_length):
        tokenized_target_sentence = hin_vector([decoded_sentence])[:, :-1]
        predictions = transformer([tokenized_input_sentence, tokenized_target_sentence])

        sampled_token_index = np.argmax(predictions[0, i, :])
        sampled_token = hindi_index_lookup[sampled_token_index]
        decoded_sentence += " " + sampled_token

        if sampled_token == "[end]":
            break
    check("decoded compeleted ok")
    return decoded_sentence[8:-5]

# with open(os.path.join(os.path.dirname(os.path.realpath(__file__)),"en_hi","hin_vector.pkl"), "rb") as hin:
#     hin = pickle.load(hin)
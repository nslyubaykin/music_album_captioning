import tensorflow as tf
import numpy as np

# Masking helper functions:

def create_padding_mask_equal(seq, value=0):
    seq = tf.cast(tf.math.equal(seq, value), tf.float32)
    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def create_padding_mask_less(seq, value=0):
    seq = tf.cast(tf.math.less(seq, value), tf.float32)
    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def create_padding_mask_greater(seq, value=0):
    seq = tf.cast(tf.math.greater(seq, value), tf.float32)
    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def create_look_ahead_mask(size):
    mask = tf.math.subtract(1.0, tf.linalg.band_part(tf.ones((size, size), name=f'lam{np.random.rand(1)[0]}'), -1, 0))
    return mask  # (seq_len, seq_len)

def create_combined_mask(seq, tf_padding_fun, tf_look_ahead_fun):
    padding_mask = tf_padding_fun(seq)
    look_ahead_mask = tf_look_ahead_fun(seq)
    return tf.maximum(padding_mask, look_ahead_mask)

# Position encoding helper functions:

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, model_dim, num_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        
        if model_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        
        self.projection_dim = model_dim // num_heads
        self.wq = tf.keras.layers.Dense(model_dim)
        self.wk = tf.keras.layers.Dense(model_dim)
        self.wv = tf.keras.layers.Dense(model_dim)
        self.join_heads = tf.keras.layers.Dense(model_dim)
    
    def attention(self, q, k, v, mask):
        # matrix multiplication
        qk = tf.matmul(q, k, transpose_b=True)
        # scale matmul_qk
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = qk / tf.math.sqrt(dk)
        # add the mask to the scaled tensor.
        if mask is not None:
            scaled_attention_logits += (mask * -1e9) 
        # softmax is normalized on the last axis (seq_len_k) so that the scores
        # add up to 1.
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)
        return output, attention_weights
    
    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]
        
        q = self.wq(q) # (batch_size, seq_len, d_model)
        k = self.wk(k) # (batch_size, seq_len, d_model)
        v = self.wv(v) # (batch_size, seq_len, d_model)
        
        q = self.separate_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.separate_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.separate_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
        
        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = self.attention(q, k, v, mask)
        
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
        
        concat_attention = tf.reshape(scaled_attention, 
                                      (batch_size, -1, self.model_dim))  # (batch_size, seq_len_q, d_model) 
        output = self.join_heads(concat_attention)
        return output, attention_weights
    
    
class TransformerEncoderBlock(tf.keras.layers.Layer):
    def __init__(self, model_dim, num_heads, ffn_dim, dropout_rate=0.1):
        super(TransformerEncoderBlock, self).__init__()
        self.mha = MultiHeadAttention(model_dim, num_heads)
        self.ffn = tf.keras.Sequential([
          tf.keras.layers.Dense(ffn_dim, activation='relu'),  # (batch_size, seq_len, dff)
          tf.keras.layers.Dense(model_dim),  # (batch_size, seq_len, d_model)
        ])
        
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        
    def call(self, x, training, mask):
        
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(attn_output + x)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(ffn_output + out1)
        
        return out2
    

class TokenAndPositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, seq_maxlen, vocab_size, embed_dim, pos_embed_type='trainable', scale_emb=False):
        super(TokenAndPositionEmbedding, self).__init__()
        
        self.embed_dim = embed_dim
        self.scale_emb = scale_emb
        if pos_embed_type not in ('trainable', 'trigonometric'):
            raise ValueError(
                f'pos_embed_type should be either "trainable" or "trigonometric", you have entered: {pos_embed_type}'
            )
        
        self.pembt = pos_embed_type
        self.token_emb = tf.keras.layers.Embedding(vocab_size, embed_dim)
        
        if self.pembt == 'trainable':
            self.pos_emb = tf.keras.layers.Embedding(seq_maxlen, embed_dim)
        else:
            self.pos_emb = positional_encoding(seq_maxlen, embed_dim)
            
    def call(self, x):
        maxlen = tf.shape(x)[1]
        token_embedding = self.token_emb(x)
        
        if self.scale_emb:
            token_embedding *= tf.math.sqrt(tf.cast(self.embed_dim, tf.float32))
        
        if self.pembt == 'trainable':
            positions = tf.range(start=0, limit=maxlen, delta=1)
            positions = self.pos_emb(positions)
        else:
            positions = self.pos_emb[:, :maxlen, :]
        
        return token_embedding + positions
    
    
class PositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, seq_maxlen, embed_dim, pos_embed_type='trainable'):
        super(PositionEmbedding, self).__init__()
        
        if pos_embed_type not in ('trainable', 'trigonometric'):
            raise ValueError(
                f'pos_embed_type should be either "trainable" or "trigonometric", you have entered: {pos_embed_type}'
            )
        
        self.pembt = pos_embed_type
        
        if self.pembt == 'trainable':
            self.pos_emb = tf.keras.layers.Embedding(seq_maxlen, embed_dim)
        else:
            self.pos_emb = positional_encoding(seq_maxlen, embed_dim)
            
    def call(self, x):
        maxlen = tf.shape(x)[1]
        
        if self.pembt == 'trainable':
            positions = tf.range(start=0, limit=maxlen, delta=1)
            positions = self.pos_emb(positions)
        else:
            positions = self.pos_emb[:, :maxlen, :]
        
        return x + positions
    
    
class TransformerDecoderBlock(tf.keras.layers.Layer):
    def __init__(self, model_dim, num_heads, ffn_dim, dropout_rate=0.1):
        super(TransformerDecoderBlock, self).__init__()
        self.mha1 = MultiHeadAttention(model_dim, num_heads)
        self.mha2 = MultiHeadAttention(model_dim, num_heads)
        self.ffn = tf.keras.Sequential([
          tf.keras.layers.Dense(ffn_dim, activation='relu'),  # (batch_size, seq_len, dff)
          tf.keras.layers.Dense(model_dim),  # (batch_size, seq_len, d_model)
        ])
        
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout3 = tf.keras.layers.Dropout(dropout_rate)
        
    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        
        attn_output1, attn_weights1 = self.mha1(x, x, x, look_ahead_mask)
        attn_output1 = self.dropout1(attn_output1, training=training)
        out1 = self.layernorm1(attn_output1 + x)
        
        attn_output2, attn_weights2 = self.mha2(enc_output, enc_output, out1, padding_mask)
        attn_output2 = self.dropout2(attn_output2, training=training)
        out2 = self.layernorm2(attn_output2 + out1)
        
        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)
        
        return out3, attn_weights1, attn_weights2
    

class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, model_dim, num_heads, ffn_dim, dropout_rate=0.1):
        super(TransformerEncoder, self).__init__()
        self.num_layers = num_layers
        # note: no embedding inside Encoder, embedding is done outside the model
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.enc_layers = [TransformerEncoderBlock(model_dim, num_heads, ffn_dim, dropout_rate) for _ in range(num_layers)]
    
    def call(self, x, training, mask):

        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model) 

    
class TransformerDecoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, model_dim, num_heads, ffn_dim, dropout_rate=0.1):
        super(TransformerDecoder, self).__init__()
        self.num_layers = num_layers
        # note: no embedding inside Decoder, embedding is done outside the model
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.dec_layers = [TransformerDecoderBlock(model_dim, num_heads, ffn_dim, dropout_rate) for _ in range(num_layers)]
        
    def call(self, x, enc_output, training, 
             look_ahead_mask, padding_mask):
        attention_weights = {}
        x = self.dropout(x, training=training)
        
        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                                   look_ahead_mask, padding_mask)

            attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i+1)] = block2
        
        return x, attention_weights
    

class Transformer(tf.keras.Model):
    def __init__(self, num_layers, enc_model_dim, enc_num_heads, enc_ffn_dim, 
                 dec_model_dim, dec_num_heads, dec_ffn_dim, dropout_rate=0.1):
        super(Transformer, self).__init__()
        # note 1: different dimentions for encoder and decoder
        # note 2: no final affine layer for more customization
        self.encoder = TransformerEncoder(num_layers, enc_model_dim, enc_num_heads, 
                                          enc_ffn_dim, dropout_rate)
        self.decoder = TransformerDecoder(num_layers, dec_model_dim, dec_num_heads, 
                                          dec_ffn_dim, dropout_rate)
        
    def call(self, enc_inp, dec_inp, training, enc_padding_mask, 
             look_ahead_mask, dec_padding_mask): 
        
        enc_output = self.encoder(enc_inp, training, enc_padding_mask)
        dec_output, attention_weights = self.decoder(dec_inp, enc_output, training, 
                                                     look_ahead_mask, dec_padding_mask)
        
        return dec_output, attention_weights

from keras.models import Model
from keras.layers import Input, LSTM, GRU, Dense, Embedding, Bidirectional, BatchNormalization
from keras import optimizers
import numpy as np
import os
import cPickle as pickle

with open('github_issues-bundle.pkl', 'rb') as fp:
    X, Y, decoder_target_data, idx2word, word2idx = pickle.load(fp)

#arbitrarly set latent dimension for embedding and hidden units
latent_dim = 300
num_encoder_tokens = num_decoder_tokens = len(X)
batch_size = 1200
epochs = 2 #change to 7 in prod
##### Define Model Architecture ######

########################
#### Encoder Model ####
encoder_inputs = Input(shape=(27,), name='Encoder-Input')

# Word embeding for encoder (ex: Issue Body)
x = Embedding(num_encoder_tokens, latent_dim, name='Body-Word-Embedding', mask_zero=False)(encoder_inputs)
x = BatchNormalization(name='Encoder-Batchnorm-1')(x)

# We do not need the `encoder_output` just the hidden state.
_, state_h = GRU(latent_dim, return_state=True, name='Encoder-Last-GRU')(x)

# Encapsulate the encoder as a separate entity so we can just
#  encode without decoding if we want to.
encoder_model = Model(inputs=encoder_inputs,
                      outputs=state_h,
                      name='Encoder-Model')

seq2seq_encoder_out = encoder_model(encoder_inputs)

########################
#### Decoder Model ####
decoder_inputs = Input(shape=(None,), name='Decoder-Input')  # for teacher forcing

# Word Embedding For Decoder (ex: Issue Titles)
dec_emb = Embedding(num_decoder_tokens, latent_dim, name='Decoder-Word-Embedding',mask_zero=False)(decoder_inputs)
dec_bn = BatchNormalization(name='Decoder-Batchnorm-1')(dec_emb)

# Set up the decoder, using `decoder_state_input` as initial state.
decoder_gru = GRU(latent_dim,
                  return_state=True,
                  return_sequences=True,
                  name='Decoder-GRU')

decoder_gru_output, _ = decoder_gru(dec_bn, initial_state=seq2seq_encoder_out)
x = BatchNormalization(name='Decoder-Batchnorm-2')(decoder_gru_output)

# Dense layer for prediction
decoder_dense = Dense(num_decoder_tokens, activation='softmax', name='Final-Output-Dense')

decoder_outputs = decoder_dense(x)

########################
#### Seq2Seq Model ####

#seq2seq_decoder_out = decoder_model([decoder_inputs, seq2seq_encoder_out])
seq2seq_Model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
seq2seq_Model.compile(optimizer=optimizers.Nadam(lr=0.001), loss='sparse_categorical_crossentropy')

seq2seq_Model.summary()


########################
#### Train the Model ####

from keras.callbacks import CSVLogger, ModelCheckpoint

#setup callbacks for model logging
script_name_base = 'keras_seq2seq'
csv_logger = CSVLogger('{:}.log'.format(script_name_base))
model_checkpoint = ModelCheckpoint('{:}.epoch{{epoch:02d}}-val{{val_loss:.5f}}.hdf5'.format(script_name_base),
                                   save_best_only=True)

# pass arguments to model.fit
history = seq2seq_Model.fit([X, Y], np.expand_dims(decoder_target_data, -1),
                            batch_size=batch_size,
                            epochs=epochs,
                            validation_split=0.12, callbacks=[csv_logger, model_checkpoint])

seq2seq_Model.save('seq2seq_model_keras.h5')
print('EOP')

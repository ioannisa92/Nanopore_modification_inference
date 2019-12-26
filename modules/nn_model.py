from keras.models import Model
from keras.layers import Input, Dense, Flatten, Dropout, Activation, Conv1D, MaxPooling1D, AveragePooling1D
from keras.optimizers import Adam
import keras.models
from keras_deep_graph_learning.keras_dgl.layers import MultiGraphCNN
from keras_deep_graph_learning.examples import utils
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import json
#imports


def initialize_filters(A):
    '''
    initaite filters
    '''
    filters = utils.preprocess_adj_tensor_with_identity(A, True)
    return filters


def initialize_model(X, filters,
                     n_gcn=2,
                     n_cnn=2, kernal_size_cnn=3,
                     n_dense=2,
                     dropout=0.2):
    '''
    initaite model
    '''  
    X_shape = Input(shape=(X.shape[1], X.shape[2]))
    filters_shape = Input(shape=(filters.shape[1], filters.shape[2]))
    #input dimensions

    for n in range(n_gcn, 0, -1):
        if n == n_gcn:
            output = MultiGraphCNN(16 * (2**(n - 1)), 2, activation='elu')([X_shape, filters_shape])
        else:
            output = MultiGraphCNN(16 * (2**(n - 1)), 2, activation='elu')([output, filters_shape])
        output = Dropout(dropout)(output)
    #GCN
    
    for n in range(n_cnn, 0, -1):
        output = Conv1D(16 * (2**(n)), kernal_size_cnn)(output)
    output = AveragePooling1D()(output)
    #CNN
    
    output = Flatten()(output)
    output = Dense(16 * (2**(n_dense - 1)), kernel_initializer="glorot_uniform")(output)
    output = Activation('elu')(output)
    output = Dense(1, kernel_initializer="glorot_uniform")(output)
    output = Activation('linear')(output)
    #flatten and dense
    
    model = Model(inputs=[X_shape, filters_shape], outputs=output)
    return model


def tune_model(A, X, Y,
               list_n_gcn,
               list_n_cnn, list_kernal_size_cnn,
               list_n_dense,
               list_dropout,
               batch_size=128, validation_split=0.4, epochs=100):
    '''
    tune model
    '''  
    history = defaultdict(list)
    for n_gcn in list_n_gcn:
        for n_cnn in list_n_cnn:
            for kernal_size_cnn in list_kernal_size_cnn:
                for n_dense in list_n_dense:
                    for dropout in list_dropout:
                        print("processing:"+str(n_gcn)+'_'+str(n_cnn)+'_'+str(kernal_size_cnn)+'_'+str(n_dense)+'_'+str(dropout))
                        flt = initialize_filters(A)
                        mdl = initialize_model(X, flt, n_gcn , n_cnn, kernal_size_cnn, n_dense, dropout)
                        mdl.compile(loss='mean_squared_error', optimizer=Adam())
                        hs = mdl.fit([X, flt], Y, batch_size=batch_size, validation_split=validation_split, epochs=epochs, verbose=0)
                        history[str(n_gcn)+'_'+str(n_cnn)+'_'+str(kernal_size_cnn)+'_'+str(n_dense)+'_'+str(dropout)].append(hs.history)
    return history


def train_model(X, Y, model, filters, batch_size=128, validation_split=0.4, epochs=100):
    '''
    train model
    ''' 
    model.compile(loss='mean_squared_error', optimizer=Adam())
    model.fit([X, filters], Y, batch_size=batch_size, validation_split=validation_split, epochs=epochs, verbose=0)
    return model


def fit_model(X, model, filters):
    '''
    fit model
    ''' 
    predict = model.predict([X, filters], verbose=1)
    return predict


def plot_epoch_loss(history, keys, loss='loss'):
    '''
    plot model tunning process
    '''  
    fig = plt.gcf()
    fig.set_size_inches(5, 5)
    for key in keys:
        hs = history.get(key)[0]
        plt.plot(np.log10(hs.get(loss)), label=key)
    plt.legend(loc='best', fontsize=10)
    plt.title(loss+' VS epochs', fontsize=25)
    plt.xlabel('epochs', fontsize=25)
    plt.ylabel('log10(loss)', fontsize=25)
    plt.show()


def save_tuning_history(name, history):
    '''
    save tunning history
    '''
    json_file = json.dumps(history)
    f = open(name+'_history.json', 'w')
    f.write(json_file)
    f.close()


def load_tuning_history(name):
    '''
    load tunning history
    '''
    json_file = open(name+'_history.json')
    json_string = json_file.read()
    json_data = json.loads(json_string)
    return json_data


def save_model(name, model):
    '''
    save model
    '''
    model_json = model.to_json()
    with open(name+'_model.json', 'w') as json_file:
        json_file.write(model_json)
    model.save_weights(name+'_model.h5')


def load_model(name):
    '''
    load model
    '''
    with open(name+'_model.json', 'r') as json_file:
        architecture = json_file.read()
    model = keras.models.model_from_json(architecture, custom_objects={'MultiGraphCNN':MultiGraphCNN})
    model.load_weights(name+'_model.h5')
    return model
#functions


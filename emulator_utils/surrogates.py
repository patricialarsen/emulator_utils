from tensorflow.keras import Sequential
from keras.models import Model
from tensorflow.keras.layers import Dense, Conv1D, Activation, Dropout, Flatten, Input
import tensorflow as tf
from tensorflow.keras import backend as K
tf.random.set_seed(3)


__all__ = ("simple_mlp", "train_mlp", "save_mlp", "load_mlp", "train_pca", "save_pca", "load_pca", )

def simple_mlp(input_shape, output_shape, hidden_dims):
    '''
    Parameters
    input_shape: integer 
    output_shape: integer
    hidden dim: numpy array with integers

    TO-DO: add options for changing loss, metrics, and optimizer

    '''
    p_dropout = 0.2

    model = Sequential()

    model.add(Dense(hidden_dims[0], activation='relu', kernel_initializer='he_normal', input_shape=(input_shape,)))

    for hidden_shape in hidden_dims[1:]:
        model.add(Dense(hidden_shape, activation='relu', kernel_initializer='he_normal'))
        model.add(Dropout(p_dropout))


    model.add(Dropout(p_dropout))

    model.add(Dense(output_shape, activation='linear'))
    print(model.summary())

    return model

def train_mlp(model, train_data, train_target, validation_data, validation_target, learning_rate, decay_rate, num_epochs, batch_size):

    '''
    Training model

    Parameters
    model: tensorflow model
    train_data, train_target
    validation_data, validation_target
    learning_rate, decay_rate, num_epochs, batch_size: hyper-parameters
    fileout: full path to save the trained model

    '''

    # compile the model
    model.compile(optimizer='adam', loss='mse', metrics=['binary_crossentropy'])

    K.set_value(model.optimizer.lr, learning_rate)
    K.set_value(model.optimizer.decay, decay_rate)


    train_history = model.fit(train_data, train_target, epochs=num_epochs, batch_size=batch_size, verbose=0, validation_data=(validation_data, validation_target))
    print('Training complete')
    # evaluate the model
    loss, acc = model.evaluate(validation_data, validation_target, verbose=0)
    print('Test loss after training: %.3f' % loss)
    # save the model

    return model, train_history

def save_mlp(model, fileout):

    # save the model
    tf.keras.models.save_model(model, fileout, overwrite=True, include_optimizer=True, save_format=None, signatures=None, options=None)

    print('Model saved at'+fileout)

def load_mlp(fileout):

    # load a trained model
    model = tf.keras.models.load_model(fileout)
    return model


#####################################################


def simple_gp():
    return NotImplemented




def train_pca(data, num_components):
    '''
    run pca compression
    Parameters
    ----------
    x: ndarray(float)
       input array to compress
    PCAmodel: str
       path to output PCA model
    nComp: int
       number of PCA components
    Returns
    -------
    pca_model: addtype
    principalComponents: addtype
    pca_bases: addtype
    '''
    #TODO: add types and explanations to doc string
    # x is in shape (nparams, nbins)

    pca_model = PCA(n_components=num_components)
    principalComponents = pca_model.fit_transform(data)
    pca_bases = pca_model.components_

    print("original shape:   ", x.shape)
    print("transformed shape:", principalComponents.shape)
    print("bases shape:", pca_bases.shape)

    return pca_model, np.array(principalComponents), np.array(pca_bases)


def save_pca(model, fileout):

    pickle.dump(pca_model, open(fileout, 'wb'))

    print('Model saved at'+fileout)

def load_pca(fileout):
    model = pickle.load(open(fileout, 'rb'))
    return model



from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils.visualize_util import plot

model = Sequential()
model.add(Dense(output_dim=64, input_dim=100))
model.add(Activation('relu'))
model.add(Dense(output_dim=10))
model.add(Activation('softmax'))

if __name__ == '__main__':
    plot(model, to_file='model.png',
         show_shapes=True, show_layer_names=True)

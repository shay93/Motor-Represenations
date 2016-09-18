# Motor-Represenations
We aim to learn transferrable visuomotor representations by learning a transform between some combination of control spaces state spaces and action spaces. The following is a list of experiments that have been run along with a brief description of the results obtained:

The action used to study the above is the creation of geometric shapes, by n-degree of freedom arms. The geometric shapes consist of various Triangles,Squares and Rectangles. 

1. The first experiment is just a toy autoencoder that aims to learn the identity for random data. This experiment was primarily to learn tensorflow functionality. The file used is the autoencoder_toy.py.  

2. The first baseline experiment aims to recreate the shapes via a convolutional autoencoder, the architecture of the above may be found in the file shape_autoencoder.py. We are however, unable to regenerate shapes using the above.

3. The next experiment is to figure out RNN functionality in Tensorflow. This is done by implementing an LSTM that aims to fit a sinuisoid. This is achieved for a sinuisoid with fixed frequency and amplitude, but with uniform noise added to the sinuisoid. This architecture may be found in lstm_toy2.py

The experiments to still be performed or tested are listed below:

1. A sequence to sequence mapping is implemented via a LSTM and the architecture is shown in seq2seq.py but a good test needs to be performed in order to iron out any bugs. Currently the seq2seq is carried out between the two control spaces. 

2. Alternatively one may learn the control inputs of one arm given the control inputs of the other. 

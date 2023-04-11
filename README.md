## Deep Learning Decoder for Linear Block Code
This repository contains an implementation of a deep-learning based soft-decision decoder for a linear block code.

The decoder is based on the work by Bennatan et al. [1], and this code replicates their method using Python and TensorFlow. To speed up training, the block size has been reduced compared to the original work.

## Replication of Bennatan et al.'s Work
One of the objectives of this repo is to replicate the results of Bennatan et al. [1]. To achieve this, the same neural network architecture was used, but with a simplified training method. 

## Training on Application-Specific Noise Patterns
In addition to the reproduction of Bennatan et al.'s [1] work, this repo aims to demonstrate that training a DNN on application-specific noise patterns yields superior error correction performance.

To simulate a less-than-random noise pattern, the Gaussian noise is passed through a Butterworth lowpass filter with order 3 and cutoff frequency set to half that of the Nyquist frequency. In effect, this increases the likelihood that error bits appear in pairs, which could simulate cross-talk interference from a device on the same or similar carrier frequency but with a lower transmission rate, for example. To compensate for the attenuation of the lowpass filter, the noise was then amplified to attain the original signal-to-noise ratio (SNR).

The same neural network architecture as before was then trained on this filtered noise using the same method.

## Requirements and Installation
The code is implemented in Python and is primarily based on TensorFlow and Galois. The required packages are listed in the requirements.txt file. To install the dependencies, run the following command:

```
pip install -r requirements.txt
```

## Usage
To train the model, run the model_train.py file:
```
python model_train.py
```
The code generates the error patterns, pairity check matrix, and trains the DNN. Once completed, the trained model is saved.

To test the performance of the trained model and produce graphs, run model_analysis.py:
```
python model_analysis.py
```

## References
[1]	A. Bennatan, Y. Choukroun and P. Kisilev, "Deep Learning for Decoding of Linear Codes - A Syndrome-Based Approach," 2018 IEEE International Symposium on Information Theory (ISIT), Vail, CO, USA, 2018, pp. 1595-1599, doi: 10.1109/ISIT.2018.8437530.J. Clerk Maxwell, A Treatise on Electricity and Magnetism, 3rd ed., vol. 2. Oxford: Clarendon, 1892, pp.68–73. 
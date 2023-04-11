import galois as ga
from galois import Poly
import numpy as np

import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

USE_GPU = False

if USE_GPU:
    if len(tf.config.list_physical_devices('GPU')) == 0:
        raise Exception("GPU not found")
else:
    tf.config.set_visible_devices([], 'GPU')  # Disable GPU
    # tf.debugging.set_log_device_placement(True)

assert tf.executing_eagerly()

NUM_SAMPLES = 400000
MAX_EPOCHS = 20  # Maximum number of epochs to train on before giving up
BATCH_SIZE = 512  # How many series are trained together
VALIDATE_BATCHES = max(1000 // BATCH_SIZE, 1)  # How many batches to reserve for validation
SNR_DB = 4
NOISE_STDEV = 10**(-SNR_DB / 20)
FILTER_CUTOFF = 0.5
MODEL_PATH = f'./cache/model_filter{FILTER_CUTOFF}_{SNR_DB}dB'

GF = ga.GF(2)
g = ga.Poly.Str('x**4 + x + 1', field=GF)

h = ga.Poly.Str('x**15 + 1', field=GF) // g

print("\nParity polynomial h(X) =")
print(f"\t{h}")

n = 15
k = 11

P = GF.Zeros((k, n-k))
for i in range(k):
    p = ga.Poly.Degrees([n-k+i], [1], field=GF)
    b = p % g

    P[i] = b.coefficients(size=n - k, order='desc')

I = GF.Identity(k)
G = np.column_stack([P, I])
print("\nGenerator matrix G =")
print(f"{G}")

H = np.column_stack([GF.Identity(n-k), P.T])
print("\nParity check matrix H =")
print(f"{H}")

U = ga.GF(2**k).Range(0, 2 ** k).vector()
V_gf2 = np.dot(U, G)

w_H = np.sum(np.array(V_gf2[1:], dtype=int), axis=1)
d_min = np.min(w_H)
print(f"d_min = {d_min}")
print(f"Detectable errors = {d_min - 1}")

V_b = np.array(V_gf2, dtype=np.float32)  # Float since we are preparing for network training, saves casting later
V_s = V_b*2-1

def butter_lowpass(x, cutoff_nyquist , order=1):
    b, a = butter(order, cutoff_nyquist, btype='low', analog=False)
    y = filtfilt(b, a, x, axis=-1)
    return y

def gen_error(num, noise_stdev, filter_cutoff):
    Z = tf.random.normal((num, n), mean=0, stddev=noise_stdev)

    if filter_cutoff:
        Z = butter_lowpass(Z, filter_cutoff, order=3)
        filter_stdev = np.std(Z)
        Z *= noise_stdev/filter_stdev

    Z += 1
    return Z


def main():

    Z = gen_error(NUM_SAMPLES, NOISE_STDEV, FILTER_CUTOFF)

    Z_abs = tf.abs(Z)
    Z_sign = tf.sign(Z)
    Z_bin = (-Z_sign + 1)/2

    synd = np.array(np.dot(GF(np.array(Z_bin, dtype=int)), H.T), dtype=np.float32)
    F_out = Z_bin
    F_in = tf.concat((Z_abs, synd), axis=1)

    ds = tf.data.Dataset.from_tensor_slices((F_in, F_out))
    ds = ds.batch(BATCH_SIZE)

    train_ds = ds.skip(VALIDATE_BATCHES)
    val_ds = ds.take(VALIDATE_BATCHES)

    model = tf.keras.Sequential([
            tf.keras.layers.RepeatVector(5),
            tf.keras.layers.GRU(5*n, return_sequences=True),
            tf.keras.layers.GRU(5*n, return_sequences=True),
            tf.keras.layers.GRU(5*n, return_sequences=True),
            tf.keras.layers.GRU(n, return_sequences=False, activation='sigmoid'),
        ])

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), #from_logits=True
                  optimizer=tf.keras.optimizers.Adam(20E-4),
                  metrics=['accuracy', 'mean_absolute_error'])

    model.build(train_ds.element_spec[0].shape)

    model.summary()

    history = model.fit(train_ds, epochs=MAX_EPOCHS,
                        validation_data=val_ds,
                        validation_steps=30,
                        callbacks=[])

    test_loss, test_acc, mean_absolute_error = model.evaluate(val_ds)
    print('Test Loss:', test_loss)

    model.save(MODEL_PATH)

    x_real, y_real = zip(*[x for x in val_ds.as_numpy_iterator()])
    x_real = np.concatenate(x_real)
    y_real = np.concatenate(y_real)

    plt.plot(history.history['loss'], label='train')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    y_pred = model.predict(x_real)

    for i in range(100):
        print(f"Real: {y_real[i].astype(int)} -> Predicted: {np.round(y_pred[i]).astype(int)}")

    print("Done.")

if __name__ == '__main__':
    main()
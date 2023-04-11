import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from model_train import MODEL_PATH, n, k, GF, H, gen_error

tf.config.set_visible_devices([], 'GPU')  # Disable GPU

NUM_ERRS = 10000

def calc_ber(model, noise_stdev, filter_cutoff):

    Z = gen_error(NUM_ERRS, noise_stdev, filter_cutoff)

    Z_abs = tf.abs(Z)
    Z_sign = tf.sign(Z)
    Z_bin = (-Z_sign + 1) / 2

    synd = np.array(np.dot(GF(np.array(Z_bin, dtype=int)), H.T), dtype=np.float32)

    F_in = tf.concat((Z_abs, synd), axis=1)

    z_pred = model.predict(F_in)
    z_pred = np.round(z_pred)

    ber = np.sum(z_pred != Z_bin) / z_pred.size

    return ber

def calc_ber_simple(noise_stdev, filter_cutoff):

    Z = gen_error(NUM_ERRS, noise_stdev, filter_cutoff)

    Z_abs = tf.abs(Z)
    Z_sign = tf.sign(Z)
    Z_bin = (-Z_sign + 1) / 2

    synd = np.array(np.dot(GF(np.array(Z_bin, dtype=int)), H.T), dtype=np.float32)

    z_pred = Z_abs.numpy().copy()
    
    for i in range(z_pred.shape[0]):
        if np.sum(synd[i]) > 0:
            bad_idx = np.argmin(z_pred[i])
            z_pred[i, bad_idx] = -1

    z_pred = np.sign(z_pred)
    z_pred = (-z_pred + 1) / 2

    ber = np.sum(z_pred != Z_bin) / z_pred.size

    return ber

def raw_ber(noise_stdev, filter_cutoff):
    Z = gen_error(NUM_ERRS, noise_stdev, filter_cutoff)

    Z_sign = tf.sign(Z)
    Z_bin = (-Z_sign + 1) / 2

    ber = np.sum(Z_bin) / Z_bin.numpy().size

    return ber

def main():
    model_mu = tf.keras.models.load_model(r'./cache/model_filterNone_4dB')
    model_mf = tf.keras.models.load_model(r'./cache/model_filter0.5_4dB')

    snr_dbs = np.arange(1, 6+1)
    noise_stdevs = 10 ** (-snr_dbs / 20)

    # mu -> Model = Unfiltered
    # mf -> Model = Filtered
    # eu -> Error = Unfiltered
    # ef -> Error = Filtered

    bers_mu_eu = []
    bers_mf_eu = []
    bers_simple_eu = []
    bers_raw_eu = []

    bers_mu_ef = []
    bers_mf_ef = []
    bers_simple_ef = []
    bers_raw_ef = []

    for noise_stdev in noise_stdevs:
        bers_mu_eu.append(calc_ber(model_mu, noise_stdev, None))
        bers_mf_eu.append(calc_ber(model_mf, noise_stdev, None))
        bers_simple_eu.append(calc_ber_simple(noise_stdev, None))
        bers_raw_eu.append(raw_ber(noise_stdev, None))

        bers_mu_ef.append(calc_ber(model_mu, noise_stdev, 0.5))
        bers_mf_ef.append(calc_ber(model_mf, noise_stdev, 0.5))
        bers_simple_ef.append(calc_ber_simple(noise_stdev, 0.5))
        bers_raw_ef.append(raw_ber(noise_stdev, 0.5))

    plt.title("BER vs. SNR")
    plt.xlabel("$E_{b}/N_{0}$ (dB)")
    plt.ylabel("BER")
    plt.plot(snr_dbs, bers_mu_eu, label="DNN Decoder")
    plt.plot(snr_dbs, bers_mf_eu, label="DNN Decoder, Low Freq.")
    plt.plot(snr_dbs, bers_simple_eu, label="Simple Decoder")
    plt.plot(snr_dbs, bers_raw_eu, label="Raw")
    plt.axhline(y=1/n, color='r', linestyle='--', label="Correctable")
    plt.yscale('log')
    plt.legend()
    plt.show()

    plt.title("BER vs. SNR, Low Frequency Noise")
    plt.xlabel("$E_{b}/N_{0}$ (dB)")
    plt.ylabel("BER")
    plt.plot(snr_dbs, bers_mu_ef, label="DNN Decoder")
    plt.plot(snr_dbs, bers_mf_ef, label="DNN Decoder, Low Freq.")
    plt.plot(snr_dbs, bers_simple_ef, label="Simple Decoder")
    plt.plot(snr_dbs, bers_raw_ef, label="Raw")
    plt.axhline(y=1 / n, color='r', linestyle='--', label="Correctable")
    plt.yscale('log')
    plt.legend()
    plt.show()

    plt.title("BER vs. SNR")
    plt.xlabel("$E_{b}/N_{0}$ (dB)")
    plt.ylabel("BER")
    plt.plot(snr_dbs, bers_raw_eu, label="Raw")
    plt.plot(snr_dbs, bers_raw_ef, label="Raw, Low Freq.")
    plt.axhline(y=1 / n, color='r', linestyle='--', label="Correctable")
    plt.yscale('log')
    plt.legend()
    plt.show()

    print("Done.")

if __name__ == '__main__':
    main()
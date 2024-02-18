import numpy as np
import matplotlib.pyplot as plt

def plot_constellation(symbols, title='16QAM Constellation Diagram'):
    # Create a new figure
    plt.figure(figsize=(8, 8))
    
    # Scatter plot of the real and imaginary parts of the symbols
    plt.scatter(symbols.real, symbols.imag, color='blue', marker='o', label='Mapped Symbols')
    
    # Labeling the plot
    plt.title(title)
    plt.xlabel('In-phase (I)')
    plt.ylabel('Quadrature (Q)')
    plt.grid(True)  # Add grid for better readability
    plt.axhline(0, color='black', lw=0.5)
    plt.axvline(0, color='black', lw=0.5)
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    
    # Adding constellation points for reference
    for real in [-3, -1, 1, 3]:
        for imag in [-3, -1, 1, 3]:
            plt.plot(real, imag, 'rx')  # 'rx' makes the marker red x
    
    plt.legend()
    plt.show()

def map_16qam(bits):
    # Define 16QAM symbol mapping (Gray coding)
    mapping = {
        (0, 0, 0, 0): -3-3j,
        (0, 0, 0, 1): -3-1j,
        (0, 0, 1, 1): -3+1j,
        (0, 0, 1, 0): -3+3j,
        (0, 1, 1, 0): -1+3j,
        (0, 1, 1, 1): -1+1j,
        (0, 1, 0, 1): -1-1j,
        (0, 1, 0, 0): -1-3j,
        (1, 1, 0, 0):  1-3j,
        (1, 1, 0, 1):  1-1j,
        (1, 1, 1, 1):  1+1j,
        (1, 1, 1, 0):  1+3j,
        (1, 0, 1, 0):  3+3j,
        (1, 0, 1, 1):  3+1j,
        (1, 0, 0, 1):  3-1j,
        (1, 0, 0, 0):  3-3j,
    }
    # Reshape bits to groups of 4
    bits_reshaped = bits.reshape((-1, 4))
    symbols = np.array([mapping[tuple(b)] for b in bits_reshaped])
    return symbols

""" # Map source bits to 16QAM symbols
symbols = map_16qam(source_bits) """

""" The OFDM modulator involves taking the 16QAM mapped symbols, 
    performing an IFFT to convert them to the time domain, 
    adding a cyclic prefix, and, if necessary, 
    performing operations to reduce the peak-to-average power ratio. """

def ofdm_modulate(symbols, N_symbols, K, K_used, CP_length):
    # Allocate space for OFDM symbols with guards and convert to time domain
    ofdm_symbols = np.zeros((N_symbols, K), dtype=complex)
    symbols_reshaped = symbols.reshape((N_symbols, K_used))
    ofdm_symbols[:, (K-K_used)//2:(K+K_used)//2] = symbols_reshaped  # Place symbols in the middle
    ofdm_time = np.fft.ifft(np.fft.fftshift(ofdm_symbols, axes=1), n=K, axis=1)
    
    # Add cyclic prefix
    ofdm_time_cp = np.hstack((ofdm_time[:, -CP_length:], ofdm_time))
    
    return ofdm_time_cp

""" # Parameters
CP_length = K // 4  # Length of cyclic prefix

# Modulate
ofdm_time_cp = ofdm_modulate(symbols, K, K_used, CP_length) """

def insert_preamble(ofdm_frame, preamble):
    # Prepend preamble to the OFDM frame
    return np.vstack((preamble, ofdm_frame))

""" # Generate a simple preamble (can be replaced with a more sophisticated one)
preamble_length = K + CP_length  # Same length as an OFDM symbol with CP
preamble = np.ones(preamble_length) + 1j * np.ones(preamble_length)  # Example preamble

# Insert preamble
ofdm_frame_with_preamble = insert_preamble(ofdm_time_cp, preamble) """

""" Fading channel with AWGN based on the specified channel model. 
    The channel has 10 taps with amplitudes distributed as N(0, 2^(-l))/1.998, l=0,…,9. 
    We also add white Gaussian noise based on a specified SNR. """

def fading_channel(ofdm_frame, SNR_dB):
    # Channel parameters
    L = 10  # Number of channel taps
    h = np.array([np.random.normal(0, np.sqrt(2**(-l)/1.998), 1) + 1j*np.random.normal(0, np.sqrt(2**(-l)/1.998), 1) for l in range(L)])
    
    # Convolve signal with channel
    ofdm_frame_conv = np.convolve(ofdm_frame.flatten(), h.flatten(), mode='full')[:ofdm_frame.shape[0]]
    
    # Add noise
    SNR = 10**(SNR_dB / 10.0)  # Convert dB to linear
    signal_power = np.mean(np.abs(ofdm_frame_conv)**2)
    noise_power = signal_power / SNR
    noise = np.sqrt(noise_power / 2) * (np.random.randn(*ofdm_frame_conv.shape) + 1j*np.random.randn(*ofdm_frame_conv.shape))
    ofdm_frame_noisy = ofdm_frame_conv + noise
    
    return ofdm_frame_noisy, h

""" # Simulate channel
SNR_dB = 20  # Example SNR value
ofdm_frame_noisy, channel_h = fading_channel(ofdm_frame_with_preamble, SNR_dB) """

def ofdm_demodulate(ofdm_frame_noisy, CP_length, preamble_length, K):
    # Remove cyclic prefix
    ofdm_frame_noisy = ofdm_frame_noisy[:preamble_length][CP_length:]
    
    # Perform FFT
    ofdm_symbols_freq = np.fft.fftshift(np.fft.fft(ofdm_frame_noisy, n=K))
    
    return ofdm_symbols_freq

""" # Demodulate the received frame
ofdm_symbols_freq = ofdm_demodulate(ofdm_frame_noisy, CP_length, K) """


def channel_estimation(received_preamble, known_preamble, K):
    # Perform least squares estimation
    H_est = received_preamble / known_preamble
    
    # Smoothing (simple example, can be improved)
    H_est_smooth = np.convolve(H_est, np.ones(3)/3, mode='same')
    
    return H_est_smooth

""" # Estimate the channel based on the preamble
channel_estimated = channel_estimation(ofdm_symbols_freq[:preamble_length], preamble, K) """

def zf_equalize(ofdm_symbols_freq, channel_estimated, K_used, K):
    # Equalize using ZF
    H_est_full = np.zeros(K, dtype=complex)
    H_est_full[(K-K_used)//2:(K+K_used)//2] = channel_estimated  # Place estimated channel in the middle
    symbols_equalized = ofdm_symbols_freq / H_est_full
    
    return symbols_equalized[(K-K_used)//2:(K+K_used)//2]  # Return only the used subcarriers

def demap_16qam(symbols):
    # Inverse of the 16QAM mapping function
    bits = []
    for s in symbols:
        if np.real(s) < -2:
            bits.append([0, 0])
        elif np.real(s) < 0:
            bits.append([0, 1])
        elif np.real(s) < 2:
            bits.append([1, 1])
        else:
            bits.append([1, 0])
        
        if np.imag(s) < -2:
            bits.append([0, 0])
        elif np.imag(s) < 0:
            bits.append([0, 1])
        elif np.imag(s) < 2:
            bits.append([1, 1])
        else:
            bits.append([1, 0])
    return np.array(bits).flatten()

""" # Equalize and demap the received OFDM symbols
symbols_equalized = zf_equalize(ofdm_symbols_freq[preamble_length:], channel_estimated, K_used, K)
demapped_bits = demap_16qam(symbols_equalized) """


def compute_ber(transmitted_bits, received_bits):
    # Compute the number of bit errors
    bit_errors = np.sum(transmitted_bits != received_bits)
    # Compute BER
    ber = bit_errors / len(transmitted_bits)
    return ber

""" # Compute BER
ber = compute_ber(source_bits, demapped_bits[:len(source_bits)])  # Ensure length match
print(f"Bit error rate: {ber:.2e}") """

def main():
    # Simulation parameters
    K = 64  # Total number of subcarriers
    K_used = 48  # Number of used subcarriers
    CP_length = K // 4  # Length of cyclic prefix
    N_symbols = 20  # Number of OFDM symbols per frame
    bits_per_symbol = 4  # For 16QAM
    N_bits = K_used * bits_per_symbol * N_symbols  # Total number of bits per frame
    SNR_dB = 20  # Signal-to-Noise Ratio in dB

    print("Starting OFDM simulation...")

    # Step 1: Generate Source Bits
    source_bits = np.random.randint(0, 2, N_bits)
    print("Source bits generated : " + str(source_bits) + " and size : " + str(source_bits.shape[0]))

    # Step 2: 16QAM Mapping
    symbols = map_16qam(source_bits)
    plot_constellation(symbols, title='16QAM Constellation Diagram')
    # print("16QAM mapping completed : " + str(symbols))

    # Step 3: OFDM Modulation
    ofdm_time_cp = ofdm_modulate(symbols, N_symbols, K, K_used, CP_length)
    print("OFDM modulation completed : " + str(ofdm_time_cp) + " and size : " + str(ofdm_time_cp.shape))

    # Step 4: Preamble Insertion
    preamble_length = K + CP_length
    preamble = np.ones(preamble_length)
    ofdm_frame_with_preamble = insert_preamble(ofdm_time_cp, preamble)
    print("Preamble inserted : " + str(ofdm_frame_with_preamble) + " and size : " + str(ofdm_frame_with_preamble.shape))

    # Step 5: Channel Simulation
    ofdm_frame_noisy, channel_h = fading_channel(ofdm_frame_with_preamble, SNR_dB)
    print("Channel simulation completed : " + str(ofdm_frame_noisy) + " and channel_h : " + str(channel_h) + " channel size : " + str(channel_h.shape))

    # Step 6: OFDM Demodulation
    ofdm_symbols_freq = ofdm_demodulate(ofdm_frame_noisy, CP_length, preamble_length, K)
    print("OFDM demodulation completed : " + str(ofdm_symbols_freq) + " and size : " + str(ofdm_symbols_freq.shape[0])) 

    # Step 7: Channel Estimation
    print("Preamble length : " + str(preamble_length) + " and preamble : " + str(preamble))
    print("OFDM symbols freq : " + str(ofdm_symbols_freq[:preamble_length]) + "and size : " + str(ofdm_symbols_freq[:preamble_length].shape[0]))
    channel_estimated = channel_estimation(ofdm_symbols_freq[:preamble_length], preamble, K)
    print("Channel estimation completed : " + str(channel_estimated)) 

    # Step 8: Coherent Detection and Demapping
    symbols_equalized = zf_equalize(ofdm_symbols_freq[preamble_length:], channel_estimated, K_used, K)
    plot_constellation(symbols_equalized, title='16QAM Equalized Constellation Diagram')
    demapped_bits = demap_16qam(symbols_equalized)
    print("Coherent detection and demapping completed : " + str(symbols_equalized) + " and demapped_bits : " + str(demapped_bits)) 

    # Step 9: BER Computation
    ber = compute_ber(source_bits, demapped_bits[:len(source_bits)])  # Ensure length match
    print(f"BER computation completed. BER = {ber}")

if __name__ == "__main__":
    main()
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d


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

def plot_channel_response(channel_h):
    # Plot the magnitude and phase of the channel impulse response
    plt.figure(figsize=(14, 5))
    
    # Plot magnitude
    plt.subplot(1, 2, 1)
    plt.stem(np.arange(len(channel_h)), np.abs(channel_h), use_line_collection=True)
    plt.title('Channel Magnitude Response')
    plt.xlabel('Tap')
    plt.ylabel('Magnitude')
    plt.grid(True)
    
    # Plot phase
    plt.subplot(1, 2, 2)
    plt.stem(np.arange(len(channel_h)), np.angle(channel_h), use_line_collection=True)
    plt.title('Channel Phase Response')
    plt.xlabel('Tap')
    plt.ylabel('Phase (radians)')
    plt.grid(True)
    
    plt.tight_layout()
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

def generate_PLCP_preamble(K, CP_length):
       
    # Generate two long symbols based on the given L sequence
    L_sequence = [1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 0, 1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1]

    # Convert the L_sequence to a numpy array for easier handling
    L_seq = np.array(L_sequence)
    
    # Ensure the sequence fits within the FFT size, padding with zeros if necessary
    if len(L_seq) < K:
        L_seq_padded = np.fft.ifftshift(np.concatenate([
            np.zeros((K - len(L_seq)) // 2),
            L_seq,
            np.zeros(K - len(L_seq) - (K - len(L_seq)) // 2)
        ]))
    else:
        L_seq_padded = L_seq
    
    # Generate the LTS in time domain
    lts_time = np.fft.ifft(L_seq_padded)
    
    # Prepend CP to LTS
    lts_with_cp = np.concatenate((lts_time[-CP_length:], lts_time))
    
    # Assuming a simple STS for illustration purposes (not based on your input)
    sts = np.tile(np.array([1, -1]), K // 2)
    sts_with_cp = np.concatenate((sts[-CP_length:], sts))  # Optionally add CP to STS if needed
    
    # Combine STS and LTS for the complete preamble
    preamble = np.concatenate((sts_with_cp, lts_with_cp))
    
    return preamble

    """ 
    print(len(L_sequence))
    L = np.array([0] * ((K - len(L_sequence)) // 2) + L_sequence + [0] * ((K - len(L_sequence)) // 2))
    long_symbol_freq = np.fft.ifft(np.fft.fftshift(L))  # Convert L to frequency domain
    long_preamble = np.tile(long_symbol_freq, (2, 1))  # Repeat long symbol 2 times
    
    long_preamble_with_cp = np.array([np.concatenate((symbol[-CP_length:], symbol)) for symbol in long_preamble])
    
    return long_preamble_with_cp """

import numpy as np

def generate_zadoff_chu_seq(K, CP_length):
    # Generate the ZC sequence
    n = np.arange(K)
    zc_seq = np.exp(-1j * np.pi * 1 * n * (n + 1) / K)
    
    # Add the cyclic prefix
    cp = zc_seq[-CP_length:]  # Extract the last CP_length samples for the CP
    zc_seq_with_cp = np.concatenate((cp, zc_seq))  # Prepend CP to the ZC sequence
    
    return zc_seq_with_cp


def ofdm_modulate(symbols, preamble, N_symbols, K, K_used, CP_length):
        
    # Allocate space for OFDM symbols with guards and convert to time domain
    ofdm_symbols = np.zeros((N_symbols, K), dtype=complex)
    symbols_reshaped = symbols.reshape((N_symbols, K_used))
    ofdm_symbols[:, (K-K_used)//2:(K+K_used)//2] = symbols_reshaped  # Place symbols in the middle

    # Convert OFDM symbols from frequency to time domain using IFFT
    ofdm_time_domain = np.fft.ifft(ofdm_symbols, axis=1)
    
    # Add cyclic prefix
    ofdm_time_cp = np.hstack((ofdm_time_domain[:, -CP_length:], ofdm_time_domain))
    
    plot_constellation(ofdm_time_cp, title="Modulated OFDM symbols")

    return ofdm_time_cp

def insert_preamble(ofdm_frame, preamble):
    # Prepend preamble to the OFDM frame
    return np.vstack((preamble, ofdm_frame))

def fading_channel(ofdm_frame, SNR_dB):
    # Channel parameters
    L = 10  # Number of channel taps
    h = np.array([np.random.normal(0, np.sqrt(2**(-l)/1.998), 1) + 1j*np.random.normal(0, np.sqrt(2**(-l)/1.998), 1) for l in range(L)])
    
    # Reshape h for 2D convolution (make it a 2D array)
    h_2d = h.reshape((L, 1))

    # Convolve signal with channel
    ofdm_frame_conv = convolve2d(ofdm_frame, h_2d, mode='full')[:ofdm_frame.shape[0], :ofdm_frame.shape[1]]

    # Add noise
    SNR = 10**(SNR_dB / 10.0)  # Convert dB to linear
    signal_power = np.mean(np.abs(ofdm_frame_conv)**2)
    noise_power = signal_power / SNR
    noise = np.sqrt(noise_power / 2) * (np.random.randn(*ofdm_frame_conv.shape) + 1j*np.random.randn(*ofdm_frame_conv.shape))
    ofdm_frame_noisy = ofdm_frame_conv + noise
    
    return ofdm_frame_noisy, h

def ofdm_demodulate(ofdm_frame_noisy, N_symbols, CP_length, K):
    ofdm_symbols_freq = np.zeros((N_symbols+1, K), dtype=complex) # N_symbols + 1 because of the preamble
    
    for i in range(N_symbols+1):
        # Remove cyclic prefix: Assuming CP is at the start of the symbol
        symbol_without_cp = ofdm_frame_noisy[i, CP_length:CP_length+K]
        
        # Perform FFT to convert the symbol back to the frequency domain
        ofdm_symbols_freq[i, :] = np.fft.fft(symbol_without_cp)
    
    return ofdm_symbols_freq



def channel_estimation(received_preamble, original_preamble, K):
     # Ensure preamble is in frequency domain
    received_preamble_fd = np.fft.fft(received_preamble, n=K)
    original_preamble_fd = np.fft.fft(original_preamble, n=K)
    
    # Perform least squares estimation by dividing the received by the original (in frequency domain)
    H_est = received_preamble_fd / original_preamble_fd
    
    # Smoothing
    H_est_smooth = np.array([np.convolve(h, np.ones(3)/3, mode='same') for h in H_est])
    
    return H_est_smooth

def zf_equalize(ofdm_symbols_freq, channel_estimated, K_used, K):
    
    # Apply the channel estimation to the entire array of received symbols
    # This assumes channel_estimated is for all K subcarriers
    symbols_equalized_full = ofdm_symbols_freq / channel_estimated

    # Extract only the used subcarriers for further processing
    # Assuming used subcarriers are centered in the spectrum
    start_index = (K - K_used) // 2
    end_index = start_index + K_used
    symbols_equalized = symbols_equalized_full[start_index:end_index]

    return symbols_equalized


""" def zf_equalize(ofdm_symbols_freq, channel_estimated, K_used, K):

    # Equalize using ZF
    H_est_full = np.zeros(K, dtype=complex)
    H_est_full[(K-K_used)//2:(K+K_used)//2] = channel_estimated  # Place estimated channel in the middle
    symbols_equalized = ofdm_symbols_freq / H_est_full

    return symbols_equalized[(K-K_used)//2:(K+K_used)//2]  # Return only the used subcarriers
 """
def demap_16qam(symbols):
    # Inverse of the 16QAM mapping function
    bits = []
    for s in np.array(symbols).flatten():
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

def compute_ber(transmitted_bits, received_bits):
    # Compute the number of bit errors
    bit_errors = np.sum(transmitted_bits != received_bits)
    # Compute BER
    ber = bit_errors / len(transmitted_bits)
    return ber

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
    print("Number of bits generated : " + str(source_bits.shape))

    # Step 2: 16QAM Mapping
    symbols = map_16qam(source_bits)
    # plot_constellation(symbols, title='16QAM Constellation Diagram')
    print("16QAM mapping completed : " + str(symbols.shape))

    # Step 3: Generate preamble
    # preamble_PTS = generate_preamble(K, CP_length)
    preamble_length = K + CP_length
    preamble = generate_zadoff_chu_seq(K, CP_length)

    # Step 4: OFDM Modulation
    ofdm_modulated = ofdm_modulate(symbols, preamble, N_symbols, K, K_used, CP_length)
    # plot_constellation(ofdm_frame_with_preamble, title="test")
    print("OFDM modulation completed : " + str(ofdm_modulated.shape))

    # Step 4: Preamble Generation
    
    # preamble = generate_preamble(ofdm_frame_with_preamble)
    ofdm_frame_with_preamble = insert_preamble(ofdm_modulated, preamble)
    print("Preamble inserted : " + str(ofdm_frame_with_preamble.shape))

    # Step 5: Channel Simulation
    ofdm_frame_noisy, channel_h = fading_channel(ofdm_frame_with_preamble, SNR_dB)
    # plot_channel_response(channel_h)

    print("Channel simulation completed : " + str(ofdm_frame_noisy.shape))

    # Step 6: OFDM Demodulation
    ofdm_symbols_freq = ofdm_demodulate(ofdm_frame_noisy, N_symbols, CP_length, K)
    print("OFDM demodulation completed : " + str(ofdm_symbols_freq.shape)) 

    # Step 7: Channel Estimation
    channel_estimated = channel_estimation(ofdm_frame_noisy[:preamble_length], preamble, K)
    # plot_channel_response(channel_estimated)
    print("Channel estimation completed : " + str(channel_estimated.shape)) 

    # Step 8: Coherent Detection and Demapping
    symbols_equalized = zf_equalize(ofdm_symbols_freq, channel_estimated, K_used, K)
    print("Equalization completed : " + str(symbols_equalized.shape))


    # nearest neighbor quantization

    plot_constellation(symbols_equalized, title='Received OFDM symbols')
    demapped_bits = demap_16qam(symbols_equalized)
    print("Coherent detection and demapping completed : " + str(demapped_bits.shape)) 

    # Step 9: BER Computation
    ber = compute_ber(source_bits, demapped_bits) 
    print(f"BER computation completed. BER = {ber}")

if __name__ == "__main__":
    main()
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d


def plot_constellation(symbols, title='16QAM Constellation Diagram'):
    plt.figure(figsize=(8, 8))
    
    plt.scatter(symbols.real, symbols.imag, color='blue', marker='o', label='Mapped Symbols')

    plt.title(title)
    plt.xlabel('In-phase (I)')
    plt.ylabel('Quadrature (Q)')
    plt.grid(True)  
    plt.axhline(0, color='black', lw=0.5)
    plt.axvline(0, color='black', lw=0.5)
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)

    for real in [-3, -1, 1, 3]:
        for imag in [-3, -1, 1, 3]:
            plt.plot(real, imag, 'rx') 
    
    plt.legend()
    plt.show()

def plot_channel_response(channel_h):
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
    bits_reshaped = bits.reshape((-1, 4))
    symbols = np.array([mapping[tuple(b)] for b in bits_reshaped])
    return symbols

def generate_PLCP_preamble(K, CP_length): # For synchronization
    L_sequence = [1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 0, 1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1]

    L_seq = np.array(L_sequence)

    if len(L_seq) < K:
        L_seq_padded = np.fft.ifftshift(np.concatenate([
            np.zeros((K - len(L_seq)) // 2),
            L_seq,
            np.zeros(K - len(L_seq) - (K - len(L_seq)) // 2)
        ]))
    else:
        L_seq_padded = L_seq
    
    lts_time = np.fft.ifft(L_seq_padded)
    
    lts_with_cp = np.concatenate((lts_time[-CP_length:], lts_time))
    
    sts = np.tile(np.array([1, -1]), K // 2)
    sts_with_cp = np.concatenate((sts[-CP_length:], sts))  
    
    preamble = np.concatenate((sts_with_cp, lts_with_cp))
    
    return preamble

def generate_zadoff_chu_seq(K, CP_length):
    n = np.arange(K)
    zc_seq = np.exp(-1j * np.pi * 1 * n * (n + 1) / K)
    
    cp = zc_seq[-CP_length:]  
    zc_seq_with_cp = np.concatenate((cp, zc_seq))  
    
    return zc_seq_with_cp

def ofdm_modulate(symbols, N_symbols, K, K_used, CP_length):
    ofdm_symbols_freq_domain = np.zeros((N_symbols, K), dtype=complex)
    
    mid_point = K // 2
    ofdm_symbols_freq_domain[:, mid_point - K_used//2:mid_point + K_used//2] = symbols.reshape((N_symbols, K_used))

    ofdm_symbols_time_domain = np.fft.ifft(ofdm_symbols_freq_domain, axis=1)
        
    ofdm_symbols_with_cp = np.hstack((ofdm_symbols_time_domain[:, -CP_length:], ofdm_symbols_time_domain))
    
    return ofdm_symbols_with_cp

def insert_preamble(ofdm_frame, preamble):
    return np.vstack((preamble, ofdm_frame))

def fading_channel(ofdm_frame, SNR_dB):
    # Channel parameters
    L = 10  
    h = np.array([np.random.normal(0, np.sqrt(2**(-l)/1.998), 1) + 1j*np.random.normal(0, np.sqrt(2**(-l)/1.998), 1) for l in range(L)])
    
    h_2d = h.reshape((L, 1))

    ofdm_frame_conv = convolve2d(ofdm_frame, h_2d, mode='full')[:ofdm_frame.shape[0], :ofdm_frame.shape[1]]

    # Add noise
    SNR = 10**(SNR_dB / 10.0) 
    signal_power = np.mean(np.abs(ofdm_frame_conv)**2)
    noise_power = signal_power / SNR
    noise = np.sqrt(noise_power / 2) * (np.random.randn(*ofdm_frame_conv.shape) + 1j*np.random.randn(*ofdm_frame_conv.shape))
    ofdm_frame_noisy = ofdm_frame_conv + noise
    
    return ofdm_frame_noisy, h

def ofdm_demodulate(ofdm_frame_noisy, N_symbols, CP_length, K):
    ofdm_symbols_freq = np.zeros((N_symbols+1, K), dtype=complex) # N_symbols + 1 because of the preamble
    
    for i in range(N_symbols+1):
        symbol_without_cp = ofdm_frame_noisy[i, CP_length:CP_length+K]
        
        ofdm_symbols_freq[i, :] = np.fft.fft(symbol_without_cp)
    
    return ofdm_symbols_freq

def channel_estimation(received_preamble, original_preamble):
    received_preamble_fd = np.fft.fft(received_preamble, n=80)
    original_preamble_fd = np.fft.fft(original_preamble, n=80)
    
    H_est = received_preamble_fd / original_preamble_fd
    
    H_est_smooth = np.array([np.convolve(h, np.ones(3)/3, mode='same') for h in H_est])
    
    return H_est_smooth

def zf_equalize(ofdm_symbols_freq, channel_estimated, K_used, K, N_symbols):
    symbols_equalized = np.zeros((N_symbols, K_used), dtype=complex)  # Adjusted to N_symbols

    start_index = (K - K_used) // 2
    for i in range(N_symbols):         
        symbols_equalized[i, :] = ofdm_symbols_freq[i + 1, start_index:start_index + K_used] / channel_estimated[i + 1, start_index:start_index + K_used]

    return symbols_equalized

def nearest_neighbor_quantization(symbols):
    constellation_points = [
        -3-3j, -3-1j, -3+1j, -3+3j,
        -1-3j, -1-1j, -1+1j, -1+3j,
        1-3j,  1-1j,  1+1j,  1+3j,
        3-3j,  3-1j,  3+1j,  3+3j
    ]
    
    quantized_symbols = np.zeros(symbols.shape, dtype=complex)
    
    for i in range(symbols.shape[0]):
        for j in range(symbols.shape[1]):
            symbol = symbols[i, j]
            distances = [np.abs(symbol - cp) for cp in constellation_points]
            nearest_point_index = np.argmin(distances)
            quantized_symbols[i, j] = constellation_points[nearest_point_index]
    
    return quantized_symbols

def demap_16qam(symbols):
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
    bit_errors = np.sum(transmitted_bits != received_bits)
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

    snr_values = [0, 5, 10, 15, 20, 25, 30]
    ber_values = []

    for SNR_dB in snr_values:

        print("Starting OFDM simulation for SNR = " + str(SNR_dB) + " dB...")

        # Step 1: Generate Source Bits
        source_bits = np.random.randint(0, 2, N_bits)
        print("Number of bits generated : " + str(source_bits.shape))

        # Step 2: 16QAM Mapping
        symbols = map_16qam(source_bits)
        # plot_constellation(symbols, title='16QAM Constellation Diagram')
        print("16QAM mapping completed : " + str(symbols.shape))

        # Step 3: OFDM Modulation
        ofdm_modulated = ofdm_modulate(symbols, N_symbols, K, K_used, CP_length)
        print("OFDM modulation completed : " + str(ofdm_modulated.shape))

        # Step 4: Preamble
        # preamble_PTS = generate_preamble(K, CP_length)
        preamble_length = K + CP_length
        preamble = generate_zadoff_chu_seq(K, CP_length) # made so that Fourrier transform will not give zero values
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
        channel_estimated = channel_estimation(ofdm_frame_noisy[:preamble_length], preamble)
        print("Channel estimation completed : " + str(channel_estimated.shape)) 

        # Step 8: Equalization
        symbols_equalized = zf_equalize(ofdm_symbols_freq, channel_estimated, K_used, K, N_symbols)
        print("Equalization completed : " + str(symbols_equalized.shape))

        # Step 9: Nearest Neighbor Quantization
        symbols_quantized = nearest_neighbor_quantization(symbols_equalized)
        print("Nearest neighbor quantization completed : " + str(symbols_quantized.shape))

        # Step 10: Demapping 
        demapped_bits = demap_16qam(symbols_quantized)
        print("Coherent detection and demapping completed : " + str(demapped_bits.shape)) 

        # Step 11: BER Computation
        ber = compute_ber(source_bits, demapped_bits) 
        print(f"BER computation completed. BER = {ber}")
        ber_values.append(ber)

    plt.figure(figsize=(10, 6))
    plt.semilogy(snr_values, ber_values, '-o')
    plt.title('BER vs. SNR')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Bit Error Rate (BER)')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
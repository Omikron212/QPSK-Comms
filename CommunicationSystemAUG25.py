# QPSK Communication System
# Michael Rogers | AUG/2025

# Communicates strings from the source to the sink. WGN added
# Uses QPSK bit modulation, rectangular pulses, matched filter

#------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

# BIT ENCODING (Strings into bits)

def bitEncoder(string):
    
    strLen = len(string)
    bitSequence = []
    
    for i in range(strLen):
        
        asc = bin(ord(string[i]))[2:] # Take the binary ascii code for each character
        
        while (len(asc) < 8):
            
            asc = '0' + asc # Pad it with 0s until it is 8 characters long
        
        for j in range(8):
            
            bitSequence.append(int(asc[j])) # Assemble the bit sequence for the string
            
    return np.array(bitSequence)
    
# MODULATION MAPPING (Bits into symbols)

def modulationMapper(bitSequence, constellation):
    
    symbolSequence = []
    
    for i in range(len(bitSequence) // 2): # Because each symbol correlates to two bits, the symbol sequence is half the length of the bit sequence
        
        symbolSequence.append(constellation[str(bitSequence[2 * i]) + str(bitSequence[2 * i + 1])]) # The bits are passed through the constellation to retrieve the symbols
        
    return np.array(symbolSequence)

# SHAPING (Symbols into baseband signal)

def signalSynthesis(symbolSequence, res): # res is "resolution", or points per symbol
    
    time = np.arange(0, len(symbols), 1 / res) # Form a time grid based on number of symbols
    real_signal = np.zeros(len(symbols) * res) # Split the real and imaginary signals. In analog systems, they are consolated via upconversion
    imag_signal = np.zeros(len(symbols) * res)
    
    for i in range(len(symbols)):
        
        for j in range(res):
            
            real_signal[res * i + j] += symbols[i].real # The real signal contains the real components of each symbol
            imag_signal[res * i + j] += symbols[i].imag # The imaginary signal contains the imaginary components of each symbol
    
    return time, real_signal, imag_signal

# CHANNEL (baseband signal into noisy signal)

def channel(original_real_signal, original_imag_signal, stdev):
    
    new_real_signal = original_real_signal + np.random.normal(0, stdev, size=original_real_signal.shape) # Each signal experiences additive white Gaussian noise
    new_imag_signal = original_imag_signal + np.random.normal(0, stdev, size=original_imag_signal.shape)
    
    return new_real_signal, new_imag_signal

# DEMODULATION (estimated symbols into estimated bits)

def demodulator(symbolSequence, constellation):
    
    bitSequence = []
    
    for i in range(len(symbolSequence)):
        
        bitSequence.append(int(constellation[symbolSequence[i]][0])) # The constellation is uesd to retrieve the bits from the symbol pattern
        bitSequence.append(int(constellation[symbolSequence[i]][1]))
    
    return np.array(bitSequence)

# DECODING (estimated bits into estimated string)

def decoder(bitSequence):
    
    asc = ''
    string = ''
    
    for i in range(len(bitSequence)):
        
        if (i + 1) % 8 == 0: # If the current character is the 8th, the ascii code gets converted into a character and added to the return string
            
            asc += str(bitSequence[i])
            asc.lstrip('0')
            string += chr(int(asc, 2))
            asc = ''
            
        else:
            
            asc += str(bitSequence[i])
    
    return string

#------------------------------------------------------------------------------

msg = input("Hello there! Please enter a string: ")
QPSK_modulator = {'00' : 1 + 1j, '01' : -1 + 1j, '11' : -1 - 1j, '10' : 1 - 1j} # Defining modulation and demodulation dictionaries
QPSK_demodulator = {1 + 1j : '00', -1 + 1j : '01', -1 - 1j : '11', 1 - 1j : '10'}

bits = bitEncoder(msg)
        
#print('BITS:', bits)

symbols = modulationMapper(bits, QPSK_modulator) # Bits are turned into symbols

#print('SYMBOLS:', symbols)

res = 100 # Points per symbol
time_grid, I_signal, Q_signal = signalSynthesis(symbols, res) # Symbols are turned into signals

N0 = 2 # Noise level
noisy_I_signal, noisy_Q_signal = channel(I_signal, Q_signal, N0)

# MATCHED FILTER (noisy signal into estimated symbols)

pulse = np.ones(res) # Defining a pulse function

est_I_signal = np.convolve(noisy_I_signal, pulse, mode='same') / res # Applying the matched filter
est_Q_signal = np.convolve(noisy_Q_signal, pulse, mode='same') / res

I_samples = est_I_signal[res // 2 - 1 : : res] # Collecting samples from the filtered signal, accounting for convolution delay of half a symbol period
Q_samples = est_Q_signal[res // 2 - 1 : : res]

est_I_symbols = np.where(I_samples >= 0, 1, -1) # Decision rule is applied to each sample set to retrieve estimated symbol pattern
est_Q_symbols = np.where(Q_samples >= 0, 1, -1)

est_symbols = []

for i in range(len(est_I_symbols)):
    
    est_symbols.append(complex(est_I_symbols[i], est_Q_symbols[i])) # The complex symbols are recovered

est_symbols = np.array(est_symbols)

#print('EST SYMBOLS:', est_symbols)

est_bits = demodulator(est_symbols, QPSK_demodulator) # Turning symbols into bits

#print('EST BITS:', est_bits)

est_msg = decoder(est_bits) # Turning bits into an estimated messages

print('ESTIMATED MESSAGE:', est_msg)

plt.plot(time_grid, noisy_I_signal, label='Noisy Signal') # Plotting the noisy I-channel signal
plt.plot(time_grid, I_signal, 'r', linestyle='--', label='Transmitted Signal') # Plotting the transmitted I-channel signal
plt.legend(loc='upper right')
plt.title("Noisy I-Channel Signal")
plt.ylabel("Symbols")
plt.xlabel("Time")
plt.grid()
plt.show()

plt.plot(time_grid, noisy_Q_signal, label='Noisy Signal') # Plotting the noisy Q-channel signal
plt.plot(time_grid, Q_signal,'r', linestyle='--', label='Transmitted Signal') # Plotting the transmitted Q-channel signal
plt.legend(loc='upper right')
plt.title("Noisy Q-Channel Signal")
plt.ylabel("Symbols")
plt.xlabel("Time")
plt.grid()
plt.show()

plt.plot(time_grid, est_I_signal, label='Filtered Signal') # Plotting the filtered I-channel signal
plt.plot(time_grid[res // 2 - 1 : : res], I_samples, 'x', color='r', label='Collected Samples') # Plotting the real components of the collected samples
plt.legend(loc='upper right')
plt.title("Filtered I-Channel Signal")
plt.ylabel("Symbols")
plt.xlabel("Time")
plt.grid()
plt.show()

plt.plot(time_grid, est_Q_signal, label='Filtered Signal') # Plotting the filtered Q-channel signal
plt.plot(time_grid[res // 2 - 1 : : res], Q_samples, 'x', color='r', label='Collected Samples') # Plotting the imaginary components of the collected samples
plt.legend(loc='upper right')
plt.title("Filtered Q-Channel Signal")
plt.ylabel("Symbols")
plt.xlabel("Time")
plt.grid()
plt.show()

plt.plot(np.array([-2, 2]), np.zeros(2), '--', color='b', label='Decision Boundary') # Plotting the decision boundaries
plt.plot(np.zeros(2), np.array([-2, 2]), '--', color='b')
plt.scatter(I_samples, Q_samples, marker='x', color='r', label='Collected Samples')  # Plotting the collected samples on the complex plane
plt.legend(loc='upper right')
plt.title("Collected Samples")
plt.ylabel('imag(s)')
plt.xlabel('real(s)')
plt.grid()
plt.show()
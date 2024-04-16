import numpy as np
from dimod.generators.wireless import mimo, _constellation_properties, _symbols_to_bits
from dwave.system import DWaveSampler, EmbeddingComposite

# Define the system parameters
num_receiver = 1  # Number of receive antennas (base station)
num_users = 3  # Number of users

# Define the user distances and path loss exponent
user_distances = np.array([50, 75, 100])  # Distances of users from the base station
path_loss_exponent = 2  # Path loss exponent

# Define the noise variance and transmit power
noise_variance = 0.1  # Adjust the noise variance as needed
transmit_power = 1  # Transmit power of each user (assumed to be the same for all users)

# Generate the Rayleigh fading coefficients for each user
rayleigh_fading = np.random.rayleigh(scale=1, size=(num_receiver, num_users))

# Calculate the channel coefficients based on Rayleigh fading and path loss
channel_coefficients = rayleigh_fading / np.sqrt(user_distances ** path_loss_exponent)

# Scale the channel coefficients with the square root of the transmit power
channel_coefficients_with_power = np.sqrt(transmit_power) * channel_coefficients

# Decide on the modulation scheme
modulation = 'BPSK'  # BPSK modulation

# Get the constellation properties based on the modulation scheme
bits_per_user, amps, constellation_mean_power = _constellation_properties(modulation)

# Manually generate random transmitted symbols for each user
transmitted_symbols = _create_transmitted_symbols(num_transmitter, amps=[-1, 1], quadrature=False)

# Generate the received signal at the base station (equation (1) in the paper)
received_signal = np.sum(channel_coefficients_with_power * transmitted_symbols) + np.sqrt(noise_variance) * (np.random.randn() + 1j * np.random.randn())

# Reshape the received signal to have a shape of (1,)
received_signal = np.reshape(received_signal, (1,))

# Generate the Ising model using the mimo function from dimod
bqm = mimo(modulation=modulation, y=received_signal, F=channel_coefficients_with_power, channel_noise=np.sqrt(noise_variance), num_receivers=num_receiver, num_transmitters=num_users)

# Set up the D-Wave sampler
sampler = EmbeddingComposite(DWaveSampler())

# Solve the BQM using the D-Wave sampler
sampleset = sampler.sample(bqm, num_reads=1000)

# Get the best solution
best_solution = sampleset.first.sample

# Convert the best solution to a numpy array of spins
spins_best = np.array([best_solution[i] for i in range(len(best_solution))], dtype=int)

# Convert the transmitted symbols to spins
spins_ideal = _symbols_to_bits(transmitted_symbols, modulation=modulation).flatten()

# Calculate the bit error rate
bit_error_rate = np.mean(spins_ideal != spins_best)

# Print the results
print("Transmitted symbols:")
print(transmitted_symbols)
print("Best solution (spins):")
print(spins_best)
print("Bit Error Rate:", bit_error_rate)
print("Number of bits per user:", bits_per_user)
print("Total number of bits sent:", num_users * bits_per_user)
import numpy as np
import random
import math
import time
import os
from numba import jit
import matplotlib.pyplot as plt

# Parameters
NX = 64
NY = 64
ntherm = 1000
VisualDisplay = 1
SleepTime = 300000  # in microseconds

@jit
def update_spin(nx, ny, env, spin):
    """Do a metropolis update on a spin at position (nx, ny) whose environment is env"""
    current_spin = spin[nx, ny]
    newspin = 1 if np.random.random() < 0.5 else -1
    DeltaBetaE = -(newspin - current_spin) * env
    if DeltaBetaE <= 0 or np.random.random() < math.exp(-DeltaBetaE):
        spin[nx, ny] = newspin

@jit
def sweep(beta, h, spin):
    """Sweep through all lattice sites"""
    for nx in range(1, NX + 1):
        for ny in range(1, NY + 1):
            environment = (beta * (spin[nx, ny-1] + spin[nx, ny+1] + 
                                 spin[nx-1, ny] + spin[nx+1, ny]) + h)
            update_spin(nx, ny, environment, spin)

@jit
def initialize_hot(spin):
    """Initialize lattice with random spins"""
    spin = np.zeros((NX + 2, NY + 2), dtype=np.int8)
    spin[1:-1, 1:-1] = np.where(np.random.random((NX, NY)) < 0.5, 1, -1)

@jit
def magnetization(spin):
    """Calculate average magnetization"""
    return np.mean(spin[1:-1, 1:-1])

@jit
def calculate_energy(spin, h):
    energy = 0.0
    for nx in range(1, NX + 1):
        for ny in range(1, NY + 1):
            neighbor_sum = (spin[nx, ny-1] + spin[nx, ny+1] + spin[nx-1, ny] + spin[nx+1, ny])
            energy -= spin[nx, ny] * neighbor_sum / 2.0
            energy -= h * spin[nx, ny]
    return energy

def display_lattice(T, spin):
    """Display the lattice configuration"""
    if SleepTime > 0:
        os.system('cls' if os.name == 'nt' else 'clear')
    
    # Convert spins to characters
    chars = np.where(spin[1:-1, 1:-1] == 1, 'X', '-')
    for row in chars:
        print(''.join(row))
    
    print(f"T = {T:.6f}:   magnetization <sigma> = {magnetization(spin):.6f}")
    
    if SleepTime > 0:
        time.sleep(SleepTime / 1_000_000)  # Convert microseconds to seconds
    else:
        print()

def main():
    # Initialize the lattice with boundary spins as a global numpy array
    spin = np.zeros((NX + 2, NY + 2), dtype=np.int8)
    
    output_filename = "ising2d_vs_T.dat"
    
    print(f"Program calculate <sigma> vs. T for a 2D Ising model of "
          f"{NX}x{NY} spins with free boundary conditions.\n")
    
    np.random.seed(int(time.time()))
    
    nsweep = int(input("Enter # sweeps per temperature sample:\n"))
    h = float(input("Enter value of magnetic field parameter h:\n"))
    Tmax = float(input("Enter starting value (maximum) of temperature T (=1/beta):\n"))
    ntemp = int(input("Enter # temperatures to simulate:\n"))
    
    initialize_hot(spin)
    
    temperatures = []
    energies = []
    magnetizations = []
    specific_heats = []
    
    with open(output_filename, 'w') as output:
        # Do ntemp temperatures between Tmax and 0
        for itemp in range(ntemp, 0, -1):
            T = (Tmax * itemp) / ntemp
            beta = 1/T
            
            # Thermalization sweeps
            for _ in range(ntherm):
                sweep(beta, h, spin)
            
            # Main sweeps
            total_mag = 0.0
            total_energy = 0.0
            total_energy_sq = 0.0
            
            for _ in range(nsweep):
                sweep(beta, h, spin)
                mag = np.sum(spin[1:-1, 1:-1])
                total_mag += mag
                E = calculate_energy(spin, h)
                total_energy += E
                total_energy_sq += E * E
            
            avg_mag = total_mag / (nsweep * NX * NY)
            avg_energy = total_energy / nsweep
            avg_energy_per_spin = avg_energy / (NX * NY)
            avg_energy_sq = total_energy_sq / nsweep
            variance_energy = avg_energy_sq - avg_energy * avg_energy
            specific_heat = variance_energy / (T * T * NX * NY)
            temperatures.append(T)
            energies.append(avg_energy_per_spin)
            magnetizations.append(avg_mag)
            specific_heats.append(specific_heat)
            
            output.write(f"{T:.6f} {avg_mag:.6f}\n")
            
            if VisualDisplay:
                display_lattice(T, spin)
    
    print(f"Output file is {output_filename}")

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12)) 
    ax1.plot(temperatures, energies, 'bo-', linewidth=2, markersize=4)
    ax1.set_xlabel('Temperature', fontsize=12)
    ax1.set_ylabel('Mean Energy per Spin', fontsize=12)
    ax1.set_title(f'Mean Energy vs Temperature', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax2.plot(temperatures, magnetizations, 'ro-', linewidth=2, markersize=4)
    ax2.set_xlabel('Temperature', fontsize=12)
    ax2.set_ylabel('Magnetization', fontsize=12)
    ax2.set_title(f'Magnetization vs Temperature', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax3.plot(temperatures, specific_heats, 'go-', linewidth=2, markersize=4)
    ax3.set_xlabel('Temperature', fontsize=12)
    ax3.set_ylabel('Specific Heat', fontsize=12)
    ax3.set_title(f'Specific Heat vs Temperature', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ising.pdf', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()

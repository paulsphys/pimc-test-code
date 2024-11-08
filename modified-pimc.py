'''
A path-integral :quantum Monte Carlo program to compute the energy of the simple
harmonic oscillator in one spatial dimension.
'''
from __future__ import print_function
from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
import argparse
"""
parser = argparse.ArgumentParser()
parser.add_argument('-s','--seed',type=int,default=1173,help="The seed of the random number generator")
parser.add_argument('--beta','-T',type=float,required = True, help="The inverse temperature of the simulation")
parser.add_argument('--NumTimeSlices','-P',type=int,default=20,help="The number of beads in a wordline")
parsed_args = vars(parser.parse_args())
"""
# ------------------------------------------------------------------------------------------- 
def SHOEnergyExact(T,omega):
    '''The exact SHO energy when \hbar \omega/ k_B = 1.''' 
    return 0.5*omega/np.tanh(0.5*omega/T)

# ------------------------------------------------------------------------------------------- 
def HarmonicOscillator(R):
    '''Simple harmonic oscillator potential with m = 1 and \omega = 1.'''
    R = R
    return 0.5*np.dot(R,R);

def AnharmonicOscillator(R):
    R = R
    return np.dot(R,np.dot(R,R)) + 0.5*np.dot(np.dot(R,R),np.dot(R,R)) + (27.0/32.0)

def HarmonicApprox(R):
    R = R + 1.5
    return (9.0/4.0)*np.dot(R,R)

# ------------------------------------------------------------------------------------------- 
class Paths:
    '''The set of worldlines, action and estimators.'''
    def __init__(self,beads,tau,lam,omega,xmin):
        self.tau = tau
        self.lam = lam
        self.beads = np.copy(beads)
        self.numTimeSlices = len(beads)
        self.numParticles = len(beads[0])
        self.omega = omega
        self.xmin = xmin

    def SetPotential(self,externalPotentialFunction):
        '''The potential function. '''
        self.VextHelper = externalPotentialFunction
    
    def SetHarmonicApproximation(self, harmonicApproximation):
        self.VextHarmonic = harmonicApproximation

    def Vext(self,R):
        '''The external potential energy.'''
        return self.VextHelper(R)

    def AnharmonicPot(self,R):
        '''Given the harmonic approximation to the potential return the anharmonic part'''
        return  self.VextHelper(R) - self.VextHarmonic(R)
    
    def ParticlePosition(self):
        particle_position = np.zeros(50)
        for i in self.beads:
            for j in range(50):
                if (i >= -5 + j*(10.0/50.0)) and (i < -5 + (j+1)*(10.0/50.0)):
                    particle_position[j] += 1/0.2
        return particle_position/self.numTimeSlices


    def PotentialAction(self,tslice):
        '''The potential action.'''
        pot = 0.0
        tslicep1 = (tslice + 1) % self.numTimeSlices
        for ptcl in range(self.numParticles):
            pot += self.Vext(self.beads[tslice,ptcl])
        return self.tau*pot
    
    def ResidualActionHarmonic(self):
        '''The total action for harmonic sampling'''
        act = 0.0
        for tslice in range(self.numTimeSlices):
            tslicep1 = (tslice + 1) % self.numTimeSlices
            for ptcl in range(self.numParticles):
                delR = (self.beads[tslicep1,ptcl] - self.beads[tslice,ptcl])*np.sqrt(self.omega)
                addR = (self.beads[tslicep1,ptcl] + self.beads[tslice,ptcl] - 2*self.xmin)*np.sqrt(self.omega)
                act += np.dot(addR,addR)*np.tanh(self.tau*self.omega/2)/4 + np.dot(delR,delR)/(4 * np.tanh(self.tau*self.omega/2))
        return act

    def KineticEnergy(self):
        '''The thermodynamic kinetic energy estimator.'''
        tot = 0.0
        norm = 1.0/(4.0*self.lam*self.tau*self.tau)
        for tslice in range(self.numTimeSlices):
            tslicep1 = (tslice + 1) % self.numTimeSlices
            for ptcl in range(self.numParticles):
                delR = self.beads[tslicep1,ptcl] - self.beads[tslice,ptcl]
                tot = tot - norm*np.dot(delR,delR)
        
        KE = 0.5*self.numParticles/self.tau + tot/(self.numTimeSlices)
        return KE

    def PotentialEnergy(self):
        '''The operator potential energy estimator.'''
        PE = 0.0
        for tslice in range(self.numTimeSlices):
            tslicep1 = (tslice + 1) % self.numTimeSlices
            for ptcl in range(self.numParticles):
                R = self.beads[tslice,ptcl]
                Rp1 = self.beads[tslicep1,ptcl]
                PE = PE + self.Vext(R)
        return PE/(self.numTimeSlices)

    def Energy(self):
        '''The total energy.'''
        return self.PotentialEnergy() + self.KineticEnergy()
    
    def Energyharmonic(self):
        '''The harmonic energy estimator'''
        taupr = self.tau*self.omega
        Energy_harmonic = 0
        for tslice in range(self.numTimeSlices):
            tslicep1 = (tslice + 1) % self.numTimeSlices
            for ptcl in range(self.numParticles):
                delR = (self.beads[tslicep1,ptcl] - self.beads[tslice,ptcl])*np.sqrt((self.omega))
                addR = (self.beads[tslicep1,ptcl] + self.beads[tslice,ptcl] - 2*self.xmin)*np.sqrt(self.omega)
                Energy_harmonic = Energy_harmonic - self.omega*np.dot(delR,delR)/(8*(np.sinh(taupr/2))**2) + self.omega*np.dot(addR,addR)/(8*(np.cosh(taupr/2))**2)
        
        Energy_harmonic = self.omega/(2*np.tanh(taupr)) + Energy_harmonic/self.numTimeSlices
        return Energy_harmonic
       
    def AnharmonicEnergy(self):
        AnhE = 0.0
        for tslice in range(self.numTimeSlices):
            tslicep1 = (tslice + 1) % self.numTimeSlices
            for ptcl in range(self.numParticles):
                R = self.beads[tslice,ptcl]
                Rp1 = self.beads[tslicep1,ptcl]
                AnhE = AnhE + self.AnharmonicPot(R)
        return AnhE/(self.numTimeSlices)

    def AnharmonicAction(self,tslice):
        '''The anharmonic action.'''
        anhpot = 0.0
        tslicep1 = (tslice + 1) % self.numTimeSlices
        for ptcl in range(self.numParticles):
            #anhpot += self.AnharmonicPot(self.beads[tslice,ptcl])/2 + self.AnharmonicPot(self.beads[tslicep1,ptcl])/2
            anhpot += self.AnharmonicPot(self.beads[tslice,ptcl])
        return self.tau*anhpot
    
    def TotalEnergyharmonic(self):
        return self.Energyharmonic() + self.AnharmonicEnergy()

# ------------------------------------------------------------------------------------------- 
def PIMC(numSteps,Path):
    '''Perform a path integral Monte Carlo simulation of length numSteps.'''
    observableSkip = 1
    equilSkip = 1000
    numAccept = {'CenterOfMass':0,'Staging':0}
    EnergyTrace = []
    particle_position = []
    EnergyharmonicTrace = []
    Energyharmonic2Trace = []
    for steps in range(0,numSteps): 
        # for each particle try a center-of-mass move
        #for ptcl in np.random.randint(0,Path.numParticles,Path.numParticles):
        #    numAccept['CenterOfMass'] += CenterOfMassMove(Path,ptcl)
        # for each particle try a center-of-mass move (in harmonic sampling)
        for ptcl in np.random.randint(0,Path.numParticles,Path.numParticles):
           numAccept['CenterOfMass'] += CenterOfMassMoveHarmonic(Path,ptcl)
        # for each particle try a staging move
        for ptcl in np.random.randint(0,Path.numParticles,Path.numParticles): 
            numAccept['Staging'] += StagingMoveHarmonic(Path,ptcl)
        #for ptcl in np.random.randint(0,Path.numParticles,Path.numParticles):
        #    numAccept['Staging'] += StagingMove(Path,ptcl)
        # measure the energy
        if steps % observableSkip == 0 and steps > equilSkip:
            EnergyTrace.append(Path.Energy())
            #particle_position.append(Path.ParticlePosition())   
            EnergyharmonicTrace.append(Path.TotalEnergyharmonic())
            
    print('Acceptance Ratios:')
    print('Center of Mass: %4.3f' %
          ((1.0*numAccept['CenterOfMass'])/(numSteps*Path.numParticles)))
    print('Staging:        %1.3f\n' %
          ((1.0*numAccept['Staging'])/(numSteps*Path.numParticles)))
    #return np.array(EnergyTrace), np.array(particle_position)
    return np.array(EnergyTrace), np.array(EnergyharmonicTrace)

# ------------------------------------------------------------------------------------------- 
def CenterOfMassMove(Path,ptcl):
    '''Attempts a center of mass update, displacing an entire particle
    worldline.'''
    delta = 0.5
    shift = delta*(-1.0 + 2.0*np.random.random())

    # Store the positions on the worldline
    oldbeads = np.copy(Path.beads[:,ptcl])

    # Calculate the potential action
    oldAction = 0.0
    for tslice in range(Path.numTimeSlices):
        oldAction += Path.PotentialAction(tslice)

    # Displace the worldline
    for tslice in range(Path.numTimeSlices):
        Path.beads[tslice,ptcl] = oldbeads[tslice] + shift

    # Compute the new action
    newAction = 0.0
    for tslice in range(Path.numTimeSlices):
        newAction += Path.PotentialAction(tslice)

    # Accept the move, or reject and restore the bead positions
    if np.random.random() < np.exp(-(newAction - oldAction)):
        return True
    else:
        Path.beads[:,ptcl] = np.copy(oldbeads)
        return False

# ------------------------------------------------------------------------------------------- 
def CenterOfMassMoveHarmonic(Path,ptcl):
    '''Attempts a center of mass update, displacing an entire particle
    worldline.'''
    delta = 0.5
    shift = delta*(-1.0 + 2.0*np.random.random())

    # Store the positions on the worldline
    oldbeads = np.copy(Path.beads[:,ptcl])

    # Calculate the action
    oldAction = Path.ResidualActionHarmonic()
    for tslice in range(Path.numTimeSlices):
        oldAction += Path.AnharmonicAction(tslice)
    
    # Displace the worldline
    for tslice in range(Path.numTimeSlices):
        Path.beads[tslice,ptcl] = oldbeads[tslice] + shift

    # Compute the new action
    newAction = Path.ResidualActionHarmonic()
    for tslice in range(Path.numTimeSlices):
        newAction += Path.AnharmonicAction(tslice)

    # Accept the move, or reject and restore the bead positions
    if np.random.random() < np.exp(-(newAction - oldAction)):
        return True
    else:
        Path.beads[:,ptcl] = np.copy(oldbeads)
        return False
# -------------------------------------------------------------------------------------------
def StagingMove(Path,ptcl):
    '''Attempts a staging move, which exactly samples the free particle
    propagator between two positions.

    See: http://link.aps.org/doi/10.1103/PhysRevB.31.4234
    
    Note: does not work for periodic boundary conditions.
    '''

    # the length of the stage, must be less than numTimeSlices
    m = 16

    # Choose the start and end of the stage
    alpha_start = np.random.randint(0,Path.numTimeSlices)
    alpha_end = (alpha_start + m) % Path.numTimeSlices

    # Record the positions of the beads to be updated and store the action
    oldbeads = np.zeros(m-1)
    oldAction = 0.0
    for a in range(1,m):
        tslice = (alpha_start + a) % Path.numTimeSlices
        oldbeads[a-1] = Path.beads[tslice,ptcl]
        oldAction += Path.PotentialAction(tslice)
    """
    for a in range(1,m):
        tslice = (alpha_start + a) % Path.numTimeSlices
        tslicem1 = (tslice - 1) % Path.numTimeSlices
        tau1 = (m-a)*Path.tau*Path.omega
        gamma1 = 1/(np.tanh(Path.tau*Path.omega)) + 1/(np.tanh(tau1))
        gamma2 = (Path.beads[tslicem1,ptcl]/np.sinh(Path.tau*Path.omega) + Path.beads[alpha_end,ptcl]/np.sinh(tau1))*np.sqrt(Path.omega)
        avex = gamma2/gamma1
        sigma2 = 2.0*Path.lam / gamma1
        Path.beads[tslice,ptcl] = avex + np.sqrt(sigma2)*np.random.randn()
    return True

    """
    # Generate new positions and accumulate the new action
    newAction = 0.0;
    for a in range(1,m):
        tslice = (alpha_start + a) % Path.numTimeSlices
        tslicem1 = (tslice - 1) % Path.numTimeSlices
        tau1 = (m-a)*Path.tau
        avex = (tau1*Path.beads[tslicem1,ptcl] +
                Path.tau*Path.beads[alpha_end,ptcl]) / (Path.tau + tau1)
        sigma2 = 2.0*Path.lam / (1.0 / Path.tau + 1.0 / tau1)
        Path.beads[tslice,ptcl] = avex + np.sqrt(sigma2)*np.random.randn()
        newAction += Path.PotentialAction(tslice)

    # Perform the Metropolis step, if we rejct, revert the worldline
    if np.random.random() < np.exp(-(newAction - oldAction)):
        return True
    else:
        for a in range(1,m):
            tslice = (alpha_start + a) % Path.numTimeSlices
            Path.beads[tslice,ptcl] = oldbeads[a-1]
        return False        
        
# -------------------------------------------------------------------------------------------
def StagingMoveHarmonic(Path,ptcl):
    '''Attempts a staging move, which exactly samples the free particle
    propagator between two positions.

    See: http://link.aps.org/doi/10.1103/PhysRevB.31.4234
    
    Note: does not work for periodic boundary conditions.
    '''

    # the length of the stage, must be less than numTimeSlices
    m = int(0.8*Path.numTimeSlices)
    #m = 16
    # Choose the start and end of the stage
    alpha_start = np.random.randint(0,Path.numTimeSlices)
    alpha_end = (alpha_start + m) % Path.numTimeSlices

    # Record the positions of the beads to be updated and store the action
    oldbeads = np.zeros(m-1)
    #oldAction = Path.ResidualActionHarmonic() 
    oldAction = 0
    for a in range(1,m):
        tslice = (alpha_start + a) % Path.numTimeSlices
        oldbeads[a-1] = Path.beads[tslice,ptcl]
        oldAction += Path.AnharmonicAction(tslice)
    
    newAction = 0.0 
    # Harmonic Path sampling 
    for a in range(1,m):
        tslice = (alpha_start + a) % Path.numTimeSlices
        tslicem1 = (tslice - 1) % Path.numTimeSlices
        tau1 = (m-a)*Path.tau*Path.omega
        gamma1 = (1/(np.tanh(Path.tau*Path.omega)) + 1/(np.tanh(tau1)))*Path.omega
        gamma2 = ((Path.beads[tslicem1,ptcl])/np.sinh(Path.tau*Path.omega) + (Path.beads[alpha_end,ptcl])/np.sinh(tau1) + Path.xmin*(np.tanh(Path.tau*Path.omega/2) + np.tanh(tau1/2)))*Path.omega
        avex = gamma2/gamma1
        sigma2 = 2.0*Path.lam / gamma1 
        Path.beads[tslice,ptcl] = avex + np.sqrt(sigma2)*np.random.randn()
        newAction += Path.AnharmonicAction(tslice)
    #newAction += Path.ResidualActionHarmonic()
    
    #return True
    
    # Perform the Metropolis step, if we rejct, revert the worldline
    if np.random.random() < np.exp(-(newAction - oldAction)):
        return True
    else:
        for a in range(1,m):
            tslice = (alpha_start + a) % Path.numTimeSlices
            Path.beads[tslice,ptcl] = oldbeads[a-1]
        return False
    
# -------------------------------------------------------------------------------------------
def main(beta,seed,numTimeSlices):
    T = 1.00/beta  # temperature in Kelvin  
    lam = 0.5 # \hbar^2/2m k_B
    omega = np.sqrt(9.0/2.0)
    xmin = -1.5

    numParticles = 1    
    #numTimeSlices = 20
    numMCSteps = 100000
    tau = 1.0/(T*numTimeSlices)
    binSize = 1

    print('Simulation Parameters:')
    print('N      = %d' % numParticles)
    print('tau    = %6.4f' % tau)
    print('lambda = %6.4f' % lam)
    print('T      = %4.2f\n' % T)

    # fix the random seed
    np.random.seed(seed)

    # initialize main data structure
    beads = np.zeros([numTimeSlices,numParticles])

    # random initial positions (classical state) 
    for tslice in range(numTimeSlices):
        for ptcl in range(numParticles):
            beads[tslice,ptcl] = 0.5*(-1.0 + 2.0*np.random.random())

    # setup the paths
    Path = Paths(beads,tau,lam,omega,xmin)
    Path.SetPotential(AnharmonicOscillator)
    Path.SetHarmonicApproximation(HarmonicApprox)

    # compute the energy via path-integral Monte Carlo
    #Energy, linear_density_rho = PIMC(numMCSteps,Path)
    Energy,Energyharmonic = PIMC(numMCSteps,Path)
    #np.savetxt("data2/harmonic_newes_%.6f_%.6f.dat"%(beta,seed),Energyharmonic)
    
    # Do some simple binning statistics
    numBins = int(1.0*len(Energy)/binSize)
    slices = np.linspace(0, len(Energy),numBins+1,dtype=int)
    binnedEnergy = np.add.reduceat(Energy, slices[:-1]) / np.diff(slices)
    binnedEnergyH = np.add.reduceat(Energyharmonic, slices[:-1]) / np.diff(slices)
    
    """
    #Average_the_particle_position
    linear_density_rho_error = np.std(linear_density_rho,axis=0,ddof=1) 
    linear_density_rho = np.mean(linear_density_rho,axis=0)
    print(linear_density_rho_error)
    #Calculate chisquared
    x = np.arange(-4.9,5.0,0.2)
    exact = np.exp(-(x**2)*np.tanh(beta/2))/np.sqrt(np.pi/np.tanh(beta/2))
    chisquared = np.sum((exact - linear_density_rho)**2)
    print((exact - linear_density_rho))
    print(np.multiply((exact - linear_density_rho),linear_density_rho_error))
    chisquared_error = np.sum(2*np.abs(np.multiply((exact - linear_density_rho),linear_density_rho_error)))
    print(chisquared)
    print(chisquared_error)
    plt.plot(x,linear_density_rho,'r*',label='MC')
    plt.plot(x,exact,label='exact')
    plt.xlabel('r')
    plt.ylabel('Particle density (1/Å)')
    plt.legend()
    plt.show()
    """
    CalcEH = np.mean(binnedEnergyH)
    CalcEHError = np.std(binnedEnergyH)/np.sqrt(numBins-1)
    CalcE = np.mean(binnedEnergy)
    CalcEError = np.std(binnedEnergy)/np.sqrt(numBins-1)
    # output the final result
    print('Harmonic estimator = %8.4f +/- %6.4f' %(CalcEH, CalcEHError)) 
    print('Energy = %8.4f +/- %6.4f' % (CalcE,CalcEError))
    print('Eexact = %8.4f' % SHOEnergyExact(T,1)) 
    #with open("data4/harmonic-tau-scaling_%d_%d.txt"%(seed,numTimeSlices),'a') as f:
    #    f.write("%7.4f %7.4f %7.4e %7.4e %7.4e\n" %(tau, CalcE, CalcEError, chisquared, chisquared_error))
    with open("data-anharmonic/harmonic-com-tau-scaling_%d_%d.txt"%(seed,numTimeSlices),'w') as f:
        f.write("%7.4f %7.4f %7.4e \n" %(tau, CalcEH, CalcEHError))
# ----------------------------------------------------------------------
if __name__ == "__main__": 
    #for beta in [2.0,4.0,8.0,16.0,32.0]:
    #    for seed in list(range(80)):
    #        main(beta,seed,50)
    ncore = 80
    init = 1
    seeds = list(range(init,init+ncore))
    for i in [5,10,20,40,50]:
        pool = Pool(processes=ncore)
        args_list = [(1, seed, i) for seed in seeds]
        pool.starmap(main, args_list)
        pool.close()
        pool.join()
    #main(parsed_args["beta"],parsed_args["seed"],parsed_args["NumTimeSlices"])
    #main(4,20,5)

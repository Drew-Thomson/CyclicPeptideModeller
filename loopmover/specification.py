import copy
import math
import random
import sys

import ampal
from ampal.geometry import distance
from isambard.specifications.ta_polypeptide import TAPolypeptide


def calc_rmsd(frag1, frag2):
    """Returns the maximum distance between a pair of atoms in equivalent sets."""
    frag1_atoms = frag1.backbone.get_atoms()
    frag2_atoms = frag2.backbone.get_atoms()
    return (0.25 * sum([distance(x, y) for x, y in zip(frag1_atoms, frag2_atoms)]))**0.5

def rand_mac(res, max_iter=20000, max_attempts=5, max_rmsd=0.05):
    """Builds a macrocycle by starting with random phi and psi angles, then mutating them till the ends overlap."""
    
    def get_best_initial(res_count, samples=100):
        best_angles = None
        best_rmsd_val = float('inf')
        for _ in range(samples):
            angles = [[180, random.uniform(-180, 180), random.uniform(-180, 180)] for _ in range(res_count)]
            test_model = TAPolypeptide([angles[-1]] + angles + angles[0:2])
            rmsd_val = calc_rmsd(test_model[1], test_model[-2])
            if rmsd_val < best_rmsd_val:
                best_rmsd_val = rmsd_val
                best_angles = angles
        return best_angles, best_rmsd_val

    best_angles, best_rmsd = get_best_initial(res)
    print(f"Starting rmsd is {best_rmsd}")
    
    cached_tries = []
    attempts = 0
    i = 0
    
    while best_rmsd > max_rmsd and i < max_iter:
        working_angles = copy.deepcopy(best_angles)
        
        # Flipped Adaptive mutation strategy: more angles perturbed
        if best_rmsd < 1.5:
            num_mutations = random.randint(3, max(6, res))
        else:
            num_mutations = random.randint(2, max(4, int(res * 0.75)))
            
        for j in range(num_mutations):
            working_angles[random.choice(range(len(best_angles)))][random.choice(range(1,3))] += random.gauss(0, 0.5*best_rmsd)

        test_model = TAPolypeptide([working_angles[-1]] + working_angles + working_angles[0:2])
        new_rmsd = calc_rmsd(test_model[1], test_model[-2])
        
        # Metropolis criterion for simulated annealing (exponential cooling)
        temp = max(0.001, 0.1 * (0.9995 ** i))
        accept = False
        if new_rmsd < best_rmsd:
            accept = True
        else:
            prob = math.exp(-(new_rmsd - best_rmsd) / temp)
            if random.random() < prob:
                accept = True
                
        if accept:
            best_angles = copy.deepcopy(working_angles)
            best_rmsd = new_rmsd
            
        if not(i % 100):
            sys.stdout.write(f"\rAt iter {i} best rmsd is {best_rmsd})")
            sys.stdout.flush()
            
        i += 1
        if i == max_iter:
            print(f'\nManaged only rmsd of {best_rmsd}: resampling!')
            cached_tries.append((best_angles, best_rmsd))
            if attempts == max_attempts:
                cached_tries.sort(key=lambda x: x[1])
                print(f"Ran out of attempts, best rmsd was {cached_tries[0][1]}")
                return(cached_tries[0][0])
            
            best_angles, best_rmsd = get_best_initial(res)
            attempts += 1
            i = 0

    print(f"\nAfter {i} iterations and {attempts} attempts best rmsd is {best_rmsd}")
    return(best_angles)
    
def build_mac(angles):
    """Builds a cyclic peptide in isambard from a list of phi, psi angles."""
    mac = TAPolypeptide([angles[-1]] + angles + [angles[0]])
    mac.tag_torsion_angles()
    actualmac = mac[1:-1]
    actualmac.tags['cyclic'] = True
    return ampal.assembly.Assembly(actualmac)

class CyclicPeptide(ampal.Assembly):
    """Models a cyclic peptide."""

    def __init__(self, sequence, angles=None, auto_build=True):
        super().__init__()
        self.sequence = sequence
        self.angles = angles
        if auto_build:
            self.build()

    def build(self):
        """Builds the cyclic peptide using random macrocycle generator or provided angles."""
        if self.angles is None:
            self.angles = rand_mac(len(self.sequence))
        self.actualmac = build_mac(self.angles)
        self._molecules = list(self.actualmac)
        self.tags['cyclic'] = True  # Tag the assembly
        self.relabel_all()
        for m in self._molecules:
            m.ampal_parent = self

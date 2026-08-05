"""
Principle 26-30, 37: Structure vs. Chaos Equilibrium Balancer
Enforces 1:1 balance between bound structural mutations and the unattached chaos pool.
"""

from typing import Set, Dict, Any
from computational_consciousness_engine.config import rng


class ChaosStructureBalancer:
    """
    Chaos & Structure Equilibrium System.

    Invariants:
    1. Structure (bound strand mutations) and Chaos (unattached mutation pool) are capped 1:1.
    2. Chaos acts as the raw material comparator for structural evolution (Principle 26).
    3. Unattached mutations pool into chaos, increasing potential for future mutation attraction (Principle 37).
    4. Generational death releases strand-bound mutations back into chaos (Principle 24: death/rebirth balance).
    """
    def __init__(self, initial_pool_size: int = 500):
        # The chaos pool is no longer a finite list of static integers.
        # It is a thermodynamic temperature that dictates the variance of the stochastic field.
        self.chaos_temperature: float = float(initial_pool_size)
        self.entropy_decay_rate: float = 0.01  # Thermodynamic decay per emission
        self.base_temperature: float = 10.0

    def emit_unattached_mutation(self) -> int:
        """Pulls a stochastic mutation from the Gaussian chaos field."""
        # Thermodynamic decay: the field cools slightly as it emits structure
        self.chaos_temperature = max(self.base_temperature, self.chaos_temperature * (1.0 - self.entropy_decay_rate))
        
        # Sample an integer from the Gaussian field defined by the current temperature
        mutation = int(rng.normal(loc=10000, scale=self.chaos_temperature))
        return mutation

    def return_to_chaos(self, mutation: int):
        """Returns a discarded mutation, adding friction/heat back to the field."""
        self.chaos_temperature += 0.5

    def recycle_dying_strand(self, dying_mutations: list):
        """
        Principle 24: When a strand dies at -1, its raw structural heat
        is released back into the chaos field, spiking the temperature.
        """
        self.chaos_temperature += len(dying_mutations) * 0.5

    def evaluate_equilibrium_state(self, bound_structure_count: int) -> Dict[str, Any]:
        """
        Principle 28 & 29: Evaluates the Structure / Chaos ratio.
        Structure = Bound Count, Chaos = Thermodynamic Temperature.
        """
        chaos_count = self.chaos_temperature

        if chaos_count <= self.base_temperature and bound_structure_count == 0:
            ratio = 1.0
        elif chaos_count <= self.base_temperature:
            ratio = float('inf')
        else:
            ratio = bound_structure_count / chaos_count

        # Check for imbalance (Principle 27: Balloon breaking point / vacuum collapse)
        is_balanced = 0.5 <= ratio <= 2.0
        if ratio > 2.0:
            status = "STRUCTURAL_DOMINANCE"
        elif ratio < 0.5:
            status = "CHAOS_OVERPOWERING"
        else:
            status = "EQUILIBRIUM_STABLE"

        return {
            "bound_structure_count": bound_structure_count,
            "chaos_pool_count": chaos_count,
            "structure_chaos_ratio": round(ratio, 4),
            "is_balanced": is_balanced,
            "status": status
        }

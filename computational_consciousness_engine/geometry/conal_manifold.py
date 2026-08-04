import jax.numpy as jnp
from jax import jit
import numpy as np
from typing import Dict, Any

class ConalManifold:
    def __init__(self, cone_height: float = 1.0, max_radius: float = 5.0):
        self.height = cone_height
        self.max_radius = max_radius

    @staticmethod
    @jit
    def _compute_metrics_jax(progress, max_radius, height):
        """
        JIT-compiled XLA function. Runs orders of magnitude faster 
        when computing thousands of geometric vectors simultaneously.
        """
        radius = max_radius * jnp.sin(jnp.pi * progress)
        z = height * progress
        surface_area = 2 * jnp.pi * radius * jnp.sqrt(radius**2 + height**2)
        
        # Unfolded state metric (Avoid division by zero)
        unfolded_degree = jnp.where(max_radius > 0, radius / max_radius, 0.0)
        is_fully_unfolded = unfolded_degree > 0.95
        
        return z, radius, surface_area, unfolded_degree, is_fully_unfolded

    def compute_conal_metric(self, progress: float) -> Dict[str, float]:
        # Execute the JIT compiled mathematics
        z, radius, surface_area, unfolded_degree, is_fully_unfolded = self._compute_metrics_jax(
            jnp.array(progress, dtype=jnp.float32), 
            self.max_radius, 
            self.height
        )
        
        return {
            "progress": float(progress),
            "z": float(z),
            "radius": float(radius),
            "surface_area": float(surface_area),
            "unfolded_degree": float(unfolded_degree),
            "is_fully_unfolded": bool(is_fully_unfolded)
        }
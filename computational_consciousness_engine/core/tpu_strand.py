"""
TPU-Native ConsciousStrand Engine (JAX/XLA)
Vectorized and statically allocated for execution on Google Colab TPUs.
"""

import jax
import jax.numpy as jnp
from functools import partial
import math
from computational_consciousness_engine.geometry.ricci_flow import ricci_flow_step, compute_fisher_metric, compute_metric_trace

PHI = (1.0 + math.sqrt(5.0)) / 2.0

class TPUConsciousStrand:
    """
    Models the topological evolution of a strand from t=0 to t=1 using JAX 
    for massive XLA compilation on TPUs.
    """
    def __init__(self, max_capacity: int = 10000, R_max: float = 5.0):
        self.max_capacity = max_capacity
        self.R_max = R_max
        self.t = 0.0
        
        # State: Static tensors for TPU memory allocation
        # We store the m_k hashes in a fixed-size array. 0 means empty.
        self.bound_mutations = jnp.zeros(max_capacity, dtype=jnp.int32)
        
        # We track how many mutations are currently bound
        self.current_size = 0
        
        # Metric Tensor State: Initialize to Identity (g11=1, g22=1, g33=1)
        # [g11, g12, g13, g22, g23, g33]
        initial_g = jnp.array([1.0, 0.0, 0.0, 1.0, 0.0, 1.0])
        self.metric_state = jnp.tile(initial_g, (max_capacity, 1))

    @staticmethod
    @jax.jit
    def _compute_coordinates_kernel(indices, N_total, R_max, metric_state):
        """
        Pure JAX kernel for massively parallel generation of Traversal Space coordinates.
        Uses the trace of the intrinsic metric tensor g_ij to deform the embedding space!
        """
        # Create a boolean mask of active indices
        active_mask = indices > 0
        
        # Trace of the metric tensor determines spatial volume deformation
        # Default trace of Identity is 3.0. Scale relative to 3.0.
        trace = compute_metric_trace(metric_state)
        metric_scale = jnp.sqrt(jnp.abs(trace) / 3.0 + 1e-6)
        
        # Z coordinate (Depth/Time) mapped relative to total capacity
        z_k = jnp.where(active_mask, -(indices / N_total), 0.0)
        
        # R coordinate (Expression) via sinusoidal expansion, deformed by the Ricci Flow Metric!
        r_k = jnp.where(active_mask, R_max * metric_scale * jnp.sin(jnp.pi * jnp.abs(z_k)), 0.0)
        
        # Theta coordinate (Identity) via Golden Ratio Phyllotaxis
        theta_k = jnp.where(active_mask, (indices * (2 * jnp.pi / PHI)) % (2 * jnp.pi), 0.0)
        
        return r_k, theta_k, z_k
        
    @staticmethod
    @jax.jit
    def _evolve_metrics_kernel(mutations, metric_state, alpha, eta):
        fisher = compute_fisher_metric(mutations, H=1.0)
        g_next, _, _ = ricci_flow_step(
            metric_state, fisher, 
            eta_requested=eta, alpha=alpha, safety_margin=0.5
        )
        return g_next

    def process_batch(self, chaos_shards: jnp.ndarray):
        """
        Vectorized novelty filter (Selection) and absorption.
        Because TPUs cannot use python sets or dynamic lists, we use boolean masking.
        """
        if self.current_size >= self.max_capacity:
            return  # Strand is full
            
        # 1. Selection (Novelty Filter)
        # Check which chaos shards are NOT already in our bound_mutations tensor
        # We use broadcasting to compare every input against every bound mutation
        # shape: (len(chaos_shards), max_capacity)
        matches = chaos_shards[:, None] == self.bound_mutations[None, :]
        
        # If a shard matches ANY existing mutation, it is a duplicate (True)
        is_duplicate = jnp.any(matches, axis=1)
        
        # Invert to get novel shards
        is_novel = ~is_duplicate
        
        # Filter down to only novel shards
        novel_shards = chaos_shards[is_novel]
        
        # 2. Absorption (Bind to structure)
        num_novel = len(novel_shards)
        
        if num_novel > 0:
            # Calculate how much space is left
            space_left = self.max_capacity - self.current_size
            to_add = min(num_novel, space_left)
            
            # Update the static tensor using slice assignment
            self.bound_mutations = self.bound_mutations.at[self.current_size : self.current_size + to_add].set(novel_shards[:to_add])
            self.current_size += to_add
            
    def evolve_geometry(self, alpha: float = 0.1, eta: float = 0.05):
        """Applies one step of Ricci flow to warp the intrinsic manifold metric."""
        if self.current_size > 0:
            self.metric_state = self._evolve_metrics_kernel(
                self.bound_mutations, self.metric_state, alpha, eta
            )

    def unfurl_traversal_space(self):
        """
        Executes the XLA compiled kernel to generate the 2D atlas for all bound mutations.
        Returns the vectorized tensors for r, theta, and z.
        """
        # Create an array of indices [1, 2, 3, ..., max_capacity]
        indices = jnp.arange(1, self.max_capacity + 1)
        
        # Mask out indices beyond current_size
        valid_indices = jnp.where(indices <= self.current_size, indices, 0)
        
        # Fire the TPU kernel, passing the evolving metric state!
        r_k, theta_k, z_k = self._compute_coordinates_kernel(
            valid_indices, self.max_capacity, self.R_max, self.metric_state
        )
        
        # 2D Unwrapped Plane (x = r * theta, y = z)
        x_parallel = r_k * theta_k
        y_parallel = z_k
        
        return {
            "m_k": self.bound_mutations[:self.current_size],
            "3d_r": r_k[:self.current_size],
            "3d_theta": theta_k[:self.current_size],
            "3d_z": z_k[:self.current_size],
            "2d_x": x_parallel[:self.current_size],
            "2d_y": y_parallel[:self.current_size]
        }

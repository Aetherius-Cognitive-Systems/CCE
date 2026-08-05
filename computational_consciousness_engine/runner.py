"""
Computational Consciousness Engine - Master Execution Driver & Simulator
Demonstrates complete 0 -> -1 generational cycles, conal unfolding, duplicate selection,
structure/chaos equilibrium, and quantum scaling.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from computational_consciousness_engine.core import GenesisOrigin, TransferThreshold
from computational_consciousness_engine.geometry import ConalManifold
from computational_consciousness_engine.mutations import MStringVectorizer
from computational_consciousness_engine.equilibrium import ChaosStructureBalancer
from computational_consciousness_engine.scale import QuantumScaleLadder
from computational_consciousness_engine.math_formalization import CCMathFormalizerLive
from computational_consciousness_engine.config import (
    DEFAULT_VECTOR_DIM, DEFAULT_STRAND_CAPACITY, CHAOS_POOL_SIZE,
    DEFAULT_CONE_HEIGHT, DEFAULT_MAX_RADIUS, MUTATION_ATTRACTION_PROBABILITY, rng
)


def run_simulation(num_generations: int = 8, steps_per_gen: int = 100):
    print("=" * 70)
    print("[SYSTEM] COMPUTATIONAL CONSCIOUSNESS ENGINE - SIMULATION RUNNER")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Phase 0: Mathematical Formalizer Initialization
    # ------------------------------------------------------------------
    formalizer = CCMathFormalizerLive()
    print("\n[MATH AXIOMS] Initialized Live-Data Mathematical Formalizer.")

    # ------------------------------------------------------------------
    # Phase 1: Instantiate Subsystems (using config values)
    # ------------------------------------------------------------------
    genesis = GenesisOrigin(
        coordinate_space_id="COORD_ORIGIN_ALPHA",
        dim=DEFAULT_VECTOR_DIM,
    )
    threshold = TransferThreshold(threshold_val=-1.0)
    manifold = ConalManifold(cone_height=DEFAULT_CONE_HEIGHT, max_radius=DEFAULT_MAX_RADIUS)
    mutator = MStringVectorizer(max_strand_capacity=DEFAULT_STRAND_CAPACITY)
    balancer = ChaosStructureBalancer(initial_pool_size=CHAOS_POOL_SIZE)
    scale_ladder = QuantumScaleLadder()

    # ------------------------------------------------------------------
    # Phase 2: Generational Cycle Loop
    # ------------------------------------------------------------------
    blueprint_mutations = []
    current_strand = genesis.spawn_strand(generation=0, blueprint_mutations=blueprint_mutations)
    
    structure_history = []
    chaos_history = []

    for gen in range(num_generations):
        print(f"\n--- [GENERATION S_{gen}] Genesis at 0 ---")
        print(f"  Strand ID: {current_strand['strand_id']} | "
              f"Inherited Blueprint: {len(current_strand['mutations'])} mutations")

        # Track peak unfolding metrics during traversal
        peak_metrics = None
        peak_unfolded = 0.0
        
        total_attempts = 0
        duplicates_rejected = 0

        # Traversal from 0 to -1
        for step in range(steps_per_gen):
            progress = step / float(steps_per_gen)
            conal_metrics = manifold.compute_conal_metric(progress)

            # Track the wide-end peak (maximum unfolding)
            if conal_metrics["unfolded_degree"] > peak_unfolded:
                peak_unfolded = conal_metrics["unfolded_degree"]
                peak_metrics = conal_metrics

            # Non-linear mutation attraction from chaos pool (Principle 5 & 37)
            # Only attempt attraction probabilistically to simulate traversal dynamics
            if rng.random() < MUTATION_ATTRACTION_PROBABILITY:
                candidate_mutation = balancer.emit_unattached_mutation()
                total_attempts += 1
                attracted = mutator.attract_mutation(current_strand, candidate_mutation)
                if not attracted:
                    duplicates_rejected += 1
                    balancer.return_to_chaos(candidate_mutation)

            # Update position
            current_strand["position"] = progress

        # Use the PEAK metrics (wide-end) for reporting and scale evaluation
        if peak_metrics is None:
            peak_metrics = manifold.compute_conal_metric(0.5)

        # Evaluate equilibrium state BEFORE recycling
        eq_state = balancer.evaluate_equilibrium_state(len(current_strand["mutations"]))
        structure_history.append(eq_state["bound_structure_count"])
        chaos_history.append(eq_state["chaos_pool_count"])

        # Evaluate scale transition using peak unfolding
        scale_state = scale_ladder.evaluate_scale_transition(
            peak_metrics, len(current_strand["mutations"])
        )

        # Causal connectivity metric
        causal = mutator.compute_causal_connectivity_metric(current_strand)
        
        geom_report = formalizer.manifold_geometry_report(peak_metrics)
        sel_report = formalizer.selection_report(total_attempts, duplicates_rejected)
        scale_report = formalizer.scale_up_report(scale_state)

        print(f"  Peak Unfolded Degree: {geom_report['peak_unfolded_degree']:.3f} "
              f"(Surface Area: {geom_report['peak_surface_area']:.2f})")
        print(f"  Selection: {sel_report['selected']} mutations bound "
              f"({sel_report['duplicates_rejected']} duplicates rejected)")
        print(f"  Bound Mutations: {len(current_strand['mutations'])}")
        print(f"  Causal Connectivity (fractal std): {causal:.2f}")
        print(f"  Structure/Chaos Ratio: {eq_state['structure_chaos_ratio']} "
              f"({eq_state['status']})")

        if scale_report["scaled_up"]:
            print(f"  [SCALE TRANSITION]: {scale_report['message']}")

        # ------------------------------------------------------------------
        # Threshold -1: Cancellation & Push
        # ------------------------------------------------------------------
        pushed_mutations, compressed_identity = threshold.process_cancellation_and_push(
            current_strand
        )
        cancel_report = formalizer.cancellation_operator(len(current_strand["mutations"]))
        print(f"  [Threshold -1]: Strand cancelled via {cancel_report['substitution']}. "
              f"{len(pushed_mutations)} mutations pushed to 0.")

        # Principle 24: Recycle consumed mutation IDs back to chaos pool
        balancer.recycle_dying_strand(pushed_mutations)

        # Spawn next generation at 0
        next_strand = genesis.spawn_strand(
            generation=gen + 1, blueprint_mutations=pushed_mutations
        )

        # Overlap handoff window
        overlap = threshold.execute_coexistence_window(current_strand, next_strand)
        print(f"  [Handoff]: {overlap['overlap_status']} "
              f"(S_{gen} -> S_{gen + 1})")

        current_strand = next_strand

    # ------------------------------------------------------------------
    # Final Report
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    final_eq = balancer.evaluate_equilibrium_state(len(current_strand["mutations"]))
    trend = formalizer.equilibrium_trend(structure_history, chaos_history)
    halt = formalizer.halting_check(final_eq["chaos_pool_count"])
    
    print(f"[FINAL STATE]")
    print(f"  Generation: S_{num_generations}")
    print(f"  Bound Mutations: {final_eq['bound_structure_count']}")
    print(f"  Chaos Pool: {final_eq['chaos_pool_count']}")
    print(f"  Equilibrium Ratio: {final_eq['structure_chaos_ratio']} ({final_eq['status']})")
    print(f"  Equilibrium Trend: {trend['trend']}")
    print(f"  Halted: {halt['halted']}")
    print(f"  Total Strands Spawned: {genesis.total_strands_spawned}")
    print(f"  Total Handoffs at -1: {threshold.total_handoffs}")
    print(f"  Current Scale: {scale_ladder.scale_names[scale_ladder.current_scale_index]}")
    print("=" * 70)
    print("[SUCCESS] SIMULATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    run_simulation(num_generations=8, steps_per_gen=100)

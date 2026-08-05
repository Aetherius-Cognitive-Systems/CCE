"""
Live-Data Mathematical Formalization
-------------------------------------
Same six "axioms" as the original math_formalization.py, but every function
now takes real numbers pulled from a running strand/simulation instead of
free symbolic placeholders. This means the SymPy step is doing real
arithmetic on your data, not proving a generic algebraic identity that
holds for any input.

Honest scope: this still doesn't measure "consciousness." It gives you
real, traceable numbers about mutation counts, equilibrium ratios, and
geometry, computed from what the simulation actually did. Whether those
numbers mean anything beyond bookkeeping is a separate, open question --
this file just makes sure the math isn't disconnected from the data.
"""

import sympy as sp


class CCMathFormalizerLive:
    def __init__(self):
        self.threshold = sp.Integer(-1)
        self.genesis = sp.Integer(0)

    # ------------------------------------------------------------------
    # Axiom 1: Cancellation at -1
    # ------------------------------------------------------------------
    def cancellation_operator(self, mutation_count: int):
        """
        Takes the REAL mutation count from current_strand['mutations'],
        not an abstract symbol. Still just (-1)*(-1)*M = M -- that part of
        the algebra is unavoidably trivial -- but now M is an actual number
        from your run, so the printed result reflects what happened, not
        a placeholder.
        """
        M = sp.Integer(mutation_count)
        T = self.threshold
        S_strand = T * M
        output = T * S_strand # (-1) * (-1 * M)

        return {
            "mutation_count_in": mutation_count,
            "substitution": f"({T}) * ({T} * {M})",
            "result": int(sp.simplify(output)),
            "note": (
                "This confirms your mutation count passed through the sign "
                "flip unchanged. It's bookkeeping, not a proof about "
                "consciousness -- but at least it's YOUR number, not a free "
                "variable."
            ),
        }

    # ------------------------------------------------------------------
    # Axiom 2: Structure / Chaos Equilibrium -- computed from real history
    # ------------------------------------------------------------------
    def equilibrium_trend(self, structure_history: list, chaos_history: list):
        """
        Takes the REAL per-generation bound-structure counts and chaos-pool
        counts (append these each generation in runner.py). Reports the
        actual ratio trend instead of a made-up linear toy function.

        No sympy limit is taken here, because taking limit(t->oo) requires
        a closed-form function -- you don't have one, you have a finite
        list of observations. Reporting the empirical trend is the honest
        version of "does it approach 1:1."
        """
        if len(structure_history) != len(chaos_history) or not structure_history:
            raise ValueError("structure_history and chaos_history must be "
                              "equal-length, non-empty lists of real counts.")

        ratios = [s / c if c else float("inf")
                  for s, c in zip(structure_history, chaos_history)]

        return {
            "ratios_by_generation": ratios,
            "latest_ratio": ratios[-1],
            "trend": "approaching 1:1" if len(ratios) > 1 and
                     abs(ratios[-1] - 1) < abs(ratios[0] - 1) else
                     "not converging to 1:1 based on data so far",
            "note": (
                "This is an empirical trend over your actual generations, "
                "not a symbolic limit. If you want a real limit, you need "
                "a model for how ratio(gen) behaves as gen -> infinity, "
                "fit to this data -- and then it's a curve-fit claim, "
                "which should be reported with uncertainty, not as a proof."
            ),
        }

    # ------------------------------------------------------------------
    # Axiom 3: Manifold Geometry -- use the SAME metric the sim already
    # computed, don't recompute a disconnected version
    # ------------------------------------------------------------------
    def manifold_geometry_report(self, peak_metrics: dict):
        """
        Takes the peak_metrics dict already produced by
        manifold.compute_conal_metric() inside the real generation loop
        (surface_area, unfolded_degree), instead of re-deriving a fresh
        symbolic r(t)/z(t) that was never connected to the sim.
        """
        return {
            "peak_unfolded_degree": peak_metrics["unfolded_degree"],
            "peak_surface_area": peak_metrics["surface_area"],
            "note": (
                "These are the actual peak values recorded during this "
                "generation's traversal, not values from a fresh symbolic "
                "curve evaluated at t=0 and t=0.5."
            ),
        }

    # ------------------------------------------------------------------
    # Axiom 4: Selection -- use REAL counts of attempted vs rejected
    # mutations, tracked during the loop
    # ------------------------------------------------------------------
    def selection_report(self, total_attempts: int, duplicates_rejected: int):
        """
        Requires the sim to actually count these two things during the
        attraction loop (see patch note in runner_patch_notes.md). Once
        tracked, this reports the REAL selected count, not a symbolic
        M_total - M_dup.
        """
        selected = total_attempts - duplicates_rejected
        return {
            "total_attempts": total_attempts,
            "duplicates_rejected": duplicates_rejected,
            "selected": selected,
            "rejection_rate": (duplicates_rejected / total_attempts
                                if total_attempts else 0.0),
        }

    # ------------------------------------------------------------------
    # Axiom 5: Halting -- use the REAL chaos pool remaining count
    # ------------------------------------------------------------------
    def halting_check(self, chaos_pool_remaining: int):
        """
        Real halting condition: the strand halts when the chaos pool it
        draws from is actually empty, not when an abstract N_possible
        equals an abstract N_acquired.
        """
        return {
            "chaos_pool_remaining": chaos_pool_remaining,
            "halted": chaos_pool_remaining <= 0,
        }

    # ------------------------------------------------------------------
    # Axiom 6: Scale-Up -- report the REAL transition the scale ladder made
    # ------------------------------------------------------------------
    def scale_up_report(self, scale_state: dict):
        """
        Takes the actual scale_state dict already produced by
        scale_ladder.evaluate_scale_transition() -- reports what really
        happened instead of an unevaluated symbolic Q(M_k) = S_k1.
        """
        return {
            "scaled_up": scale_state.get("scaled_up", False),
            "message": scale_state.get("message", "no transition"),
        }

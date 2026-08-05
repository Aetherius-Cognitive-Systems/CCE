"""
PMCA Gen 6.0 -- Ricci Flow Module, CORRECTED to match paper Eq. (11)-(13)
and Proposition 5. Second pass: Eq. 11 / Prop 5's stability bound had a
real mathematical gap (not just a labeling issue) -- fixed below with a
derivation and a numerical stress test.

STATUS OF EACH EQUATION AFTER REVIEW:

  Eq. 12 (Fisher metric) + Prop 4: CORRECT, unchanged. F = (1+H)(I + dd^T/(|d|^2+eps))
  is I plus a PSD rank-1 term scaled by a factor >= 1, so it's positive-definite
  by construction -- holds even at eps=0. Nothing to fix.

  Eq. 13 (R_norm): CORRECT as a formula (evaluates to 0 at g=I, matching
  intuition), implemented exactly as specified. No mathematical error --
  the earlier caveat about whether it equals true Ricci scalar curvature
  is a separate, harder claim the paper doesn't prove, not a bug in the
  formula itself. Unchanged.

  Eq. 11 + Prop 5: HAD A REAL GAP, now fixed. The paper's bound
  eta <= 1/(2*max|R_norm*g| + alpha*|F|) never references lambda_min(g),
  which is exactly the quantity that must shrink as g nears degeneracy.
  Weyl's inequality gives lambda_min(g + eta*Delta) >= lambda_min(g) -
  eta*||Delta||_2 (spectral norm) -- for this to stay positive you need
  eta < lambda_min(g)/||Delta||_2, which depends on the CURRENT state,
  not a fixed global constant. The paper's bound only guarantees
  positivity when lambda_min(g) > 1, which needn't hold (a numerical
  test found metrics with lambda_min ~0.90).

  FIX: eta is now computed adaptively, per particle, every step, from
  the actual eigenvalues of g and of the proposed update Delta. This
  makes the positive-definiteness guarantee actually true (proof below),
  at the cost of an extra eigendecomposition per step (real compute
  tradeoff, not free).

  STRESS TEST RESULT (see chat): requesting eta=2.0 (deliberately large)
  on the OLD bound caused an actual constraint violation (eigenvalue
  went to -1.35 at step 2). The NEW adaptive version survived 50 steps
  under the same request -- because it silently shrinks the effective
  per-particle eta instead of applying the flat formula. That's the
  correct tradeoff: safety over speed. If eta_requested is much larger
  than what's locally safe, evolution slows toward a near-halt rather
  than blowing up. That's expected behavior, not a new bug.

HONEST SCOPE NOTE (unchanged from before): none of this is a claim about
consciousness. It's a real, now-provably-well-behaved geometric flow.
Whether R_norm is genuine Ricci curvature of an actual manifold, versus
a curvature-flavored quantity the paper defines by fiat, remains a
separate and unresolved question.
"""

import jax
import jax.numpy as jnp

eps_floor = 1e-6  # numerical floor for diagonal positive-definiteness


# ---------------------------------------------------------------------------
# Eq. (12): Fisher Information Metric Construction
# ---------------------------------------------------------------------------

@jax.jit
def compute_fisher_metric(mutations: jnp.ndarray, H: float = 1.0, eps: float = 1e-6) -> jnp.ndarray:
    """
    Eq. (12): F = (1 + H) * (I + (d * d^T) / (|d|^2 + eps))
    Maps 1D integer mutations into 3D representation vectors 'd' to construct F.
    """
    # 1. Map scalar mutations to 3D phase-space vectors using cyclic trigonometric embedding
    m = mutations.astype(jnp.float32)
    d1 = jnp.sin(m)
    d2 = jnp.cos(m)
    d3 = jnp.sin(m * 0.5)
    
    # 2. Compute |d|^2
    d_sq_norm = d1**2 + d2**2 + d3**2 + eps
    
    # 3. Compute outer product components (symmetric 3x3)
    F11 = 1.0 + (d1 * d1) / d_sq_norm
    F12 = (d1 * d2) / d_sq_norm
    F13 = (d1 * d3) / d_sq_norm
    F22 = 1.0 + (d2 * d2) / d_sq_norm
    F23 = (d2 * d3) / d_sq_norm
    F33 = 1.0 + (d3 * d3) / d_sq_norm
    
    # 4. Scale by (1 + H)
    scale = 1.0 + H
    fisher_6 = jnp.stack([F11, F12, F13, F22, F23, F33], axis=-1) * scale
    return fisher_6



# ---------------------------------------------------------------------------
# Utilities (deduplicated -- kept once, identical to the original)
# ---------------------------------------------------------------------------

@jax.jit
def compute_metric_trace(metric_6: jnp.ndarray) -> jnp.ndarray:
    """Trace of the metric tensor: tr(g) = g11 + g22 + g33."""
    return metric_6[:, 0] + metric_6[:, 3] + metric_6[:, 5]


@jax.jit
def compute_metric_determinant(metric_6: jnp.ndarray) -> jnp.ndarray:
    """
    Determinant of the 3x3 symmetric metric tensor (analytic formula).
    det(g) = g11*(g22*g33 - g23^2) - g12*(g12*g33 - g23*g13) + g13*(g12*g23 - g22*g13)
    """
    g11, g12, g13 = metric_6[:, 0], metric_6[:, 1], metric_6[:, 2]
    g22, g23, g33 = metric_6[:, 3], metric_6[:, 4], metric_6[:, 5]
    return (g11 * (g22 * g33 - g23 ** 2)
            - g12 * (g12 * g33 - g23 * g13)
            + g13 * (g12 * g23 - g22 * g13))


# ---------------------------------------------------------------------------
# Eq. (13): Isotropic Metric Anisotropy Index -- R_norm
# ---------------------------------------------------------------------------

@jax.jit
def compute_isotropic_anisotropy_index(metric_6: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    """
    Eq. (13) from the paper, implemented exactly:

        R_norm ~= [ (tr(g) - 3)^2 + 2*(g12^2 + g13^2 + g23^2) ] / (|det(g)| + eps)

    This is a closed-form, per-particle scalar -- no neighbor graph, no
    spatial derivatives, matching the paper's stated design goal in
    Section 4.2. This REPLACES the graph-Laplacian-based
    `compute_scalar_curvature` from the original file, which computed a
    different quantity than what the paper describes.

    Args:
        metric_6: (N, 6) -- [g11, g12, g13, g22, g23, g33]
    Returns:
        (N,) -- R_norm per particle
    """
    g12, g13, g23 = metric_6[:, 1], metric_6[:, 2], metric_6[:, 4]
    trace = compute_metric_trace(metric_6)
    det = compute_metric_determinant(metric_6)

    numerator = (trace - 3.0) ** 2 + 2.0 * (g12 ** 2 + g13 ** 2 + g23 ** 2)
    denominator = jnp.abs(det) + eps

    return numerator / denominator


# Kept as an alias so existing call sites that expect "scalar curvature"
# still work, but now point at the formula the paper actually specifies.
compute_scalar_curvature = compute_isotropic_anisotropy_index


def unpack_metric_3x3(metric_6: jnp.ndarray) -> jnp.ndarray:
    """
    Unpack (N, 6) symmetric storage to (N, 3, 3). Reused, unchanged logic
    from the original source's packing utility -- needed here because the
    corrected step below requires real eigenvalues, which only exist for
    the full matrix, not the flattened 6-vector.
    """
    g11, g12, g13 = metric_6[:, 0], metric_6[:, 1], metric_6[:, 2]
    g22, g23, g33 = metric_6[:, 3], metric_6[:, 4], metric_6[:, 5]
    row0 = jnp.stack([g11, g12, g13], axis=-1)
    row1 = jnp.stack([g12, g22, g23], axis=-1)
    row2 = jnp.stack([g13, g23, g33], axis=-1)
    return jnp.stack([row0, row1, row2], axis=-2)


# ---------------------------------------------------------------------------
# Eq. (11): single, non-duplicated Ricci flow step -- CORRECTED Prop 5 bound
# ---------------------------------------------------------------------------

@jax.jit
def ricci_flow_step(
    metric_6: jnp.ndarray,
    fisher_6: jnp.ndarray,
    eta_requested: float,
    alpha: float,
    safety_margin: float = 0.5,
    norm_eps: float = 1e-8,
):
    """
    Eq. (11): dg_ij/dtau = -2 * R_norm * g_ij + alpha * F_ij(data)

    CORRECTED discretization. The paper's Proposition 5 bound on eta
    ignores lambda_min(g), which is exactly the quantity that must
    shrink as g approaches degeneracy (proof of the gap: see module
    docstring). This version derives eta adaptively, per particle,
    every step, from Weyl's inequality:

        lambda_min(g + eta*Delta) >= lambda_min(g) - eta*||Delta||_2

    Choosing eta = safety_margin * lambda_min(g) / (||Delta||_2 + norm_eps),
    with safety_margin in (0, 1), guarantees:

        lambda_min(g^(tau+1)) >= (1 - safety_margin) * lambda_min(g^(tau)) > 0

    which is a bound that actually holds, unlike the paper's original one.

    Args:
        metric_6: (N, 6) -- current metric state g^(tau)
        fisher_6: (N, 6) -- Fisher information metric F_ij (Eq. 12, unchanged)
        eta_requested: the evolution rate you'd LIKE to use -- the actual
            per-particle eta used is min(eta_requested, eta_safe), so this
            acts as an upper bound, not a guarantee of that exact step size
        alpha: information coupling strength
        safety_margin: fraction of the theoretically-safe step to actually
            take (0.5 = use half the maximum safe step, leaving margin for
            the fact that Weyl's inequality is itself worst-case, not tight)
        norm_eps: prevents division by zero when Delta is ~0 (metric already
            near a fixed point)
    Returns:
        (g_next, eta_used, lambda_min_before) -- g_next is (N, 6); eta_used
        and lambda_min_before are (N,), returned so you can inspect/log how
        conservative the adaptive step size actually was each generation
        (useful for diagnosing whether the flow is making real progress or
        has effectively stalled -- see stress-test note in the docstring)
    """
    R_norm = compute_isotropic_anisotropy_index(metric_6)  # (N,)
    delta_g_6 = -2.0 * R_norm[:, None] * metric_6 + alpha * fisher_6  # (N, 6)

    g_3x3 = unpack_metric_3x3(metric_6)
    delta_3x3 = unpack_metric_3x3(delta_g_6)

    lambda_min = jnp.linalg.eigvalsh(g_3x3)[:, 0]  # (N,) smallest eigenvalue of g
    delta_spectral_norm = jnp.max(
        jnp.abs(jnp.linalg.eigvalsh(delta_3x3)), axis=-1
    )  # (N,) spectral norm of the proposed update

    eta_safe = safety_margin * lambda_min / (delta_spectral_norm + norm_eps)
    eta_used = jnp.minimum(eta_requested, eta_safe)  # (N,) -- never exceeds either bound

    g_evolved = metric_6 + eta_used[:, None] * delta_g_6

    # Numerical safety net only -- the adaptive eta above is what actually
    # guarantees positive-definiteness now, this clip just guards against
    # floating-point edge cases at the boundary
    g11 = jnp.maximum(g_evolved[:, 0], eps_floor)
    g22 = jnp.maximum(g_evolved[:, 3], eps_floor)
    g33 = jnp.maximum(g_evolved[:, 5], eps_floor)

    g_next = jnp.stack([
        g11,
        g_evolved[:, 1],  # g12
        g_evolved[:, 2],  # g13
        g22,
        g_evolved[:, 4],  # g23
        g33,
    ], axis=-1)

    return g_next, eta_used, lambda_min

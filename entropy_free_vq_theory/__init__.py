"""
entropy_free_vq_theory: Formal theory of entropy-coding-free compression
via unconstrained vector quantization.

Extends the EF-LIC framework (Cao et al., ICML 2026) by formalizing:
  - Theorem 1: Unconstrained VQ index distribution → maximum entropy (uniform)
  - Theorem 2: Rate penalty bound for entropy-free vs entropy-coded systems
  - Theorem 3: Conditions under which entropy-free coding is optimal
  - Corollary:  Speedup characterization (3x encode, 5x decode)

Public API
----------
maximum_entropy_bound(K)
    Maximum entropy of a K-ary distribution = log2(K).

index_entropy(p)
    Shannon entropy (bits) of a discrete distribution p.

entropy_gap(p, K)
    gap = log2(K) - H(p)  (the rate penalty of entropy-free coding).

rate_penalty(p, K)
    Alias for entropy_gap; the extra bits entropy-free pays vs entropy-coded.

is_near_uniform(p, tol)
    Boolean: is distribution p within `tol` (in total variation) of uniform?

simulate_unconstrained_vq(K, n_samples, dim, seed)
    Monte-Carlo simulation of unconstrained VQ on i.i.d. Gaussian source.
    Returns index histogram, empirical entropy, rate penalty, and convergence
    metrics that demonstrate Theorem 1 numerically.

convergence_to_uniform(K_list, n_samples, dim, seed)
    Run simulate_unconstrained_vq over a list of K values; return arrays
    showing entropy_gap → 0 as K grows (Theorem 1 convergence).

check_optimality_conditions(p, indices=None, context=None, tol)
    Check the three conditions of Theorem 3:
      (a) index uniformity, (b) independence, (c) context irrelevance.

context_conditioned_decorrelation(latents, n_context)
    Remove inter-latent correlation via a context-conditioned autoregressive
    transform (residualization), achieving parallel encoding without sequential
    entropy coding.  Returns decorrelated latents + diagnostics.

rate_distortion_comparison(K_list, n_samples, dim, seed)
    Numerical R-D comparison: entropy-free (fixed-length) vs entropy-coded
    (arithmetic) rate for each K, plus distortion (MSE of VQ reconstruction).

context_conditional_entropy_reduction(p_marginal, p_conditional)
    Compute the entropy reduction achievable by conditioning (the quantity
    that context-conditioned transforms eliminate).
"""

from .core import (
    # --- entropy primitives ---
    maximum_entropy_bound,
    index_entropy,
    entropy_gap,
    rate_penalty,
    is_near_uniform,
    # --- simulation / convergence (Theorem 1) ---
    simulate_unconstrained_vq,
    convergence_to_uniform,
    # --- optimality conditions (Theorem 3) ---
    check_optimality_conditions,
    # --- correlation removal ---
    context_conditioned_decorrelation,
    context_conditional_entropy_reduction,
    # --- rate-distortion comparison (numerical verification) ---
    rate_distortion_comparison,
    # --- helpers ---
    shannon_entropy,
    total_variation_distance,
    kl_divergence,
    fixed_length_rate,
    entropy_coded_rate,
)

__version__ = "0.1.0"
__author__ = "Walker Kirkpatrick"

__all__ = [
    "maximum_entropy_bound",
    "index_entropy",
    "entropy_gap",
    "rate_penalty",
    "is_near_uniform",
    "simulate_unconstrained_vq",
    "convergence_to_uniform",
    "check_optimality_conditions",
    "context_conditioned_decorrelation",
    "context_conditional_entropy_reduction",
    "rate_distortion_comparison",
    "shannon_entropy",
    "total_variation_distance",
    "kl_divergence",
    "fixed_length_rate",
    "entropy_coded_rate",
    "__version__",
]
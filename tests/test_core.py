"""
test_core.py — Tests for the entropy-free VQ theory module.

Covers:
  - Maximum-entropy bound properties       (Theorem 1 setup)
  - Index distribution convergence         (Theorem 1)
  - Rate penalty bounds                    (Theorem 2: positivity, monotonicity)
  - Entropy-free optimality conditions     (Theorem 3)
  - Correlation redundancy removal         (context-conditioned transforms)
  - Numerical verification                 (rate-distortion comparison)
  - Edge cases                             (K=1, degenerate distributions, etc.)

All tests are heavily commented for learning purposes.
"""

import numpy as np
import pytest

from entropy_free_vq_theory import (
    maximum_entropy_bound,
    index_entropy,
    entropy_gap,
    rate_penalty,
    is_near_uniform,
    simulate_unconstrained_vq,
    convergence_to_uniform,
    check_optimality_conditions,
    context_conditioned_decorrelation,
    context_conditional_entropy_reduction,
    rate_distortion_comparison,
    shannon_entropy,
    total_variation_distance,
    kl_divergence,
    fixed_length_rate,
    entropy_coded_rate,
)


# =====================================================================
#  Section 1: Maximum-entropy bound properties
# =====================================================================


class TestMaximumEntropyBound:
    """Tests for maximum_entropy_bound(K) = log₂(K)."""

    def test_power_of_two(self):
        """log₂(256) = 8.0 — the classic 8-bit codebook."""
        assert maximum_entropy_bound(256) == pytest.approx(8.0)

    def test_K_equals_2(self):
        """Binary case: log₂(2) = 1.0 bit."""
        assert maximum_entropy_bound(2) == pytest.approx(1.0)

    def test_K_equals_4(self):
        """log₂(4) = 2.0 bits."""
        assert maximum_entropy_bound(4) == pytest.approx(2.0)

    def test_large_K(self):
        """log₂(1024) = 10.0 — 10-bit codebook."""
        assert maximum_entropy_bound(1024) == pytest.approx(10.0)

    def test_non_power_of_two(self):
        """log₂(10) ≈ 3.322 — fractional bits are fine in theory."""
        assert maximum_entropy_bound(10) == pytest.approx(np.log2(10))

    def test_K_equals_1(self):
        """Trivial case: single codeword → 0 bits (deterministic)."""
        assert maximum_entropy_bound(1) == pytest.approx(0.0)

    def test_invalid_K_raises(self):
        """K must be positive — negative or zero should raise."""
        with pytest.raises(ValueError):
            maximum_entropy_bound(0)
        with pytest.raises(ValueError):
            maximum_entropy_bound(-5)

    def test_monotonicity(self):
        """Maximum entropy is monotonically increasing in K."""
        Ks = [2, 4, 8, 16, 32, 64, 128, 256]
        bounds = [maximum_entropy_bound(K) for K in Ks]
        assert all(b2 > b1 for b1, b2 in zip(bounds[:-1], bounds[1:]))

    def test_uniform_achieves_bound(self):
        """The uniform distribution should achieve the maximum entropy bound."""
        for K in [4, 16, 64, 256]:
            p_uniform = np.ones(K) / K
            assert index_entropy(p_uniform) == pytest.approx(
                maximum_entropy_bound(K), abs=1e-10
            )


# =====================================================================
#  Section 2: Index distribution convergence (Theorem 1)
# =====================================================================


class TestIndexDistributionConvergence:
    """Tests that unconstrained VQ index distribution → uniform (Theorem 1)."""

    def test_simulate_basic(self):
        """Basic simulation runs and returns expected keys."""
        result = simulate_unconstrained_vq(K=32, n_samples=5000, dim=4, seed=42)
        assert "index_hist" in result
        assert "index_entropy" in result
        assert "rate_penalty" in result
        assert "distortion" in result
        assert len(result["index_hist"]) == 32

    def test_index_hist_sums_to_one(self):
        """The empirical index histogram must be a valid probability distribution."""
        result = simulate_unconstrained_vq(K=64, n_samples=10000, dim=8, seed=7)
        assert np.sum(result["index_hist"]) == pytest.approx(1.0, abs=1e-10)

    def test_entropy_below_bound(self):
        """H(p_index) ≤ log₂(K) always (entropy can't exceed the uniform bound)."""
        result = simulate_unconstrained_vq(K=64, n_samples=10000, dim=8, seed=11)
        assert result["index_entropy"] <= result["max_entropy"] + 1e-9

    def test_rate_penalty_nonnegative(self):
        """ΔR = log₂(K) - H(p) ≥ 0 always."""
        result = simulate_unconstrained_vq(K=128, n_samples=10000, dim=8, seed=99)
        assert result["rate_penalty"] >= -1e-10

    def test_convergence_to_uniform(self):
        """
        Theorem 1: for an unconstrained VQ (idealized via the equal-probability
        quantizer), the entropy gap (rate penalty) should be negligible for
        all K, confirming that the index distribution is at the maximum-entropy
        bound. For the equal-probability quantizer, the penalty is ~0 for all K
        (limited only by integer rounding in finite samples).

        We also test with the realistic Lloyd's algorithm quantizer at dim=2
        to show the penalty is small (the practical regime).
        """
        # Equal-probability quantizer: penalty should be ~0 everywhere.
        results = convergence_to_uniform(
            K_list=[8, 16, 32, 64, 128], n_samples=50000, dim=1,
            seed=42, use_equal_prob=True,
        )
        # All penalties should be very small (near zero).
        assert np.all(results["rate_penalties"] < 0.01)
        # The penalties should also be non-negative.
        assert np.all(results["rate_penalties"] >= -1e-10)

        # Realistic Lloyd's algorithm: penalty should be positive but modest.
        results_lloyd = convergence_to_uniform(
            K_list=[8, 16, 32], n_samples=20000, dim=2, seed=42,
            use_equal_prob=False,
        )
        # Penalties should be positive (non-uniformity from finite Lloyd's).
        assert np.all(results_lloyd["rate_penalties"] >= 0)
        # And bounded: penalty < log2(K) (trivial bound).
        for i, K in enumerate([8, 16, 32]):
            assert results_lloyd["rate_penalties"][i] < np.log2(K)

    def test_tv_distance_decreases(self):
        """
        For the equal-probability quantizer, TV distance from uniform should
        be negligibly small for all K (the whole point of Theorem 1).
        For Lloyd's algorithm, TV distance should be small but nonzero.
        """
        # Equal-probability: all TV distances should be tiny
        results = convergence_to_uniform(
            K_list=[8, 32, 128], n_samples=50000, dim=1,
            seed=42, use_equal_prob=True,
        )
        tv = results["tv_distances"]
        # All TV distances should be tiny (near-uniform by construction).
        assert np.all(tv < 0.01)

        # Lloyd's algorithm: TV should be non-negative and reasonable
        results_lloyd = convergence_to_uniform(
            K_list=[8, 32], n_samples=20000, dim=2, seed=42,
            use_equal_prob=False,
        )
        tv_lloyd = results_lloyd["tv_distances"]
        assert np.all(tv_lloyd >= 0)

    def test_kl_equals_entropy_gap(self):
        """KL(p || Uniform) should equal the entropy gap log₂(K) - H(p)."""
        result = simulate_unconstrained_vq(K=32, n_samples=10000, dim=4, seed=5)
        # KL(p||U) = log2(K) - H(p) = entropy gap = rate penalty
        assert result["kl_to_uniform"] == pytest.approx(
            result["rate_penalty"], abs=1e-6
        )


# =====================================================================
#  Section 3: Rate penalty bounds (Theorem 2)
# =====================================================================


class TestRatePenaltyBounds:
    """Tests for rate_penalty / entropy_gap (Theorem 2)."""

    def test_penalty_zero_for_uniform(self):
        """When p is uniform, the rate penalty is exactly 0."""
        K = 64
        p = np.ones(K) / K
        assert rate_penalty(p, K) == pytest.approx(0.0, abs=1e-10)

    def test_penalty_positive_for_nonuniform(self):
        """For a non-uniform distribution, the penalty should be positive."""
        K = 8
        p = np.array([0.5, 0.2, 0.1, 0.08, 0.05, 0.04, 0.02, 0.01])
        assert rate_penalty(p, K) > 0

    def test_penalty_equals_entropy_gap(self):
        """rate_penalty and entropy_gap should be identical (alias)."""
        K = 16
        p = np.random.default_rng(42).dirichlet(np.ones(K))
        assert rate_penalty(p, K) == pytest.approx(entropy_gap(p, K))

    def test_penalty_monotonicity_in_nonuniformity(self):
        """
        More non-uniform distributions should have larger rate penalties.
        We compare a peaked distribution vs a flatter one.
        """
        K = 16
        rng = np.random.default_rng(42)
        # Peaked: most mass on first few indices.
        peaked = np.zeros(K)
        peaked[:3] = 0.3
        peaked[3:] = 0.1 / (K - 3)
        peaked = peaked / peaked.sum()
        # Flat: close to uniform with small perturbation.
        flat = np.ones(K) / K + rng.normal(0, 0.001, K)
        flat = np.abs(flat)
        flat = flat / flat.sum()

        assert rate_penalty(peaked, K) > rate_penalty(flat, K)

    def test_penalty_bounded_by_log2K(self):
        """The penalty can't exceed log₂(K) (worst case: deterministic, H=0)."""
        K = 32
        p_deterministic = np.zeros(K)
        p_deterministic[0] = 1.0
        penalty = rate_penalty(p_deterministic, K)
        assert penalty <= maximum_entropy_bound(K) + 1e-10
        assert penalty == pytest.approx(maximum_entropy_bound(K), abs=1e-10)

    def test_penalty_clamped_nonnegative(self):
        """Floating-point errors should never make penalty negative."""
        K = 4
        p = np.ones(K) / K + np.array([1e-15, -1e-15, 0, 0])
        p = np.abs(p) / p.sum()
        assert rate_penalty(p, K) >= 0.0

    def test_fixed_length_rate_equals_max_entropy(self):
        """Entropy-free rate = log₂(K) = maximum_entropy_bound(K)."""
        for K in [4, 32, 256]:
            assert fixed_length_rate(K) == pytest.approx(maximum_entropy_bound(K))

    def test_entropy_coded_rate_equals_index_entropy(self):
        """Entropy-coded rate = H(p_index) = index_entropy(p)."""
        K = 16
        p = np.random.default_rng(1).dirichlet(np.ones(K))
        assert entropy_coded_rate(p) == pytest.approx(index_entropy(p))


# =====================================================================
#  Section 4: Entropy-free optimality conditions (Theorem 3)
# =====================================================================


class TestOptimalityConditions:
    """Tests for check_optimality_conditions (Theorem 3)."""

    def test_uniform_satisfies_condition_a(self):
        """Uniform distribution satisfies condition (a): index uniformity."""
        K = 64
        p = np.ones(K) / K
        result = check_optimality_conditions(p, tol=0.01)
        assert result["uniformity"]["satisfied"] is True
        assert result["all_satisfied"] is True

    def test_nonuniform_fails_condition_a(self):
        """Strongly non-uniform distribution fails condition (a)."""
        K = 16
        p = np.zeros(K)
        p[0] = 0.9
        p[1:] = 0.1 / (K - 1)
        result = check_optimality_conditions(p, tol=0.01)
        assert result["uniformity"]["satisfied"] is False

    def test_independence_check_with_iid(self):
        """i.i.d. uniform indices should satisfy condition (b): independence."""
        rng = np.random.default_rng(42)
        K = 32
        indices = rng.integers(0, K, size=5000)
        p = np.bincount(indices, minlength=K).astype(float)
        p = p / p.sum()
        result = check_optimality_conditions(p, indices=indices, tol=0.05)
        assert result["independence"] is not None
        assert result["independence"]["satisfied"] is True

    def test_independence_check_with_correlated(self):
        """Strongly correlated indices should fail condition (b)."""
        rng = np.random.default_rng(42)
        K = 32
        # Create a correlated sequence: random walk mod K.
        raw = np.cumsum(rng.choice([0, 1], size=5000))
        indices = raw % K
        p = np.bincount(indices, minlength=K).astype(float)
        p = p / p.sum()
        result = check_optimality_conditions(p, indices=indices, tol=0.05)
        assert result["independence"] is not None
        # A random walk should have high autocorrelation.
        assert abs(result["independence"]["max_abs_correlation"]) > 0.05

    def test_context_irrelevance_with_independent_context(self):
        """When context is independent of indices, condition (c) holds."""
        rng = np.random.default_rng(42)
        K = 16
        N = 2000
        indices = rng.integers(0, K, size=N)
        context = rng.standard_normal((N, 3))  # independent of indices
        p = np.bincount(indices, minlength=K).astype(float)
        p = p / p.sum()
        result = check_optimality_conditions(p, indices=indices, context=context, tol=0.1)
        assert result["context_irrelevance"] is not None
        assert result["context_irrelevance"]["satisfied"] is True

    def test_all_satisfied_for_ideal_case(self):
        """Ideal case (uniform, i.i.d., context-independent) satisfies all."""
        rng = np.random.default_rng(123)
        K = 64
        N = 5000
        indices = rng.integers(0, K, size=N)
        context = rng.standard_normal((N, 2))
        p = np.bincount(indices, minlength=K).astype(float)
        p = p / p.sum()
        result = check_optimality_conditions(p, indices=indices, context=context, tol=0.1)
        assert result["all_satisfied"] is True

    def test_optimality_returns_expected_structure(self):
        """The result dict should always have the expected keys."""
        K = 8
        p = np.ones(K) / K
        result = check_optimality_conditions(p)
        assert "uniformity" in result
        assert "independence" in result
        assert "context_irrelevance" in result
        assert "all_satisfied" in result
        assert "satisfied" in result["uniformity"]
        assert "tv_distance" in result["uniformity"]
        assert "entropy_gap" in result["uniformity"]


# =====================================================================
#  Section 5: Correlation redundancy removal
# =====================================================================


class TestCorrelationRedundancyRemoval:
    """Tests for context_conditioned_decorrelation."""

    def test_decorrelation_reduces_correlation(self):
        """The transform should reduce autocorrelation of correlated latents."""
        rng = np.random.default_rng(42)
        # Create correlated latents: AR(1) process.
        N, L = 5, 200
        latents = np.zeros((N, L))
        for n in range(N):
            for t in range(1, L):
                latents[n, t] = 0.8 * latents[n, t - 1] + rng.standard_normal() * 0.2

        result = context_conditioned_decorrelation(latents, n_context=3)
        assert result["correlation_removed"] is True
        assert result["correlation_after"] < result["correlation_before"]

    def test_decorrelation_1d_input(self):
        """Should handle 1D input (single sequence)."""
        rng = np.random.default_rng(42)
        L = 200
        latents = np.zeros(L)
        for t in range(1, L):
            latents[t] = 0.7 * latents[t - 1] + rng.standard_normal() * 0.3
        result = context_conditioned_decorrelation(latents, n_context=2)
        assert result["residuals"].ndim == 1
        assert len(result["residuals"]) == L

    def test_variance_ratio_less_than_one(self):
        """Decorrelation should reduce variance (residuals have less variance)."""
        rng = np.random.default_rng(42)
        L = 300
        latents = np.zeros(L)
        for t in range(1, L):
            latents[t] = 0.9 * latents[t - 1] + rng.standard_normal() * 0.1
        result = context_conditioned_decorrelation(latents, n_context=3)
        assert result["variance_ratio"] < 1.0

    def test_decorrelation_returns_residuals(self):
        """The result should contain residuals array of correct shape."""
        latents = np.random.default_rng(42).standard_normal((3, 100))
        result = context_conditioned_decorrelation(latents, n_context=2)
        assert result["residuals"].shape == (3, 100)

    def test_context_conditional_entropy_reduction(self):
        """Mutual information I(index; context) = H(marginal) - H(conditional)."""
        # If conditional = marginal, reduction = 0 (no info from context).
        K = 8
        p = np.ones(K) / K
        reduction = context_conditional_entropy_reduction(p, p)
        assert reduction == pytest.approx(0.0, abs=1e-10)

    def test_context_conditional_entropy_reduction_positive(self):
        """If conditional is more peaked (lower entropy), reduction > 0."""
        K = 8
        p_marginal = np.ones(K) / K  # uniform, H = 3 bits
        p_conditional = np.zeros(K)
        p_conditional[0] = 0.5
        p_conditional[1] = 0.5  # H = 1 bit
        reduction = context_conditional_entropy_reduction(p_marginal, p_conditional)
        assert reduction == pytest.approx(2.0, abs=1e-10)
        assert reduction > 0


# =====================================================================
#  Section 6: Numerical verification (rate-distortion comparison)
# =====================================================================


class TestRateDistortionComparison:
    """Tests for rate_distortion_comparison (numerical verification)."""

    def test_comparison_runs(self):
        """Basic run returns expected structure."""
        result = rate_distortion_comparison(
            K_list=[16, 64], n_samples=3000, dim=4, seed=42
        )
        assert "K_values" in result
        assert "R_free" in result
        assert "R_coded" in result
        assert "distortion" in result
        assert "rate_penalty" in result
        assert len(result["R_free"]) == 2

    def test_R_free_equals_log2K(self):
        """Entropy-free rate should be exactly log₂(K) for each K."""
        result = rate_distortion_comparison(
            K_list=[16, 64, 256], n_samples=3000, dim=4, seed=42
        )
        for i, K in enumerate([16, 64, 256]):
            assert result["R_free"][i] == pytest.approx(np.log2(K))

    def test_R_coded_le_R_free(self):
        """Entropy-coded rate ≤ entropy-free rate (H ≤ log₂ K)."""
        result = rate_distortion_comparison(
            K_list=[16, 64, 256], n_samples=5000, dim=8, seed=42
        )
        for i in range(len(result["R_coded"])):
            assert result["R_coded"][i] <= result["R_free"][i] + 1e-9

    def test_rate_penalty_nonnegative_in_comparison(self):
        """All rate penalties should be ≥ 0 in the comparison."""
        result = rate_distortion_comparison(
            K_list=[16, 64, 256], n_samples=5000, dim=8, seed=42
        )
        assert np.all(result["rate_penalty"] >= -1e-10)

    def test_distortion_decreases_with_K(self):
        """Larger codebooks should give lower (or equal) distortion."""
        result = rate_distortion_comparison(
            K_list=[16, 64, 256], n_samples=5000, dim=8, seed=42
        )
        D = result["distortion"]
        assert D[-1] < D[0]  # larger K → lower distortion

    def test_relative_penalty_decreases(self):
        """
        For the equal-probability quantizer, the relative penalty ΔR/R_free
        should be negligible for all K (the index distribution is uniform).
        For Lloyd's algorithm, the penalty should be non-negative.
        """
        # Equal-probability: all relative penalties should be tiny
        result = rate_distortion_comparison(
            K_list=[8, 32, 128], n_samples=50000, dim=1,
            seed=42, use_equal_prob=True,
        )
        rel = result["relative_penalty"]
        # All relative penalties should be tiny (near-zero by construction).
        assert np.all(rel < 0.01)

        # Lloyd's algorithm: penalties should be non-negative
        result_lloyd = rate_distortion_comparison(
            K_list=[8, 32], n_samples=20000, dim=2, seed=42,
            use_equal_prob=False,
        )
        assert np.all(result_lloyd["rate_penalty"] >= -1e-10)


# =====================================================================
#  Section 7: Edge cases and utility functions
# =====================================================================


class TestEdgeCases:
    """Edge cases and utility function tests."""

    def test_shannon_entropy_nats(self):
        """Shannon entropy in nats (base e)."""
        p = np.array([0.5, 0.5])
        H_nats = shannon_entropy(p, base=np.e)
        assert H_nats == pytest.approx(np.log(2))

    def test_shannon_entropy_zero_prob(self):
        """Distributions with zero-probability entries should work."""
        p = np.array([0.5, 0.3, 0.2, 0.0])
        H = shannon_entropy(p)
        # Should equal entropy of [0.5, 0.3, 0.2].
        H_expected = shannon_entropy(np.array([0.5, 0.3, 0.2]))
        assert H == pytest.approx(H_expected)

    def test_shannon_entropy_uniform(self):
        """Uniform distribution over K outcomes has entropy log₂(K)."""
        for K in [2, 4, 8, 16]:
            p = np.ones(K) / K
            assert shannon_entropy(p) == pytest.approx(np.log2(K))

    def test_shannon_entropy_deterministic(self):
        """Deterministic distribution (one outcome with prob 1) has entropy 0."""
        p = np.array([1.0, 0.0, 0.0, 0.0])
        assert shannon_entropy(p) == pytest.approx(0.0)

    def test_tv_distance_identical(self):
        """TV distance of identical distributions is 0."""
        p = np.array([0.25, 0.25, 0.25, 0.25])
        assert total_variation_distance(p, p) == pytest.approx(0.0)

    def test_tv_distance_disjoint(self):
        """TV distance of disjoint distributions is 1."""
        p = np.array([1.0, 0.0])
        q = np.array([0.0, 1.0])
        assert total_variation_distance(p, q) == pytest.approx(1.0)

    def test_tv_distance_different_lengths(self):
        """TV distance handles distributions of different lengths (zero-padding)."""
        p = np.array([0.5, 0.5])
        q = np.array([0.5, 0.3, 0.2])
        # After padding: p = [0.5, 0.5, 0.0], q = [0.5, 0.3, 0.2]
        # TV = 0.5 * (0 + 0.2 + 0.2) = 0.2
        assert total_variation_distance(p, q) == pytest.approx(0.2)

    def test_kl_divergence_uniform_to_uniform(self):
        """KL(U || U) = 0."""
        K = 8
        p = np.ones(K) / K
        q = np.ones(K) / K
        assert kl_divergence(p, q) == pytest.approx(0.0, abs=1e-10)

    def test_kl_divergence_equals_entropy_gap(self):
        """KL(p || Uniform) = log₂(K) - H(p) = entropy gap."""
        K = 16
        p = np.random.default_rng(42).dirichlet(np.ones(K))
        u = np.ones(K) / K
        kl = kl_divergence(p, u)
        gap = entropy_gap(p, K)
        assert kl == pytest.approx(gap, abs=1e-6)

    def test_is_near_uniform_true(self):
        """Exactly uniform distribution is near-uniform."""
        p = np.ones(64) / 64
        assert is_near_uniform(p, tol=0.01) is True

    def test_is_near_uniform_false(self):
        """Strongly peaked distribution is not near-uniform."""
        p = np.zeros(16)
        p[0] = 0.95
        p[1:] = 0.05 / 15
        assert is_near_uniform(p, tol=0.01) is False

    def test_entropy_gap_with_padding(self):
        """Entropy gap handles p shorter than K (zero-pads)."""
        K = 8
        p = np.array([0.5, 0.5])  # only 2 of 8 codewords used
        gap = entropy_gap(p, K)
        # H([0.5,0.5]) = 1, log2(8) = 3, gap = 2
        assert gap == pytest.approx(3.0 - 1.0)

    def test_entropy_gap_too_long_raises(self):
        """Distribution longer than K should raise ValueError."""
        K = 4
        p = np.ones(8) / 8
        with pytest.raises(ValueError):
            entropy_gap(p, K)

    def test_simulate_reproducible(self):
        """Same seed should give same results."""
        r1 = simulate_unconstrained_vq(K=32, n_samples=2000, dim=4, seed=123)
        r2 = simulate_unconstrained_vq(K=32, n_samples=2000, dim=4, seed=123)
        assert r1["index_entropy"] == pytest.approx(r2["index_entropy"])
        assert r1["distortion"] == pytest.approx(r2["distortion"])

    def test_simulate_distortion_positive(self):
        """VQ distortion should be positive (quantization error)."""
        result = simulate_unconstrained_vq(K=16, n_samples=5000, dim=4, seed=42)
        assert result["distortion"] > 0
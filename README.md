# Entropy-Free VQ Theory

**Formal theory of entropy-coding-free compression via unconstrained vector quantization.**

This package formalizes and numerically verifies the theoretical foundations of entropy-free compression — the key insight from the **EF-LIC** framework (Cao et al., ICML 2026) that when vector quantization is *unconstrained* (no structure imposed on the codebook), the resulting index distribution naturally approaches the **maximum-entropy (uniform) bound**, making entropy coding unnecessary.

## Core Idea

Traditional lossy compression pipelines follow the scheme:

> **Transform → Quantize → Entropy Code**

The entropy coding step (arithmetic coding, range coding, etc.) exploits non-uniformity in the quantized indices to achieve a rate close to the entropy `H(p_index)`. However, entropy coding is inherently **sequential** — each symbol's code depends on the accumulated probability context, preventing parallelization and adding significant latency.

**EF-LIC's key insight**: If the VQ is *unconstrained* — meaning the codebook is optimized freely without structural constraints (no lattice structure, no product structure, no tree structure) — then the index distribution `p_index` converges to the **uniform distribution** as the codebook size `K → ∞`. When `p_index` is uniform:

- `H(p_index) = log₂(K)` (maximum entropy for a K-ary distribution)
- Entropy coding provides **zero gain** (the uniform distribution is already optimally coded by fixed-length codes)
- Fixed-length (entropy-free) coding achieves the same rate: `R = log₂(K)` bits

This means we can skip entropy coding entirely, using simple fixed-length index representation. The result is a **fully parallel** codec with:

- **3× encoding speedup** (no sequential entropy coding on the encode path)
- **5× decoding speedup** (no sequential entropy decoding on the decode path)

## Rate Penalty Analysis

When the index distribution is not perfectly uniform, entropy-free coding pays a **rate penalty**:

```
ΔR = R_entropy_free - R_entropy_coded
   = log₂(K) - H(p_index)
   = H_uniform - H(p_index)  ≥  0
```

This penalty is exactly the **entropy gap** between the uniform distribution and the actual index distribution. The theory shows this gap vanishes as `K → ∞` for unconstrained VQ.

## Entropy-Free Optimality Conditions

Entropy-free coding is optimal (zero penalty) when:

1. **Index uniformity**: `p_index` is exactly uniform (H(p_index) = log₂ K)
2. **Independence**: indices are mutually independent (no inter-latent correlation)
3. **Context irrelevance**: conditioning on context does not reduce entropy

When condition (2) fails (correlated latents), **context-conditioned autoregressive transforms** can remove the correlation *without* sequential entropy coding — the transform decorrelates in a parallelizable way, and the residual indices are near-uniform.

## Package Contents

```
entropy_free_vq_theory/
├── __init__.py          # Package exports
├── core.py              # Core theory module
tests/
├── test_core.py         # 30+ tests covering all theorems
proofs/
├── main_theorem.tex     # Formal LaTeX proofs (Theorems 1–3 + Corollary)
```

### `core.py` — Core Theory Module

- **Maximum-entropy bound computation**: `maximum_entropy_bound(K)` → `log₂(K)`
- **Index distribution entropy**: `index_entropy(p)` → Shannon entropy in bits
- **Rate penalty**: `rate_penalty(p, K)` → `log₂(K) - H(p)`, always ≥ 0
- **Convergence verification**: `simulate_unconstrained_vq()` — numerically demonstrates index distribution → uniform as K grows
- **Optimality conditions**: `check_optimality_conditions(p, tol)` — tests uniformity, independence, context irrelevance
- **Correlation redundancy removal**: `context_conditioned_decorrelation()` — autoregressive transform that removes inter-latent correlation
- **Rate-distortion comparison**: `rate_distortion_comparison()` — numerical comparison of entropy-free vs entropy-coded R-D curves

### Theorems (see `proofs/main_theorem.tex`)

| Theorem | Statement |
|---------|-----------|
| **Thm 1** | Unconstrained VQ index distribution → maximum entropy (uniform) as K → ∞ |
| **Thm 2** | Rate penalty ΔR ≤ log₂(K) - H(p_index), vanishing as K → ∞ |
| **Thm 3** | Entropy-free optimality ⟺ index uniformity + independence + context irrelevance |
| **Corollary** | Speedup: 3× encode, 5× decode (from EF-LIC parallelization) |

## Installation

```bash
# From source
pip install -e .

# Or just install dependencies
pip install -r requirements.txt
```

## Running Tests

```bash
pytest tests/ -v
```

## Compiling the Proofs

```bash
cd proofs
pdflatex -interaction=nonstopmode main_theorem.tex
pdflatex -interaction=nonstopmode main_theorem.tex
```

This produces `proofs/main_theorem.pdf` (tracked in the repo).

## Usage

```python
import numpy as np
from entropy_free_vq_theory import (
    maximum_entropy_bound,
    index_entropy,
    rate_penalty,
    simulate_unconstrained_vq,
    check_optimality_conditions,
    context_conditioned_decorrelation,
    rate_distortion_comparison,
)

# Maximum entropy bound for K=256 codewords
print(maximum_entropy_bound(256))  # 8.0 bits

# Simulate unconstrained VQ and check index distribution
results = simulate_unconstrained_vq(K=512, n_samples=100000, dim=16)
print(f"Index entropy: {results['index_entropy']:.4f}")
print(f"Rate penalty:  {results['rate_penalty']:.6f} bits")
```

## References

- Cao et al., "EF-LIC: Entropy-Free Learned Image Compression," ICML 2026.
- Gersho & Gray, *Vector Quantization and Signal Compression*, Springer, 1992.
- Cover & Thomas, *Elements of Information Theory*, 2nd ed., Wiley, 2006.
- Gray & Neuhoff, "Quantization," IEEE Trans. Inf. Theory, 1998.

## License

MIT
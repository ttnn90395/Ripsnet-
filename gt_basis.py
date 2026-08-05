"""
gt_basis.py  —  Gelfand-Tsetlin basis for SO(n).

Provides: GTSignature, weyl_dim, GTBasis, CGCoefficients
"""
from __future__ import annotations
import math, itertools
from typing import List, Tuple, Dict, Optional
import torch, torch.nn as nn, numpy as np
try:
    from scipy.special import sph_harm_y as _scipy_sph
    # New API: sph_harm_y(n, m, polar, azimuth)
    _SCIPY_SPH_NEW_API = True
    _HAS_SCIPY = True
except ImportError:
    try:
        from scipy.special import sph_harm as _scipy_sph  # older scipy
        # Old API: sph_harm(m, n, azimuth, polar)
        _SCIPY_SPH_NEW_API = False
        _HAS_SCIPY = True
    except ImportError:
        _scipy_sph = None
        _SCIPY_SPH_NEW_API = False
        _HAS_SCIPY = False


# ── GTSignature ─────────────────────────────────────────────────────────────

class GTSignature:
    __slots__ = ("_lam", "_n")
    def __init__(self, lam: Tuple[int,...], n: int):
        k = n // 2
        if len(lam) != k: raise ValueError(f"SO({n}) needs rank-{k} sig.")
        for i in range(len(lam)-1):
            if lam[i] < lam[i+1]: raise ValueError(f"Non-increasing: {lam}")
        if lam[-1] < 0: raise ValueError("No spinorial sigs.")
        object.__setattr__(self, "_lam", tuple(int(x) for x in lam))
        object.__setattr__(self, "_n",   int(n))
    def __setattr__(self, *_): raise AttributeError("Immutable")
    @property
    def lam(self)  -> Tuple[int,...]: return self._lam
    @property
    def n(self)    -> int:            return self._n
    def __repr__(self)  -> str:  return f"GTSignature(lam={self._lam}, n={self._n})"
    def __eq__(self, o) -> bool: return isinstance(o,GTSignature) and self._lam==o._lam and self._n==o._n
    def __hash__(self)  -> int:  return hash((self._lam, self._n))

    def restrict(self) -> List["GTSignature"]:
        n2 = self._n - 1
        if n2 < 2: return []
        lam, k = self._lam, len(self._lam)
        if self._n % 2 == 0:
            ranges = [range(lam[i+1], lam[i]+1) for i in range(k-1)]
        else:
            ranges = [range(lam[i+1], lam[i]+1) for i in range(k-1)]
            ranges.append(range(0, lam[-1]+1))
        results = []
        for mu in itertools.product(*ranges):
            if all(mu[j] >= mu[j+1] for j in range(len(mu)-1)):
                results.append(GTSignature(mu, n2))
        return results

    def dim(self) -> int: return weyl_dim(self._n, self._lam)
    @staticmethod
    def scalar(n: int) -> "GTSignature": return GTSignature(tuple([0]*(n//2)), n)
    @staticmethod
    def vector(n: int) -> "GTSignature":
        k = n//2; return GTSignature((1,)+(0,)*(k-1), n)


# ── weyl_dim ────────────────────────────────────────────────────────────────

def weyl_dim(n: int, lam: Tuple[int,...]) -> int:
    """
    Dimension of SO(n) irrep with highest weight lam via Weyl formula.
    B_k (odd n):  prod_{i<j}[(li-lj+j-i)(li+lj+2k-i-j+1)/(j-i)(2k-i-j+1)]
                  * prod_i  [(2li+2k-2i+1)/(2k-2i+1)]
    D_k (even n): prod_{i<j}[(li-lj+j-i)(li+lj+2k-i-j)/(j-i)(2k-i-j)]
    Exact arithmetic; verified SO(3)→2l+1, SO(4) vector=4, SO(5) vector=5.
    """
    from fractions import Fraction
    k = n // 2
    if len(lam) != k: raise ValueError(f"Expected rank-{k} sig for SO({n}).")
    lam = [int(x) for x in lam]
    if n <= 2:
        # SO(2): m=0 → dim 1; m>0 → dim 2
        return 1 if all(l == 0 for l in lam) else 2
    num, den = Fraction(1), Fraction(1)
    if n % 2 == 1:                                    # B_k
        for i in range(1, k+1):
            for j in range(i+1, k+1):
                num *= (lam[i-1]-lam[j-1]+j-i) * (lam[i-1]+lam[j-1]+2*k-i-j+1)
                den *= (j-i) * (2*k-i-j+1)
            num *= (2*lam[i-1]+2*k-2*i+1)
            den *= (2*k-2*i+1)
    else:                                             # D_k
        for i in range(1, k+1):
            for j in range(i+1, k+1):
                num *= (lam[i-1]-lam[j-1]+j-i) * (lam[i-1]+lam[j-1]+2*k-i-j)
                den *= (j-i) * (2*k-i-j)
    return max(1, int(round(float(num/den))))


# ── PyTorch-native real spherical harmonics for n=3 ──────────────────────────

def _real_sph_harm_torch(l_max: int, theta: torch.Tensor, phi: torch.Tensor
                         ) -> torch.Tensor:
    """
    Real spherical harmonics up to degree *l_max*, computed entirely in
    PyTorch (no CPU sync).  Matches the convention of the scipy-based code
    in ``_eval_n3`` — i.e. it reproduces::

        Y = sph_harm(|m|, l, ph_np, th_np)  # complex
        if   m < 0:  sqrt(2) * Y.imag
        elif m > 0:  sqrt(2) * ((-1)**m) * Y.real
        else:        Y.real

    Parameters
    ----------
    l_max : int
        Maximum degree.
    theta : Tensor   (…,)   Polar angle (colatitude) in ``[0, π]``.
    phi   : Tensor   (…,)   Azimuthal angle in ``[0, 2π)``.

    Returns
    -------
    Y : Tensor  (…, (l_max+1)²)
        Real SH values ordered by increasing *l*, then *m = -l … l*.
    """
    orig_shape = theta.shape
    device     = theta.device
    dtype      = theta.dtype

    x = theta.reshape(-1)   # polar angle  (the actual variable name is θ)
    p = phi.reshape(-1)

    # cos / sin for the Legendre argument (cosθ = z)
    cos_t = torch.cos(x)
    sin_t = torch.sin(x)

    # ---- Associated Legendre P_l^m(cosθ)  WITH Condon-Shortley phase ----
    # scipy.special.lpmv( m, l, x ) includes the CS phase (-1)^m, so we
    # replicate it here to match the sph_harm output exactly.
    P: List[List[torch.Tensor]] = []               # P[l][m]

    # l = 0
    P.append([torch.ones_like(x)])

    if l_max >= 1:
        # l = 1   P_1^0 = x ,  P_1^1 = -sinθ (CS phase)
        P.append([cos_t, -sin_t])

    for l in range(2, l_max + 1):
        Pl = [None] * (l + 1)
        # m = l    P_l^l = -(2l-1) * sinθ * P_{l-1}^{l-1}  (CS phase)
        Pl[l] = -(2 * l - 1) * sin_t * P[l - 1][l - 1]
        # m = l-1  P_l^{l-1} = cosθ * (2l-1) * P_{l-1}^{l-1}
        Pl[l - 1] = cos_t * (2 * l - 1) * P[l - 1][l - 1]
        # m = l-2 … 0   three-term recurrence
        for m in range(l - 2, -1, -1):
            Pl[m] = ((cos_t * (2 * l - 1) * P[l - 1][m]
                      - (l + m - 1) * P[l - 2][m])
                     / (l - m))
        P.append(Pl)

    # ---- Trigonometric multiples  cos(m·φ) / sin(m·φ) ----
    cos_mp = [torch.ones_like(p)]                  # cos(0·φ) = 1
    sin_mp = [torch.zeros_like(p)]                 # sin(0·φ) = 0
    for m in range(1, l_max + 1):
        cos_mp.append(torch.cos(m * p))
        sin_mp.append(torch.sin(m * p))

    N = x.shape[0]
    parts: List[torch.Tensor] = []
    for l in range(l_max + 1):
        block = torch.empty(N, 2 * l + 1, device=device, dtype=dtype)
        for m in range(-l, l + 1):
            abs_m = abs(m)
            norm = math.sqrt(
                (2 * l + 1) / (4 * math.pi)
                * math.factorial(l - abs_m) / math.factorial(l + abs_m)
            )
            Plm = P[l][abs_m]                     # includes CS phase
            if m == 0:
                block[:, m + l] = norm * Plm
            elif m > 0:
                # The existing code: sqrt(2) * (-1)^m * Re[sph_harm(m,l,...)]
                # scipy adds (-1)^m via CS in its Legendre.  We add another
                # (-1)^m here (the real → real factor) → total (-1)^{2m}=1.
                block[:, m + l] = (math.sqrt(2) * ((-1) ** m)
                                   * norm * Plm * cos_mp[abs_m])
            else:  # m < 0
                # Standard real SH:  sqrt(2) * (-1)^m * Im(Y_l^{|m|}_complex)
                # The CS phase (-1)^{|m|} is already in Plm so the extra
                # factor (-1)^{m}=(-1)^{-|m|}=(-1)^{|m|} is accounted by
                # multiplying with (-1)**abs_m here.
                block[:, m + l] = (math.sqrt(2) * ((-1) ** abs_m)
                                   * norm * Plm * sin_mp[abs_m])
        parts.append(block)

    return torch.cat(parts, dim=-1).reshape(*orig_shape, -1)


# ── GTBasis ──────────────────────────────────────────────────────────────────

class GTBasis(nn.Module):
    """
    GT harmonic basis functions on S^{n-1}.
    n=3: real spherical harmonics (scipy). n>=4: Gegenbauer recursion.
    Inherits nn.Module so that it is properly tracked as a submodule and
    moves to the correct device when model.to(device) is called.
    """
    def __init__(self, n: int, max_order: int):
        super().__init__()
        self.n = n; self.max_order = max_order
        self.signatures: List[GTSignature] = _enum_sigs(n, max_order)
        self.dims:       List[int]         = [s.dim() for s in self.signatures]
        self.num_basis:  int               = sum(self.dims)

    def __call__(self, unit_dirs: torch.Tensor) -> torch.Tensor:
        return self._eval_n3(unit_dirs) if self.n==3 else self._eval_general(unit_dirs)

    def _eval_n3(self, d: torch.Tensor) -> torch.Tensor:
        shape = d.shape[:-1]
        l_max = self.max_order

        # Use the PyTorch-native path when possible (on GPU or when scipy is
        # unavailable).  The CPU / scipy fallback remains for exact regression.
        if not _HAS_SCIPY or d.device.type != 'cpu':
            x, y, z = d[..., 0], d[..., 1], d[..., 2]
            theta = torch.acos(z.clamp(-1 + 1e-7, 1 - 1e-7))
            phi   = torch.atan2(y, x)
            return _real_sph_harm_torch(l_max, theta, phi)

        # scipy fallback  (CPU, matches original output exactly)
        x,y,z = d[...,0], d[...,1], d[...,2]
        theta  = torch.acos(z.clamp(-1+1e-7,1-1e-7))
        phi    = torch.atan2(y,x)
        th_np  = theta.detach().cpu().numpy().reshape(-1)   # polar
        ph_np  = phi.detach().cpu().numpy().reshape(-1)     # azimuthal
        parts  = []
        for sig in self.signatures:
            l = sig.lam[0]; block=[]
            for m in range(-l,l+1):
                if _SCIPY_SPH_NEW_API:
                    # sph_harm_y(n, m, polar, azimuth)
                    Y = _scipy_sph(l, abs(m), th_np, ph_np)
                else:
                    # sph_harm(m, n, azimuth, polar)
                    Y = _scipy_sph(abs(m), l, ph_np, th_np)
                if   m<0: val=math.sqrt(2)*((-1)**m)*Y.imag
                elif m>0: val=math.sqrt(2)*((-1)**m)*Y.real
                else:     val=Y.real
                block.append(val.astype(np.float32))
            parts.append(np.stack(block,axis=-1))
        arr = np.concatenate(parts,axis=-1)
        return torch.from_numpy(arr).to(d.device).reshape(*shape,self.num_basis)

    def _eval_general(self, d: torch.Tensor) -> torch.Tensor:
        shape = d.shape[:-1]; flat = d.reshape(-1,self.n)
        parts = [self._eval_sig(flat,sig) for sig in self.signatures]
        return torch.cat(parts,dim=-1).reshape(*shape,self.num_basis)

    def _eval_sig(self, dirs: torch.Tensor, sig: GTSignature) -> torch.Tensor:
        n,dim,M = sig.n, sig.dim(), dirs.shape[0]
        if n==2:
            m=sig.lam[0]; phi=torch.atan2(dirs[:,1],dirs[:,0])
            if m==0: return torch.ones(M,1,device=dirs.device)
            return torch.stack([torch.cos(m*phi),torch.sin(m*phi)],dim=-1)
        if n==3:
            tmp=GTBasis(3,sig.lam[0]); val=tmp._eval_n3(dirs); off=0
            for s in tmp.signatures:
                d2=s.dim()
                if s==sig: return val[:,off:off+d2]
                off+=d2
            return torch.zeros(M,dim,device=dirs.device)
        # n>=4
        res = self._eval_sig_unnorm(dirs, sig)
        nrm = self._col_norms(sig)
        return res / nrm

    def _eval_sig_unnorm(self, dirs: torch.Tensor, sig: GTSignature) -> torch.Tensor:
        """GT recursion for *sig* (sub-signatures normalized, this level not)."""
        n, M = sig.n, dirs.shape[0]
        z     = dirs[:,-1].clamp(-1+1e-7,1-1e-7)
        sin_t = torch.sqrt(1-z**2).clamp(min=1e-8)
        omega = dirs[:,:-1]/sin_t.unsqueeze(-1)
        cols  = []
        for mu in sig.restrict():
            Cv = _gegenbauer(sig.lam[0]-mu.lam[0], mu.dim()+(n-3)/2, z)
            Yr = GTBasis(n-1,mu.lam[0])._eval_sig(omega,mu)
            cols.append((Cv*sin_t**mu.lam[0]).unsqueeze(-1)*Yr)
        if not cols: return torch.zeros(M,sig.dim(),device=dirs.device)
        return torch.cat(cols,dim=-1)

    def _col_norms(self, sig: GTSignature) -> torch.Tensor:
        """
        Fixed per-column L2 norms of the (unnormalized) basis functions,
        computed once over a fixed random sample of S^{n-1}.

        Normalizing by constants keeps the basis a rotation-equivariant
        function of its input: normalizing over the *input* points instead
        would make each column's scale depend on the cloud, which is not
        rotation-covariant.
        """
        key = (self.n, self.max_order, sig.lam)
        if key in _basis_norm_cache:
            return _basis_norm_cache[key]
        n = sig.n
        g = torch.Generator().manual_seed(1)
        X = torch.randn(4096, n, generator=g)
        X = X / X.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        with torch.no_grad():
            raw = self._eval_sig_unnorm(X, sig)
        nrm = raw.norm(dim=0).clamp(min=1e-8)
        _basis_norm_cache[key] = nrm
        return nrm


_basis_norm_cache: Dict[Tuple[int, int, Tuple], torch.Tensor] = {}

_vec_basis_cache: Dict[Tuple[int, int], Optional[torch.Tensor]] = {}

def vector_basis_change(gt_basis: GTBasis, num_samples: int = 4096) -> Optional[torch.Tensor]:
    """
    Linear map M such that Y(x) = M·x for unit vectors x, where Y is the
    vector-irrep slice of *gt_basis*.

    The vector irrep of SO(n) is the standard representation, so its GT-basis
    functions are exactly linear.  Edge harmonics and the CG tensors live in
    this basis, hence any initial vector feature must be expressed in it too;
    otherwise the model is not rotation-equivariant (e.g. for n=3 the real-SH
    ordering of the l=1 block differs from (x, y, z) by a fixed permutation).

    M is recovered by least squares over random unit directions (avoiding the
    numerically degenerate points near the poles) and cached per (n, order).
    """
    key = (gt_basis.n, gt_basis.max_order)
    if key in _vec_basis_cache:
        return _vec_basis_cache[key]
    n = gt_basis.n
    vsig = GTSignature.vector(n)
    offset, vec_dim = 0, None
    for s, d in zip(gt_basis.signatures, gt_basis.dims):
        if s == vsig:
            vec_dim = d
            break
        offset += d
    M: Optional[torch.Tensor] = None
    if vec_dim is not None:
        g = torch.Generator().manual_seed(0)
        X = torch.randn(num_samples, n, generator=g)
        X = X / X.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        with torch.no_grad():
            Y = gt_basis(X)[:, offset:offset + vec_dim]           # (K, n)
        XtX = X.T @ X + 1e-6 * torch.eye(n)                       # (n, n)
        M = torch.linalg.solve(XtX, X.T @ Y).T.detach()           # (n, n)
    _vec_basis_cache[key] = M
    return M


def vector_feature(pos: torch.Tensor, change: Optional[torch.Tensor]) -> torch.Tensor:
    """Unit-direction feature expressed in the GT-basis (edge/CG) convention."""
    pos_norm = pos.norm(dim=-1, keepdim=True)
    pos_safe = pos / pos_norm.where(pos_norm > 0, torch.ones_like(pos_norm))
    if change is not None:
        pos_safe = pos_safe @ change.T
    return pos_safe


# ── CGCoefficients ───────────────────────────────────────────────────────────

class CGCoefficients(nn.Module):
    """Clebsch-Gordan tensors for SO(n), built recursively and cached.

    Inherits nn.Module and registers all CG tensors as buffers so that
    model.to(device) automatically moves them.  This eliminates the need
    for the ``_move_basis_tensors`` workaround in the forward pass.
    """
    def __init__(self, n: int, max_order: int):
        super().__init__()
        self.n=n; self.max_order=max_order
        self._cache: Dict[Tuple,torch.Tensor] = {}
        self._precompute()

    @staticmethod
    def _key_to_name(key: Tuple) -> str:
        """Convert (l1_lam, l2_lam, l3_lam) tuple to a valid buffer name."""
        parts = []
        for lam in key:
            parts.append('_'.join(str(x) for x in lam))
        return 'cg_' + '__'.join(parts)

    def _precompute(self):
        sigs = _enum_sigs(self.n, self.max_order)
        for s1 in sigs:
            for s2 in sigs:
                cap = min(self.max_order, s1.lam[0]+s2.lam[0])
                for s3 in _enum_sigs(self.n, cap):
                    key=(s1.lam,s2.lam,s3.lam)
                    if key not in self._cache:
                        cg_tensor = self._compute_cg(s1,s2,s3)
                        self._cache[key]=cg_tensor
                        buf_name = self._key_to_name(key)
                        self.register_buffer(buf_name, cg_tensor, persistent=False)

    def get(self, l1:GTSignature, l2:GTSignature, l3:GTSignature) -> Optional[torch.Tensor]:
        key = (l1.lam,l2.lam,l3.lam)
        if key in self._cache:
            return self._cache[key]
        return None

    def _compute_cg(self, s1,s2,s3) -> torch.Tensor:
        # Real CG by numerical quadrature of the *model's* basis functions on
        # S^{n-1}, self-consistent for every n>=2.  (The old n==2 branch
        # returned scalar ones/zeros with the wrong shapes, e.g. (1,1,1) for
        # the (0,1,1) identity coupling that must be (1,2,2); the resulting
        # implicit einsum broadcasting silently broke rotation equivariance.)
        if self.n==2:
            return _cg_quadrature(self.n, self.max_order, s1, s2, s3)
        if self.n==3:
            return _so3_cg_real(s1.lam[0], s2.lam[0], s3.lam[0])
        # n>=4: GT basis recursion + numerical quadrature on S^{n-1}.
        # (Self-consistent with the edge-harmonic basis by construction.)
        return _cg_quadrature(self.n, self.max_order, s1, s2, s3)


# ── helpers ──────────────────────────────────────────────────────────────────

def _enum_sigs(n:int,max_order:int)->List[GTSignature]:
    k,sigs=n//2,[]
    for l1 in range(max_order+1):
        def _rec(rem,upper,cur):
            if rem==0: sigs.append(GTSignature(tuple(cur),n)); return
            for v in range(0,upper+1): _rec(rem-1,v,cur+[v])
        _rec(k-1,l1,[l1])
    return sigs

def _gegenbauer(order:int,alpha:float,x:torch.Tensor)->torch.Tensor:
    if order==0: return torch.ones_like(x)
    if order==1: return 2*alpha*x
    C0,C1=torch.ones_like(x),2*alpha*x
    for k in range(2,order+1):
        C2=(2*x*(k+alpha-1)*C1-(k+2*alpha-2)*C0)/k; C0,C1=C1,C2
    return C1


# ── SO(3) real CG by numerical quadrature ─────────────────────────────────────

_sph_quad_cache: Dict[Tuple[int,int], Tuple[torch.Tensor, torch.Tensor]] = {}

def _sph_quad_grid(nt: int, nphi: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Spherical quadrature grid (Gauss–Legendre in θ, uniform in φ).

    Returns unit vectors X of shape (nt*nphi, 3) and solid-angle weights w
    of shape (nt*nphi,), exact for products of spherical harmonics up to the
    degree supported by the grid.
    """
    key = (nt, nphi)
    if key in _sph_quad_cache:
        return _sph_quad_cache[key]
    xg, wg = np.polynomial.legendre.leggauss(nt)
    xg = torch.tensor(xg, dtype=torch.float32)
    wg = torch.tensor(wg, dtype=torch.float32)
    theta = torch.acos(xg.clamp(-1 + 1e-6, 1 - 1e-6))
    phi = torch.linspace(0.0, 2.0 * math.pi, nphi + 1)[:-1]
    TH, PH = torch.meshgrid(theta, phi, indexing='ij')
    X = torch.stack([
        torch.sin(TH) * torch.cos(PH),
        torch.sin(TH) * torch.sin(PH),
        torch.cos(TH),
    ], dim=-1).reshape(-1, 3)
    w = (wg[:, None] * (2.0 * math.pi / nphi)).expand(nt, nphi).reshape(-1)
    _sph_quad_cache[key] = (X, w)
    return _sph_quad_cache[key]


# ── General SO(n) real CG by spherical quadrature ─────────────────────────────

def _sphere_grid(n: int, nt: int, nphi: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Product quadrature on S^{n-1} ⊂ R^n.

    Gauss–Legendre in the n-2 polar angles (via u = cos θ ∈ [-1, 1]),
    uniform in the azimuthal angle.  Returns unit vectors X of shape
    (nt^(n-2) * nphi, n) and solid-angle weights w of the same length.
    """
    xg, wg = np.polynomial.legendre.leggauss(nt)
    xg = torch.tensor(xg, dtype=torch.float32)
    wg = torch.tensor(wg, dtype=torch.float32)
    n_polar = n - 2
    thetas = torch.acos(xg.clamp(-1 + 1e-6, 1 - 1e-6))
    phis   = torch.linspace(0.0, 2.0 * math.pi, nphi + 1)[:-1]
    A = torch.meshgrid(*([thetas] * n_polar + [phis]), indexing='ij')
    shape = A[0].shape
    X = torch.zeros(*shape, n, dtype=torch.float32)
    prod_sin = torch.ones(shape)
    for i in range(n_polar):
        X[..., i] = prod_sin * torch.cos(A[i])
        prod_sin = prod_sin * torch.sin(A[i])
    X[..., n - 2] = prod_sin * torch.cos(A[n_polar])
    X[..., n - 1] = prod_sin * torch.sin(A[n_polar])
    W = torch.ones(shape)
    ws = [wg] * n_polar + [torch.full((nphi,), 2.0 * math.pi / nphi)]
    for wv in torch.meshgrid(*ws, indexing='ij'):
        W = W * wv
    return X.reshape(-1, n), W.reshape(-1)


_sphere_quad_cache: Dict[Tuple[int, int, int], Tuple[torch.Tensor, torch.Tensor]] = {}

def _cg_quadrature(n: int, max_order: int, s1: GTSignature, s2: GTSignature,
                   s3: GTSignature) -> torch.Tensor:
    """
    SO(n) real Clebsch–Gordan tensor by direct numerical integration of the
    triple product of the model's GT basis functions over S^{n-1}:

        C_{α β γ} = ∫ Y_{λ1 α} Y_{λ2 β} Y_{λ3 γ} dΩ .

    Because the harmonics are evaluated with the *same* GTBasis the model
    uses for edge features, the coupling is rotation-equivariant by
    construction for every n (the old recursive construction in the complex
    basis did not match the real basis convention and silently broke
    equivariance for n≥4 and for l>1 when n=3).

    The basis functions of a signature with top weight L are polynomials of
    degree L on S^{n-1}, so the integrand has degree ≤ l1+l2+l3 ≤ 3·max_order
    and the quadrature below is exact (Gauss–Legendre integrates polynomials
    of degree < 2·nt exactly).
    """
    nt    = 3 * max_order + n + 8
    nphi  = 4 * (3 * max_order + n) + 16
    key = (n, nt, nphi)
    if key not in _sphere_quad_cache:
        _sphere_quad_cache[key] = _sphere_grid(n, nt, nphi)
    X, w = _sphere_quad_cache[key]
    basis = GTBasis(n, max_order)
    with torch.no_grad():
        Y = basis(X)                                  # (K, num_basis)
    offs: Dict[GTSignature, int] = {}
    off = 0
    for s in basis.signatures:
        offs[s] = off
        off += s.dim()
    Y1 = Y[:, offs[s1]:offs[s1] + s1.dim()]
    Y2 = Y[:, offs[s2]:offs[s2] + s2.dim()]
    Y3 = Y[:, offs[s3]:offs[s3] + s3.dim()]
    return torch.einsum('pi,pj,pk,p->ijk', Y1, Y2, Y3, w).to(torch.float32)


def _so3_cg_real(l1: int, l2: int, l3: int) -> torch.Tensor:
    """
    Real SO(3) Clebsch–Gordan tensor by direct numerical integration of the
    triple product of real spherical harmonics:

        C_{m1 m2 m3} = ∫ Y_{l1 m1} Y_{l2 m2} Y_{l3 m3} dΩ .

    The harmonics are evaluated with ``_real_sph_harm_torch`` — the exact
    convention the model's GTBasis uses for edge features — so the resulting
    coupling is rotation-equivariant *by construction* (the earlier complex-
    basis construction via ``_real2complex`` did not match that convention for
    l > 1 and silently broke equivariance).

    Returns a float32 tensor of shape (2l1+1, 2l2+1, 2l3+1).
    """
    if abs(l1 - l2) > l3 or l3 > l1 + l2 or (l1 + l2 + l3) % 2 == 1:
        return torch.zeros(2*l1+1, 2*l2+1, 2*l3+1)
    l_max = max(l1, l2, l3)
    nt, nphi = 2 * l_max + 8, 4 * l_max + 16
    X, w = _sph_quad_grid(nt, nphi)
    theta = torch.acos(X[:, 2].clamp(-1 + 1e-7, 1 - 1e-7))
    phi   = torch.atan2(X[:, 1], X[:, 0])
    def block(l: int) -> torch.Tensor:
        Y = _real_sph_harm_torch(l, theta, phi)      # (K, (l+1)^2)
        return Y[:, l*l:(l+1)*(l+1)]
    Y1, Y2, Y3 = block(l1), block(l2), block(l3)
    return torch.einsum('pi,pj,pk,p->ijk', Y1, Y2, Y3, w).to(torch.float32)


# ── self-test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== weyl_dim ===")
    for n,lam,exp in [(3,(0,),1),(3,(1,),3),(3,(2,),5),(4,(1,0),4),(4,(1,1),3),(5,(1,0),5)]:
        got=weyl_dim(n,lam)
        print(f"  SO({n}) {lam}: {got}  {'OK' if got==exp else f'FAIL exp={exp}'}")

    print("\n=== GTBasis n=3 ===")
    b3=GTBasis(3,2); dirs=torch.randn(5,3); dirs/=dirs.norm(dim=-1,keepdim=True)
    feat=b3(dirs); print(f"  {feat.shape}  (expected (5,{b3.num_basis}))")

    print("\n=== GTBasis n=4 ===")
    b4=GTBasis(4,1); d4=torch.randn(4,4); d4/=d4.norm(dim=-1,keepdim=True)
    print(f"  {b4(d4).shape}")

    print("\n=== CGCoefficients n=3 ===")
    cg3=CGCoefficients(3,1)
    C=cg3.get(GTSignature((1,),3),GTSignature((1,),3),GTSignature((0,),3))
    print(f"  1@1->0: {C.shape if C is not None else None}  norm={C.norm():.3f}" if C is not None else "  None")

    print("\nDone.")

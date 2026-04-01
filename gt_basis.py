"""
gt_basis.py  —  Gelfand-Tsetlin basis for SO(n).

Provides: GTSignature, weyl_dim, GTBasis, CGCoefficients
"""
from __future__ import annotations
import math, itertools
from typing import List, Tuple, Dict, Optional
import torch, numpy as np
from scipy.special import sph_harm


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
    @property
    def rank(self) -> int:            return self._n // 2
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
    if n <= 2: return 1
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


# ── GTBasis ──────────────────────────────────────────────────────────────────

class GTBasis:
    """
    GT harmonic basis functions on S^{n-1}.
    n=3: real spherical harmonics (scipy). n>=4: Gegenbauer recursion.
    """
    def __init__(self, n: int, max_order: int):
        self.n = n; self.max_order = max_order
        self.signatures: List[GTSignature] = _enum_sigs(n, max_order)
        self.dims:       List[int]         = [s.dim() for s in self.signatures]
        self.num_basis:  int               = sum(self.dims)

    def __call__(self, unit_dirs: torch.Tensor) -> torch.Tensor:
        return self._eval_n3(unit_dirs) if self.n==3 else self._eval_general(unit_dirs)

    def _eval_n3(self, d: torch.Tensor) -> torch.Tensor:
        shape = d.shape[:-1]
        x,y,z = d[...,0], d[...,1], d[...,2]
        theta  = torch.acos(z.clamp(-1+1e-7,1-1e-7))
        phi    = torch.atan2(y,x)
        th_np  = theta.detach().cpu().numpy().reshape(-1)
        ph_np  = phi.detach().cpu().numpy().reshape(-1)
        parts  = []
        for sig in self.signatures:
            l = sig.lam[0]; block=[]
            for m in range(-l,l+1):
                Y = sph_harm(abs(m),l,ph_np,th_np)
                if   m<0: val=math.sqrt(2)*Y.imag
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
        z     = dirs[:,-1].clamp(-1+1e-7,1-1e-7)
        sin_t = torch.sqrt(1-z**2).clamp(min=1e-8)
        omega = dirs[:,:-1]/sin_t.unsqueeze(-1)
        cols  = []
        for mu in sig.restrict():
            Cv = _gegenbauer(sig.lam[0]-mu.lam[0], mu.dim()+(n-3)/2, z)
            Yr = GTBasis(n-1,mu.lam[0])._eval_sig(omega,mu)
            cols.append((Cv*sin_t**mu.lam[0]).unsqueeze(-1)*Yr)
        if not cols: return torch.zeros(M,dim,device=dirs.device)
        res=torch.cat(cols,dim=-1); nrm=res.norm(dim=0,keepdim=True).clamp(min=1e-8)
        return res/nrm


# ── CGCoefficients ───────────────────────────────────────────────────────────

class CGCoefficients:
    """Clebsch-Gordan tensors for SO(n), built recursively and cached."""
    def __init__(self, n: int, max_order: int):
        self.n=n; self.max_order=max_order
        self._cache: Dict[Tuple,torch.Tensor] = {}
        self._precompute()

    def _precompute(self):
        sigs = _enum_sigs(self.n, self.max_order)
        for s1 in sigs:
            for s2 in sigs:
                cap = min(self.max_order, s1.lam[0]+s2.lam[0])
                for s3 in _enum_sigs(self.n, cap):
                    key=(s1.lam,s2.lam,s3.lam)
                    if key not in self._cache:
                        self._cache[key]=self._compute_cg(s1,s2,s3)

    def get(self, l1:GTSignature, l2:GTSignature, l3:GTSignature) -> Optional[torch.Tensor]:
        return self._cache.get((l1.lam,l2.lam,l3.lam))

    def tensor_product_irreps(self, l1:GTSignature, l2:GTSignature) -> List[GTSignature]:
        cap=min(self.max_order,l1.lam[0]+l2.lam[0])
        return [s for s in _enum_sigs(self.n,cap)
                if self._cache.get((l1.lam,l2.lam,s.lam),torch.zeros(1)).norm()>1e-8]

    def _compute_cg(self, s1,s2,s3) -> torch.Tensor:
        d1,d2,d3=s1.dim(),s2.dim(),s3.dim()
        if self.n==2:
            return (torch.ones(1,1,1) if s1.lam[0]+s2.lam[0]==s3.lam[0]
                    else torch.zeros(1,1,1))
        if self.n==3:
            l1,l2,l3=s1.lam[0],s2.lam[0],s3.lam[0]
            Cc=_so3_cg_complex(l1,l2,l3)
            if Cc is None: return torch.zeros(d1,d2,d3)
            U1,U2,U3=_real2complex(l1),_real2complex(l2),_real2complex(l3)
            Cr=np.einsum('am,bn,mno,op->abo',U1.conj(),U2.conj(),Cc,U3).real
            return torch.from_numpy(Cr.astype(np.float32))
        # n>=4: GT recursion
        C=torch.zeros(d1,d2,d3)
        b1,b2,b3=s1.restrict(),s2.restrict(),s3.restrict()
        o1,o2,o3=_branch_offs(s1),_branch_offs(s2),_branch_offs(s3)
        sub=CGCoefficients(self.n-1,self.max_order)
        for i1,mu1 in enumerate(b1):
            w1=_isoscalar(s1,mu1); sl1=slice(o1[i1],o1[i1]+mu1.dim())
            for i2,mu2 in enumerate(b2):
                w2=_isoscalar(s2,mu2); sl2=slice(o2[i2],o2[i2]+mu2.dim())
                for i3,mu3 in enumerate(b3):
                    cg_sub=sub.get(mu1,mu2,mu3)
                    if cg_sub is None: continue
                    w3=_isoscalar(s3,mu3); sl3=slice(o3[i3],o3[i3]+mu3.dim())
                    C[sl1,sl2,sl3]+=w1*w2*w3*cg_sub
        mat=C.reshape(d1*d2,d3)
        if mat.shape[0]>=mat.shape[1] and mat.norm()>1e-10:
            Q,_=torch.linalg.qr(mat); C=Q.reshape(d1,d2,d3)
        return C


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

def _branch_offs(sig:GTSignature)->List[int]:
    offs,pos=[],0
    for mu in sig.restrict(): offs.append(pos); pos+=mu.dim()
    return offs

def _isoscalar(parent:GTSignature,child:GTSignature)->float:
    branches=parent.restrict(); total=sum(b.dim() for b in branches)
    if total==0 or child not in branches: return 0.
    return math.sqrt(child.dim()/total)


# ── SO(3) CG in complex basis ────────────────────────────────────────────────

def _so3_cg_complex(l1:int,l2:int,l3:int):
    """
    Validated Condon-Shortley two-pass recursion.
    Returns numpy complex64 (d1,d2,d3) or None.
    Pass 1: GS top-rung against ALL previous columns (ensures global unitarity).
    Pass 2: J- recursion fills each block; every new column joins GS basis.
    """
    if abs(l1-l2)>l3 or l3>l1+l2: return None
    d1,d2=2*l1+1,2*l2+1
    def idx(l,m): return int(m+l)
    def Jm(l,m):  return math.sqrt(max(l*(l+1)-m*(m-1),0.))
    all_cols: List[np.ndarray]=[]
    tables:   Dict[int,np.ndarray]={}
    for l3c in range(l1+l2,abs(l1-l2)-1,-1):
        d3c=2*l3c+1; C=np.zeros((d1,d2,d3c),dtype=np.complex128)
        pairs=[(m1,l3c-m1) for m1 in range(-l1,l1+1) if -l2<=l3c-m1<=l2]
        v=None
        for m1s,m2s in pairs:
            cand=np.zeros(d1*d2,dtype=np.complex128)
            cand[idx(l1,m1s)*d2+idx(l2,m2s)]=1.
            for u in all_cols: cand-=np.dot(u.conj(),cand)*u
            nc=np.linalg.norm(cand)
            if nc>1e-10: v=cand/nc; break
        if v is None: continue
        best=max(pairs,key=lambda p:abs(v[idx(l1,p[0])*d2+idx(l2,p[1])]))
        c=v[idx(l1,best[0])*d2+idx(l2,best[1])]
        if abs(c)>1e-12: v*=abs(c)/c
        for m1 in range(-l1,l1+1):
            for m2 in range(-l2,l2+1):
                C[idx(l1,m1),idx(l2,m2),idx(l3c,l3c)]=v[idx(l1,m1)*d2+idx(l2,m2)]
        all_cols.append(v.copy())
        for m3 in range(l3c,-l3c,-1):
            den=Jm(l3c,m3)
            if den<1e-14: continue
            for m1 in range(-l1,l1+1):
                for m2 in range(-l2,l2+1):
                    val=0.
                    if -l1<=m1+1<=l1: val+=Jm(l1,m1+1)*C[idx(l1,m1+1),idx(l2,m2),  idx(l3c,m3)]
                    if -l2<=m2+1<=l2: val+=Jm(l2,m2+1)*C[idx(l1,m1),  idx(l2,m2+1),idx(l3c,m3)]
                    C[idx(l1,m1),idx(l2,m2),idx(l3c,m3-1)]=val/den
            nc=C[:,:,idx(l3c,m3-1)].reshape(-1).copy()
            nn=np.linalg.norm(nc)
            if nn>1e-12: all_cols.append(nc/nn)
        tables[l3c]=C
    if l3 not in tables: return None
    return tables[l3].astype(np.complex64)


def _real2complex(l:int)->np.ndarray:
    """Real-to-complex SH basis change, numpy complex128."""
    d=2*l+1; U=np.zeros((d,d),dtype=np.complex128)
    for m in range(-l,l+1):
        i=m+l
        if m<0:   U[i,(-m)+l]=1j/math.sqrt(2); U[i,m+l]=-1j/math.sqrt(2)*((-1)**m)
        elif m>0: U[i,m+l]=1/math.sqrt(2)*((-1)**m); U[i,(-m)+l]=1/math.sqrt(2)
        else:     U[i,l]=1.
    return U


# ── self-test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== weyl_dim ===")
    for n,lam,exp in [(3,(0,),1),(3,(1,),3),(3,(2,),5),(4,(1,0),4),(4,(1,1),3),(5,(1,0),5)]:
        got=weyl_dim(n,lam)
        print(f"  SO({n}) {lam}: {got}  {'OK' if got==exp else f'FAIL exp={exp}'}")

    print("\n=== _so3_cg_complex ===")
    for l1,l2,l3 in [(1,1,0),(1,1,1),(1,1,2),(2,1,1),(2,2,2)]:
        C=_so3_cg_complex(l1,l2,l3)
        if C is None: continue
        d1,d2,d3=2*l1+1,2*l2+1,2*l3+1
        err=abs((C.reshape(d1*d2,d3).conj().T@C.reshape(d1*d2,d3))-np.eye(d3)).max()
        viol=sum(1 for m1 in range(-l1,l1+1) for m2 in range(-l2,l2+1)
                 for m3 in range(-l3,l3+1) if m1+m2!=m3 and abs(C[m1+l1,m2+l2,m3+l3])>1e-6)
        print(f"  <{l1},{l2}|{l3}>: unit_err={err:.1e} msel={viol}")

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

import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
import numpy as np
from datasets.shapes3d import generate_dataset, DATASET_CONFIGS

np.random.seed(42)

def cov_eig_features(pc):
    c = pc.mean(0)
    C = np.cov((pc - c).T)
    w = np.linalg.eigvalsh(C)[::-1]
    w = np.maximum(w, 0)
    s = w.sum() + 1e-12
    return w / s

def global_features(pc):
    c = pc.mean(0)
    d = np.linalg.norm(pc - c, axis=1)
    w = cov_eig_features(pc)
    return np.concatenate([
        w,                                   # normalized covariance spectrum (sorted)
        [d.mean(), d.std(), np.percentile(d, 90)],
        [np.linalg.norm(c)],                 # centroid drift
    ])

for ds in ['shapes3d_complex', 'shapes3d_8way']:
    n_per = 150
    N = 600
    _, _, clean, yc, names = generate_dataset(ds, 0, n_per, N, noise_sigma=0.0, seed=42)
    noisy, yn, _, _, _ = generate_dataset(ds, n_per, 0, N, noise_sigma=0.15, seed=123)
    yc = np.array(yc); yn = np.array(yn)

    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    from sklearn.metrics import accuracy_score

    Xc = np.array([global_features(p) for p in clean])
    Xn = np.array([global_features(p) for p in noisy])

    for label, X, y in [('clean', Xc, yc), ('noisy', Xn, yn)]:
        lda = LDA().fit(X, y)
        acc = accuracy_score(y, lda.predict(X))
        print(f"{ds} [{label}] LDA on global invariants: {acc:.3f}")

    # is torus separable from double_torus under noise?
    if 'torus' in names and 'double_torus' in names:
        i_t = names.index('torus'); i_dt = names.index('double_torus')
        sel = (yn == i_t) | (yn == i_dt)
        lda2 = LDA().fit(Xn[sel], yn[sel])
        acc2 = accuracy_score(yn[sel], lda2.predict(Xn[sel]))
        print(f"  torus vs double_torus under noise: LDA {acc2:.3f} "
              f"(n_torus={ (yn==i_t).sum() }, n_dt={ (yn==i_dt).sum() })")

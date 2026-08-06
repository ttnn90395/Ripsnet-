"""
Synthetic 3D shape generators for point cloud classification.

Shapes are topologically distinct:
  - sphere:      genus 0, 1 component
  - torus:       genus 1, 1 component
  - double_torus: genus 2, 1 component
  - two_spheres: genus 0, 2 components
  - cylinder:    genus 0, 1 component, boundary
  - cube:        genus 0, 1 component, corners
  - knot:        genus 1, 1 component (trefoil)
  - tetrahedron: genus 0, 1 component, flat faces
"""

import numpy as np
from tqdm import tqdm


def _sample_sphere_surface(N, r=1.0, center=(0, 0, 0)):
    """Uniform sampling on a sphere surface via normalisation."""
    pts = np.random.randn(N, 3)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    pts *= r
    return pts + np.array(center)


def _sample_torus_surface(N, R=2.0, r=0.6, center=(0, 0, 0)):
    """Uniform-ish sampling on a torus surface."""
    theta = np.random.uniform(0, 2 * np.pi, N)
    phi = np.random.uniform(0, 2 * np.pi, N)
    x = (R + r * np.cos(phi)) * np.cos(theta)
    y = (R + r * np.cos(phi)) * np.sin(theta)
    z = r * np.sin(phi)
    return np.column_stack([x, y, z]) + np.array(center)


def _sample_double_torus_surface(N, center=(0, 0, 0)):
    """Two overlapping tori sharing a neck, approximated as two side-by-side tori."""
    half = N // 2
    spacing = 3.2
    left = _sample_torus_surface(half, R=1.5, r=0.5, center=(-spacing / 2, 0, 0))
    right = _sample_torus_surface(N - half, R=1.5, r=0.5, center=(spacing / 2, 0, 0))
    return np.vstack([left, right]) + np.array(center)


def _sample_cylinder_surface(N, r=1.0, h=3.0, center=(0, 0, 0)):
    """Sampling on a cylinder surface (top + bottom rims + lateral)."""
    n_rim = N // 4
    n_lat = N - 2 * n_rim
    theta_top = np.random.uniform(0, 2 * np.pi, n_rim)
    theta_bot = np.random.uniform(0, 2 * np.pi, n_rim)
    top = np.column_stack([r * np.cos(theta_top), r * np.sin(theta_top), np.full(n_rim, h / 2)])
    bot = np.column_stack([r * np.cos(theta_bot), r * np.sin(theta_bot), np.full(n_rim, -h / 2)])
    theta_lat = np.random.uniform(0, 2 * np.pi, n_lat)
    z_lat = np.random.uniform(-h / 2, h / 2, n_lat)
    lat = np.column_stack([r * np.cos(theta_lat), r * np.sin(theta_lat), z_lat])
    return np.vstack([top, bot, lat]) + np.array(center)


def _sample_cube_surface(N, side=2.0, center=(0, 0, 0)):
    """Uniform-ish sampling on a cube surface."""
    pts_per_face = N // 6
    remainder = N - 6 * pts_per_face
    half = side / 2
    faces = []
    for _ in range(6):
        pts = np.random.uniform(-half, half, (pts_per_face, 2))
        n = np.random.randint(0, 3)
        zeros = np.zeros((pts_per_face, 1))
        if n == 0:
            col = np.column_stack([np.full(pts_per_face, half * (1 if np.random.rand() > 0.5 else -1)), pts])
        elif n == 1:
            col = np.column_stack([pts[:, 0], np.full(pts_per_face, half * (1 if np.random.rand() > 0.5 else -1)), pts[:, 1]])
        else:
            col = np.column_stack([pts[:, 0], pts[:, 1], np.full(pts_per_face, half * (1 if np.random.rand() > 0.5 else -1))])
        faces.append(col)
    if remainder > 0:
        extra = np.random.uniform(-half, half, (remainder, 3))
        faces.append(extra)
    return np.vstack(faces) + np.array(center)


def _sample_knot_surface(N, R=2.0, r=0.4, center=(0, 0, 0)):
    """Trefoil knot as a thick tube."""
    t = np.random.uniform(0, 2 * np.pi, N)
    noise = np.random.normal(0, r, (N, 3))
    x = R * (np.cos(t) + 2 * np.cos(2 * t))
    y = R * (np.sin(t) - 2 * np.sin(2 * t))
    z = R * (-np.sin(3 * t))
    pts = np.column_stack([x, y, z])
    pts += noise
    centroid = pts.mean(axis=0)
    pts -= centroid
    scale = np.max(np.abs(pts)) + 1e-8
    pts = pts / scale * R
    return pts + np.array(center)


def _sample_tetrahedron_surface(N, side=2.0, center=(0, 0, 0)):
    """Points on the surface of a tetrahedron."""
    import math
    h = side * math.sqrt(2.0 / 3.0)
    r = side / math.sqrt(3.0)
    verts = np.array([
        [r, 0, -h / 4],
        [-r / 2, r * math.sqrt(3) / 2, -h / 4],
        [-r / 2, -r * math.sqrt(3) / 2, -h / 4],
        [0, 0, 3 * h / 4],
    ])
    faces = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
    pts = []
    pts_per_face = N // 4
    remainder = N - 4 * pts_per_face
    for (a, b, c) in faces:
        va, vb, vc = verts[a], verts[b], verts[c]
        s = np.random.uniform(0, 1, pts_per_face)
        t = np.random.uniform(0, 1 - s)
        pts_face = (1 - s - t)[:, None] * va + s[:, None] * vb + t[:, None] * vc
        pts.append(pts_face)
    if remainder > 0:
        a, b, c = faces[0]
        va, vb, vc = verts[a], verts[b], verts[c]
        s = np.random.uniform(0, 1, remainder)
        t = np.random.uniform(0, 1 - s)
        pts.append(((1 - s - t)[:, None] * va + s[:, None] * vb + t[:, None] * vc))
    return np.vstack(pts) + np.array(center)


def _sample_two_spheres(N, r=1.0, center=(0, 0, 0)):
    half = N // 2
    sep = r * 3.5
    s1 = _sample_sphere_surface(half, r=r, center=(-sep / 2, 0, 0))
    s2 = _sample_sphere_surface(N - half, r=r, center=(sep / 2, 0, 0))
    return np.vstack([s1, s2]) + np.array(center)


def _sample_solid_sphere(N, r=1.0, center=(0, 0, 0)):
    """Interior + surface sampling."""
    pts = np.random.randn(N, 3)
    radii = np.random.uniform(0, 1, N) ** (1.0 / 3)
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True) * radii[:, None] * r
    return pts + np.array(center)


def _sample_solid_torus(N, R=2.0, r=0.6, center=(0, 0, 0)):
    """Interior sampling of a torus."""
    theta = np.random.uniform(0, 2 * np.pi, N)
    rho = r * np.sqrt(np.random.uniform(0, 1, N))
    phi = np.random.uniform(0, 2 * np.pi, N)
    x = (R + rho * np.cos(phi)) * np.cos(theta)
    y = (R + rho * np.cos(phi)) * np.sin(theta)
    z = rho * np.sin(phi)
    return np.column_stack([x, y, z]) + np.array(center)


def _add_noise(pts, sigma=0.05):
    return pts + np.random.normal(0, sigma, pts.shape)


SHAPE_GENERATORS_SURFACE = {
    'sphere':        lambda N, **kw: _sample_sphere_surface(N, **kw),
    'torus':         lambda N, **kw: _sample_torus_surface(N, **kw),
    'double_torus':  lambda N, **kw: _sample_double_torus_surface(N, **kw),
    'two_spheres':   lambda N, **kw: _sample_two_spheres(N, **kw),
    'cylinder':      lambda N, **kw: _sample_cylinder_surface(N, **kw),
    'cube':          lambda N, **kw: _sample_cube_surface(N, **kw),
    'knot':          lambda N, **kw: _sample_knot_surface(N, **kw),
    'tetrahedron':   lambda N, **kw: _sample_tetrahedron_surface(N, **kw),
}

SHAPE_GENERATORS_SOLID = {
    'sphere':  lambda N, **kw: _sample_solid_sphere(N, **kw),
    'torus':   lambda N, **kw: _sample_solid_torus(N, **kw),
}

SHAPE_CLASSES = list(SHAPE_GENERATORS_SURFACE.keys())

DATASET_CONFIGS = {
    'shapes3d_topology': {
        'classes': ['sphere', 'torus', 'double_torus', 'two_spheres'],
        'description': 'Topologically distinct: genus 0/1/2 and 2 components',
    },
    'shapes3d_geometry': {
        'classes': ['sphere', 'torus', 'cylinder', 'cube'],
        'description': 'Same topology (genus 0, 1 component) but different geometry',
    },
    'shapes3d_complex': {
        'classes': ['sphere', 'torus', 'double_torus', 'two_spheres', 'knot', 'tetrahedron'],
        'description': '6-class: topology + geometry + knot',
    },
    'shapes3d_8way': {
        'classes': SHAPE_CLASSES,
        'description': 'All 8 shapes',
    },
}


def generate_dataset(dataset_name, N_train_per_class, N_test_per_class, N_points,
                     noise_sigma=0.0, seed=42):
    """
    Generate a 3D point cloud classification dataset.

    Args:
        dataset_name: key in DATASET_CONFIGS
        N_train_per_class: training samples per class
        N_test_per_class: test samples per class
        N_points: points per cloud
        noise_sigma: additive Gaussian noise std (0 = clean)
        seed: random seed

    Returns:
        data_train: list of np.ndarray (N_points, 3)
        y_train: np.ndarray of int labels
        data_test: list of np.ndarray (N_points, 3)
        y_test: np.ndarray of int labels
        class_names: list of str
    """
    rng_state = np.random.get_state()
    np.random.seed(seed)

    cfg = DATASET_CONFIGS[dataset_name]
    class_names = cfg['classes']
    n_classes = len(class_names)

    data_train, y_train = [], []
    data_test, y_test = [], []

    for label_idx, cls_name in enumerate(class_names):
        gen = SHAPE_GENERATORS_SURFACE[cls_name]
        for _ in range(N_train_per_class):
            pc = gen(N_points)
            if noise_sigma > 0:
                pc = _add_noise(pc, sigma=noise_sigma)
            data_train.append(pc.astype(np.float32))
            y_train.append(label_idx)
        for _ in range(N_test_per_class):
            pc = gen(N_points)
            if noise_sigma > 0:
                pc = _add_noise(pc, sigma=noise_sigma)
            data_test.append(pc.astype(np.float32))
            y_test.append(label_idx)

    y_train = np.array(y_train, dtype=np.int64)
    y_test = np.array(y_test, dtype=np.int64)

    perm_train = np.random.permutation(len(data_train))
    perm_test = np.random.permutation(len(data_test))

    data_train = [data_train[i] for i in perm_train]
    y_train = y_train[perm_train]
    data_test = [data_test[i] for i in perm_test]
    y_test = y_test[perm_test]

    np.random.set_state(rng_state)

    return data_train, y_train, data_test, y_test, class_names

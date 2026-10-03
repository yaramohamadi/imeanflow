"""Part 3: a zoo of 2-D source -> target pairs for the Stage-0c toy.

Each cell is a SOURCE distribution and a TARGET distribution made from it by one kind of shift.
Every shape is built at about the ring's scale (overall extent ~3-4, smallest structure ~0.25),
so one MMD sigma set (0.25 / 1 / 4) and one eps_rel mean roughly the same thing in every cell.

A cell provides what `ot_toy_source_t.py` needs:
  source(key, n) -> (points, labels)   labels = source mode index, or zeros if modeless
  target(key, n) -> (points, labels)
  target_centres, target_weights        the target's modes (None for modeless shapes), used
                                        by mode_mi and the mode-weight error

`ring_rot_offset` is the exact distribution pair of EXP-126..135, so that cell doubles as a
reproduction check.
"""

import jax
import jax.numpy as jnp
import numpy as np

SPREAD = 0.25        # blob std of every GMM shape
RING_RADIUS = 3.0


# --------------------------------------------------------------------------- building blocks


def gmm(centres, weights, spread=SPREAD):
    """Sampler for a GMM with fixed centres and weights; labels are the mode index."""
    centres = jnp.asarray(centres, jnp.float32)
    logits = jnp.log(jnp.asarray(weights, jnp.float32))

    def sample(key, n):
        key_mode, key_noise = jax.random.split(key)
        which = jax.random.categorical(key_mode, logits, shape=(n,))
        return centres[which] + spread * jax.random.normal(key_noise, (n, 2)), which

    return sample


def affine(sampler, matrix, offset):
    """Push a sampler's points through x -> A x + b (blobs deform with it)."""
    a = jnp.asarray(matrix, jnp.float32)
    b = jnp.asarray(offset, jnp.float32)

    def sample(key, n):
        x, labels = sampler(key, n)
        return x @ a.T + b, labels

    return sample


def ring_centres(num_modes=6, radius=RING_RADIUS, rotation=0.0, offset=(0.0, 0.0)):
    angles = 2.0 * np.pi * np.arange(num_modes) / num_modes + rotation
    return np.stack([radius * np.cos(angles) + offset[0],
                     radius * np.sin(angles) + offset[1]], 1)


def grid_centres(spacing=2.5):
    axis = spacing * np.array([-1.0, 0.0, 1.0])
    return np.stack(np.meshgrid(axis, axis), -1).reshape(-1, 2)


def spiral(key, n):
    """One arm, 1.5 turns, radius 0.6 -> 3.6; modeless."""
    key_u, key_noise = jax.random.split(key)
    u = jax.random.uniform(key_u, (n,))
    theta = 3.0 * jnp.pi * u + 0.5
    radius = 0.6 + 3.0 * u
    x = jnp.stack([radius * jnp.cos(theta), radius * jnp.sin(theta)], 1)
    return x + 0.15 * jax.random.normal(key_noise, (n, 2)), jnp.zeros((n,), jnp.int32)


def moons(key, n):
    """Two interleaved moons, scaled to the ring's extent; labels = which moon."""
    key_moon, key_theta, key_noise = jax.random.split(key, 3)
    which = jax.random.bernoulli(key_moon, 0.5, (n,)).astype(jnp.int32)
    theta = jnp.pi * jax.random.uniform(key_theta, (n,))
    upper = jnp.stack([jnp.cos(theta), jnp.sin(theta)], 1)
    lower = jnp.stack([1.0 - jnp.cos(theta), 0.5 - jnp.sin(theta)], 1)
    x = jnp.where(which[:, None] == 0, upper, lower) - jnp.asarray([0.5, 0.25])
    return 2.4 * x + 0.2 * jax.random.normal(key_noise, (n, 2)), which


def checkerboard(key, n, cell=1.5):
    """4x4 board on [-3, 3]^2, uniform on the 8 'black' cells; modeless (uniform, not blobs)."""
    key_cell, key_pos = jax.random.split(key)
    rows, cols = np.meshgrid(np.arange(4), np.arange(4), indexing="ij")
    black = np.stack([rows[(rows + cols) % 2 == 0], cols[(rows + cols) % 2 == 0]], 1)
    corners = jnp.asarray(-3.0 + cell * black[:, ::-1], jnp.float32)   # (x, y) lower-left
    which = jax.random.randint(key_cell, (n,), 0, corners.shape[0])
    return corners[which] + cell * jax.random.uniform(key_pos, (n, 2)), \
        jnp.zeros((n,), jnp.int32)


# --------------------------------------------------------------------------- the shifts


SHIFT_SCALE = (1.3 * np.eye(2), (1.5, -1.0))             # scale x1.3, then translate
ANISO_SHEAR = (np.array([[1.4, 0.5], [0.0, 0.7]]), (0.0, 0.0))


class Cell:
    def __init__(self, name, shape, shift, source, target, centres=None, weights=None):
        self.name, self.shape, self.shift = name, shape, shift
        self.source, self.target = source, target
        self.target_centres = None if centres is None else np.asarray(centres, np.float64)
        self.target_weights = None if weights is None else np.asarray(weights, np.float64)


def _ring_cells():
    src_c = ring_centres()
    src = gmm(src_c, np.full(6, 1 / 6))
    uniform6 = np.full(6, 1 / 6)
    cells = {}

    # the EXP-126..135 pair: centres rotated 30 deg, radius x1.2, offset +1; blobs unchanged
    c = ring_centres(radius=RING_RADIUS * 1.2, rotation=0.5236, offset=(1.0, 0.0))
    cells["ring_rot_offset"] = Cell("ring_rot_offset", "ring", "rotate+offset", src,
                                    gmm(c, uniform6), c, uniform6)

    a, b = SHIFT_SCALE
    cells["ring_shift_scale"] = Cell("ring_shift_scale", "ring", "shift+scale", src,
                                     affine(src, a, b), src_c @ a.T + b, uniform6)
    a, b = ANISO_SHEAR
    cells["ring_aniso_shear"] = Cell("ring_aniso_shear", "ring", "anisotropic+shear", src,
                                     affine(src, a, b), src_c @ a.T + b, uniform6)

    # mode occlusion: the target keeps 4 of the 6 source modes, unchanged in place
    w = np.array([0.25, 0.25, 0.25, 0.25, 0.0, 0.0])
    keep = w > 0
    cells["ring_occlude"] = Cell("ring_occlude", "ring", "mode occlusion (6 -> 4)", src,
                                 gmm(src_c[keep], w[keep]), src_c[keep], w[keep])

    # mode adding: the 6 source modes plus 2 new ones at 30 and 210 degrees, all equal weight
    extra = ring_centres(num_modes=2, rotation=np.pi / 6)
    c = np.concatenate([src_c, extra])
    cells["ring_add"] = Cell("ring_add", "ring", "mode adding (6 -> 8)", src,
                             gmm(c, np.full(8, 1 / 8)), c, np.full(8, 1 / 8))

    # mode reweighting: same 6 places, two modes take 60% of the mass
    w = np.array([0.3, 0.3, 0.1, 0.1, 0.1, 0.1])
    cells["ring_reweight"] = Cell("ring_reweight", "ring", "mode reweighting", src,
                                  gmm(src_c, w), src_c, w)
    return cells


def _shape_cells():
    a, b = SHIFT_SCALE
    grid_c = grid_centres()
    grid_src = gmm(grid_c, np.full(9, 1 / 9))
    cells = {"grid_shift_scale": Cell("grid_shift_scale", "3x3 grid", "shift+scale", grid_src,
                                      affine(grid_src, a, b), grid_c @ a.T + b,
                                      np.full(9, 1 / 9))}
    for name, shape, sampler in [("spiral", "spiral", spiral), ("moons", "two moons", moons),
                                 ("checker", "checkerboard", checkerboard)]:
        cells[f"{name}_shift_scale"] = Cell(f"{name}_shift_scale", shape, "shift+scale",
                                            sampler, affine(sampler, a, b))
    return cells


CELLS = {**_ring_cells(), **_shape_cells()}


# --------------------------------------------------------------------------- mode metrics


def mode_weight_error(outputs, centres, weights):
    """Total variation between the output's per-mode share (nearest centre) and the true one.

    0 = every mode holds exactly its share; 1 = all mass in the wrong modes.
    """
    if centres is None:
        return float("nan")
    out = np.asarray(outputs, np.float64)
    assigned = np.argmin(((out[:, None, :] - centres[None]) ** 2).sum(-1), 1)
    share = np.bincount(assigned, minlength=centres.shape[0]) / out.shape[0]
    return float(0.5 * np.abs(share - weights).sum())

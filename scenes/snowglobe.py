"""A snow globe on the robot's back, over any other scene -- for video.

Pick the world with `--snowglobe-base` (default `studio`). The robot keeps its
own shell, which holds the electronics; the globe swells out of the top of it,
sunk to below its middle, in snow that covers the shell and banks up against the
glass. Inside is a snowy village under mountains: two houses with
lit windows, pine trees, a snowman and a street lamp. Flat flakes sit just
inside the glass and a few hang in the air. The rest of the robot is ice blue,
and icicles hang from the snow's edge, where the shell's top turns down its side.

Every limb keeps one solid colour, as a `.skin` carries it; only geometry fixed
to the shell is added.

**Visual only**, as `ninja` is: every added geom is `contype=0, conaffinity=0,
density=0`, so mass, contacts and the policy's observation are unchanged. The
snow and the icicles are fitted to the shell mesh by ray casts.
"""

from __future__ import annotations

import dataclasses
import functools
import os

import numpy as np

from .ninja import GREY_PART, PAD_PART, RED_PART, _add, _paint, _ribbon, _shell, _top
from .registry import Scene, load

BASE_ENV = "MJRL_SNOWGLOBE_BASE"

R = 0.090                    # globe radius: wider than the shell's top, overhanging it
RIN = 0.985 * R              # props stay inside this
CENTRE_XY = (-0.020, 0.0)    # the glass's widest point just reaches the display's top edge
GLOBE_RISE = 0.38            # centre above the shell, in R: the shell hides the rest
CONTACT = R * np.sqrt(1 - GLOBE_RISE**2)   # where the glass meets the shell

# Snow on the shell: depth, how high it banks against the glass and how far out that
# reaches, and where it stops. The bank stays below the snow inside the globe.
COVER_DEPTH, SNOW_BANK, DRIFT_REACH = 0.003, 0.08 * R, 0.015
DISPLAY_X = 0.062
ICICLE_GAP = (0.0022, 0.0035)            # spacing range along the edge
ICICLE_CLEAR = 0.0004                    # gap left between an icicle and the robot's side

GLASS = (0.85, 0.93, 1.0, 0.13)
SNOW = (0.95, 0.97, 1.0, 1.0)
ROCK = (0.36, 0.42, 0.55, 1.0)
PINE = (0.07, 0.27, 0.15, 1.0)
BARK = (0.24, 0.13, 0.07, 1.0)
ROOF = (0.20, 0.11, 0.08, 1.0)
TIMBER, BRICK = (0.50, 0.30, 0.16, 1.0), (0.55, 0.20, 0.14, 1.0)
IRON = (0.08, 0.08, 0.09, 1.0)
CARROT, COAL, SCARF = (0.95, 0.45, 0.08, 1.0), (0.04, 0.04, 0.05, 1.0), (0.78, 0.06, 0.07, 1.0)
# The robot's own colours, recoloured; see `ninja.LOOKS` for the scheme.
ICY = {
    RED_PART: (0.60, 0.80, 0.94, 1.0),
    GREY_PART: (0.86, 0.91, 0.96, 1.0),
    PAD_PART: (0.55, 0.62, 0.70, 1.0),
    "base_link_visual": (0.06, 0.10, 0.18, 1.0),
}

SPECKS, FLAKES, SEED = 850, 150, 11

# Layout in units of R from the globe's centre; +x is the robot's front.
MOUNTAINS = ((-0.52, -0.28, 0.34, 0.62), (-0.48, 0.30, 0.32, 0.52), (-0.68, 0.0, 0.24, 0.42))
HOUSES = ((-0.10, -0.30, 0.17, 0.14, 0.11, 0.12, TIMBER),
          (-0.16, 0.30, 0.14, 0.12, 0.10, 0.10, BRICK))
TREES = ((0.18, -0.62, 0.36), (0.10, 0.64, 0.32), (-0.40, -0.62, 0.30), (-0.42, 0.60, 0.28),
         (0.55, -0.40, 0.22))
SNOWMAN_UV = (0.42, 0.22)
LAMP_UV = (0.30, -0.10)

add = functools.partial(_add, prefix="globe")


def cli_args(group):
    """`--snowglobe-base`; see `scenes.load_cli_args`."""
    group.add_argument(
        "--snowglobe-base", default=os.environ.get(BASE_ENV, "studio"), metavar="NAME",
        help="the registered scene the snow globe sits in (default studio)",
    )
    return ("snowglobe_base",)


def scene(snowglobe_base: str = "studio") -> Scene:
    if snowglobe_base == "snowglobe":
        raise ValueError("--snowglobe-base must name another scene, not snowglobe itself")
    base = load(snowglobe_base)

    def decorate(spec):
        if base.decorate is not None:
            base.decorate(spec)
        dress(spec)

    return dataclasses.replace(base, decorate=decorate)


def dress(spec) -> None:
    """Sink the globe into the shell and fill it. No-op on a world with no robot."""
    import mujoco

    torso = next((b for b in spec.bodies if b.name.split("/")[-1] == "base_link"), None)
    if torso is None:
        return
    _materials(spec)
    for geom in spec.geoms:
        if geom.name.endswith("_visual"):
            colour = ICY.get(geom.name.split("/")[-1],
                             ICY.get(tuple(round(float(c), 3) for c in geom.rgba[:3])))
            if colour is not None:
                _paint(geom, colour)

    # The shell's mean level under the glass's contact ring; the snow fills where it is lower.
    th = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    ring = np.column_stack([np.cos(th), np.sin(th)]) * CONTACT + CENTRE_XY
    under = [hit[0][2] for hit in map(_top, ring) if hit is not None]  # past the edge: none
    level = np.mean(under)
    c = np.array([*CENTRE_XY, level + GLOBE_RISE * R])
    edge, th = _cover(spec, torso, level)
    _icicles(spec, torso, edge, th, np.random.default_rng(SEED + 1))
    add(torso, "glass", mujoco.mjtGeom.mjGEOM_SPHERE, (R,), "globe_glass", pos=c)

    rng = np.random.default_rng(SEED)
    _floor(spec, torso, c)
    for k, (u, v, radius, height) in enumerate(MOUNTAINS):
        _mountain(spec, torso, f"mountain{k}", c, u, v, radius, height, rng)
    for k, house in enumerate(HOUSES):
        _house(spec, torso, f"house{k}", c, *house)
    for k, (u, v, h) in enumerate(TREES):
        _tree(spec, torso, f"tree{k}", _at(c, u, v), h * R, rng, star=k == 0)
    _snowman(spec, torso, _at(c, *SNOWMAN_UV))
    _lamp(spec, torso, _at(c, *LAMP_UV))
    _specks(torso, c, rng)
    _flakes(torso, c, rng)


def _materials(spec):

    spec.add_material(name="globe_glass", rgba=GLASS, specular=1.0, shininess=1.0)
    spec.add_material(name="globe_window", rgba=(1.0, 0.78, 0.35, 1.0), emission=1.0)
    spec.add_material(name="globe_star", rgba=(1.0, 0.85, 0.3, 1.0), emission=1.0)
    spec.add_material(name="globe_brass", rgba=(0.80, 0.64, 0.32, 1.0), specular=1.0,
                      shininess=0.85)
    spec.add_material(name="globe_icicle", rgba=(0.82, 0.93, 1.0, 0.8), specular=1.0,
                      shininess=1.0)



def _mesh(spec, body, name, verts, faces, outward, colour, prefix="globe"):
    """A mesh from explicit faces, each wound to face along `outward`."""
    import mujoco

    v, f = np.asarray(verts, float), np.array(faces, int)
    n = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    flip = np.sum(n * np.asarray(outward, float), axis=1) < 0
    f[flip] = f[flip][:, ::-1]
    mesh = spec.add_mesh(name=f"{prefix}_{name}")
    mesh.uservert = v.flatten().tolist()
    mesh.userface = f.flatten().tolist()
    return _add(body, name, mujoco.mjtGeom.mjGEOM_MESH, (), colour, mesh=f"{prefix}_{name}",
                prefix=prefix)


def _hull(spec, body, name, verts, colour, prefix="globe"):
    """A convex solid through `verts`; MuJoCo builds the hull of a faceless mesh."""
    import mujoco

    mesh = spec.add_mesh(name=f"{prefix}_{name}")
    mesh.uservert = np.asarray(verts, float).flatten().tolist()
    return _add(body, name, mujoco.mjtGeom.mjGEOM_MESH, (), colour, mesh=f"{prefix}_{name}",
                prefix=prefix)




def _cover(spec, torso, level, nt=120, ns=48):
    """Snow over the shell's top, thinning to a lip at its edge and banked against the glass.

    The edge is the shell's outline seen from above -- its widest point, where icicles
    form -- and stops short of the display. The snow is laid along the shell's normal, so
    it keeps its depth on the steep sides. Returns the edge's top points and their
    bearings from the shell's centre.
    """
    m, d, geoms = _shell()
    centre = d.geom_xpos[geoms[0]][:2]
    th = 2 * np.pi * np.arange(nt) / nt
    reach = np.zeros(nt)
    for j, t in enumerate(th):
        u = np.array([np.cos(t), np.sin(t)])

        def stops(rho):
            xy = centre + rho * u
            hit, inside = _top(xy), _top(xy - 0.001 * u)
            # Off the outline, or dropped down a hole onto something lower.
            return (hit is None or inside is None or inside[0][2] - hit[0][2] > 0.004
                    or xy[0] > DISPLAY_X)

        for rho in np.arange(0.002, 0.15, 0.001):
            if stops(rho):
                break
            reach[j] = rho
        # Bisect the last step: the sides are steep enough that 1 mm is 5 mm of height.
        lo, hi = reach[j], reach[j] + 0.001
        for _ in range(8):
            mid = (lo + hi) / 2
            lo, hi = (lo, mid) if stops(mid) else (mid, hi)
        reach[j] = lo
    # Smooth out notches without ever reaching past where the shell allows: a min, then a mean.
    reach = np.min([np.roll(reach, k) for k in range(-4, 5)], axis=0)
    reach = np.mean([np.roll(reach, k) for k in range(-4, 5)], axis=0)

    def surface(xy, s):
        p, n = _top(xy)
        depth = COVER_DEPTH * (0.35 + 0.65 * min(1.0, (1 - s) / 0.15) ** 0.5)
        gap = np.linalg.norm(xy - CENTRE_XY) - CONTACT
        ramp = np.clip(1 - abs(gap) / DRIFT_REACH, 0.0, 1.0)  # a ring, under the inner snow
        drift = max(0.0, level + SNOW_BANK - p[2] - depth) * ramp * ramp * (3 - 2 * ramp)
        return p - 0.0008 * n, p + depth * n + [0.0, 0.0, drift]

    top, bottom = [], []
    for i in range(ns + 1):
        s = i / ns
        for j in range(nt if i else 1):
            xy = centre + s * reach[j] * np.array([np.cos(th[j]), np.sin(th[j])])
            under, snow = surface(xy, s)
            top.append(snow)
            bottom.append(under)
    verts = np.array(top + bottom)
    half = len(top)

    def ring(i, j):
        return 0 if i == 0 else 1 + (i - 1) * nt + j % nt

    faces, outward = [], []
    for i in range(ns):
        for j in range(nt):
            if i == 0:
                tris = [(0, ring(1, j), ring(1, j + 1))]
            else:
                tris = [(ring(i, j), ring(i, j + 1), ring(i + 1, j)),
                        (ring(i + 1, j), ring(i, j + 1), ring(i + 1, j + 1))]
            for t in tris:
                faces += [t, tuple(k + half for k in t)]
                outward += [(0.0, 0.0, 1.0), (0.0, 0.0, -1.0)]
    for j in range(nt):
        a, b = ring(ns, j), ring(ns, j + 1)
        out = (np.cos(th[j]), np.sin(th[j]), 0.0)
        faces += [(a, b, a + half), (a + half, b, b + half)]
        outward += [out, out]
    _mesh(spec, torso, "cover", verts, faces, outward, SNOW)
    return verts[half - nt:half], th


def _tube(spec, body, name, path, radii, colour, sides=8, prefix="globe"):
    """A tube through `path`, `radii` across; a radius of 0 closes that end in a point."""
    p, r = np.asarray(path, float), np.asarray(radii, float)
    t = np.gradient(p, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    # Carry the cross-section frame along the path, so the tube never twists or folds.
    a = np.zeros_like(t)
    a[0] = np.cross(t[0], np.eye(3)[np.argmin(np.abs(t[0]))])
    for k in range(len(t)):
        if k:
            a[k] = a[k - 1] - (a[k - 1] @ t[k]) * t[k]
        a[k] /= np.linalg.norm(a[k])
    b = np.cross(t, a)
    ang = 2 * np.pi * np.arange(sides) / sides
    verts, rows = [], []
    for k in range(len(p)):
        ring = [p[k]] if r[k] == 0 else p[k] + r[k] * (np.outer(np.cos(ang), a[k])
                                                       + np.outer(np.sin(ang), b[k]))
        rows.append(list(range(len(verts), len(verts) + len(ring))))
        verts.extend(ring)
    faces, outward = [], []
    for k in range(len(p) - 1):
        A, B = rows[k], rows[k + 1]
        axis = (t[k] + t[k + 1]) / np.linalg.norm(t[k] + t[k + 1])
        for j in range(sides):
            jj = (j + 1) % sides
            if len(A) == 1:
                tris = [(A[0], B[j], B[jj])]
            elif len(B) == 1:
                tris = [(A[j], A[jj], B[0])]
            else:
                tris = [(A[j], A[jj], B[j]), (B[j], A[jj], B[jj])]
            for tri in tris:
                out = np.mean([verts[i] for i in tri], axis=0) - (p[k] + p[k + 1]) / 2
                faces.append(tri)
                outward.append(out - (out @ axis) * axis)
    for end, sign in ((0, -1.0), (len(p) - 1, 1.0)):
        if len(rows[end]) > 1:
            centre = len(verts)
            verts.append(p[end])
            for j in range(sides):
                faces.append((centre, rows[end][j], rows[end][(j + 1) % sides]))
                outward.append(sign * t[end])
    return _mesh(spec, body, name, verts, faces, outward, colour, prefix)


def _side(pnt, vec):
    """Where a ray first meets the shell, display or lower body, and the normal facing it."""
    import mujoco

    m, d, geoms = _shell()
    pnt, vec = np.asarray(pnt, float), np.asarray(vec, float)
    best = None
    for g in (*geoms, m.geom("base_link_visual").id):
        n = np.zeros(3)
        dist = mujoco.mj_rayMesh(m, d, g, pnt, vec, n)
        if dist >= 0 and (best is None or dist < best[0]):
            best = (dist, -n if n @ vec > 0 else n)
    return None if best is None else (pnt + best[0] * vec, best[1])


def _icicles(spec, torso, edge, th, rng):
    """Icicles hanging from the snow's edge, where the shell's top turns down its side.

    None hangs over the display.
    """
    m, d, geoms = _shell()
    centre = d.geom_xpos[geoms[0]][:2]
    ok = edge[:, 0] < DISPLAY_X - 0.012
    runs, run = [], []
    for j in np.roll(np.arange(len(edge)), -int(np.argmin(ok))):
        if ok[j]:
            run.append(j)
        elif run:
            runs.append(run)
            run = []
    if run:
        runs.append(run)

    placed = 0
    for run in (r for r in runs if len(r) > 2):
        lip = edge[run] - [0.0, 0.0, 0.0005]     # just under the snow's lip
        bearing = np.unwrap(th[run])
        along = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(lip, axis=0), axis=1))])
        at = rng.uniform(*ICICLE_GAP) / 2
        while at < along[-1] - ICICLE_GAP[0]:
            root = np.array([np.interp(at, along, lip[:, k]) for k in range(3)])
            bear = np.interp(at, along, bearing)
            _icicle(spec, torso, f"icicle{placed}", root, bear, centre, rng)
            placed += 1
            at += rng.uniform(*ICICLE_GAP)


def _icicle(spec, torso, name, root, bearing, centre, rng):
    """One icicle hanging straight down from `root`, thick at the root, tapering to a point.

    It stands off the robot's side by its radius and ICICLE_CLEAR, and is shortened where
    the side leans out under it; one that would be under 3 mm is left out.
    """
    length = 0.004 + 0.010 * rng.beta(1.3, 2.5)
    if rng.random() < 0.15:
        length = rng.uniform(0.014, 0.024)
    r0 = 0.0006 + 0.05 * length + rng.uniform(0.0, 0.0004)
    u = np.array([np.cos(bearing), np.sin(bearing), 0.0])
    s = np.linspace(0.0, 1.0, 14)
    radii = r0 * (1 - s) ** 1.2

    def wall(z):
        hit = _side([*centre, z] + 0.3 * u, -u)
        return 0.0 if hit is None else np.linalg.norm(hit[0][:2] - centre)

    rho = max(np.linalg.norm(root[:2] - centre), wall(root[2]) + r0 + ICICLE_CLEAR)
    while length >= 0.003:
        z = root[2] - s * length
        if all(rho >= wall(zk) + rk + ICICLE_CLEAR for zk, rk in zip(z, radii)):
            break
        length *= 0.8
    else:
        return
    path = np.column_stack([np.full(len(s), centre[0] + rho * u[0]),
                            np.full(len(s), centre[1] + rho * u[1]), z])
    _tube(spec, torso, name, path, radii, "globe_icicle", sides=10)



def _snow_z(x, y):
    """Height of the snow inside the globe, relative to its centre; banked at the glass."""
    rho, th = np.hypot(x, y) / R, np.arctan2(y, x)
    return R * (-0.30 + 0.10 * rho**4 + 0.015 * rho**2 * np.sin(3 * th + 1.0)
                + 0.01 * rho**3 * np.cos(5 * th))


def _at(c, u, v, dz=0.0):
    """The snow surface at (u, v) R from the globe's centre, in the torso frame."""
    return c + [u * R, v * R, _snow_z(u * R, v * R) + dz]


def _inside(c, p, margin=0.9):
    """`p` pulled in horizontally until it is inside the glass."""
    rel = np.asarray(p, float) - c
    limit = margin * np.sqrt(max(RIN**2 - rel[2] ** 2, 0.0))
    rho = np.hypot(rel[0], rel[1])
    if rho > limit:
        rel[:2] *= limit / rho
    return c + rel


def _floor(spec, torso, c, nt=72, ns=20, nc=8):
    """The snow: a banked surface on top, filling the glass below it."""
    th = 2 * np.pi * np.arange(nt) / nt
    rim = np.full(nt, 0.9 * R)
    for _ in range(8):
        rim = 0.985 * np.sqrt(RIN**2 - _snow_z(rim * np.cos(th), rim * np.sin(th)) ** 2)
    verts = [[0.0, 0.0, _snow_z(0.0, 0.0)]]
    for i in range(1, ns + 1):
        x, y = i / ns * rim * np.cos(th), i / ns * rim * np.sin(th)
        verts += np.column_stack([x, y, _snow_z(x, y)]).tolist()
    rim_z = np.array(verts[-nt:])[:, 2]
    for k in range(1, nc + 1):
        phi = np.arcsin(rim_z / RIN) + (-np.pi / 2 + 0.2 - np.arcsin(rim_z / RIN)) * k / nc
        verts += (0.985 * RIN * np.column_stack(
            [np.cos(phi) * np.cos(th), np.cos(phi) * np.sin(th), np.sin(phi)])).tolist()
    verts.append([0.0, 0.0, -0.985 * RIN])
    verts = np.array(verts)

    def ring(i, j):
        return 1 + (i - 1) * nt + j % nt

    faces, outward = [], []
    up = np.array([0.0, 0.0, 1.0])
    for j in range(nt):
        faces.append((0, ring(1, j), ring(1, j + 1)))
        outward.append(up)
    for i in range(1, ns + nc):
        for j in range(nt):
            quad = [(ring(i, j), ring(i, j + 1), ring(i + 1, j)),
                    (ring(i + 1, j), ring(i, j + 1), ring(i + 1, j + 1))]
            for t in quad:
                faces.append(t)
                outward.append(up if i < ns else verts[list(t)].mean(axis=0))
    pole = len(verts) - 1
    for j in range(nt):
        t = (ring(ns + nc, j), ring(ns + nc, j + 1), pole)
        faces.append(t)
        outward.append(verts[list(t)].mean(axis=0))
    _mesh(spec, torso, "floor", verts + c, faces, outward, SNOW)


def _cone(spec, body, name, apex, ring, base, colour, prefix="globe"):
    """Faces from `apex` and from `base` to a closed `ring`: a cone with any outline."""
    apex, base, ring = np.asarray(apex, float), np.asarray(base, float), np.asarray(ring, float)
    axis = (apex - base) / np.linalg.norm(apex - base)
    verts = np.vstack([apex, base, ring])
    faces, outward = [], []
    for i in range(len(ring)):
        j = (i + 1) % len(ring)
        rel = (apex + ring[i] + ring[j]) / 3 - base
        faces += [(0, 2 + i, 2 + j), (1, 2 + j, 2 + i)]
        outward += [rel - (rel @ axis) * axis, -axis]
    return _mesh(spec, body, name, verts, faces, outward, colour, prefix)


def _snowcap(spec, body, name, apex, ring, fractions, colour=SNOW, push=0.06):
    """Snow over the top of a cone: its ring partway down each edge, pushed off the faces."""
    apex = np.asarray(apex, float)
    cap = apex + np.asarray(fractions)[:, None] * (np.asarray(ring) - apex)
    cap[:, :2] += push * (cap[:, :2] - apex[:2])
    depth = apex[2] - cap[:, 2].mean()
    _cone(spec, body, name, apex + [0.0, 0.0, 0.08 * depth], cap,
          apex - [0.0, 0.0, 1.3 * depth], colour)


def _mountain(spec, torso, name, c, u, v, radius, height, rng):
    """A rock peak with a ragged snow cap, its foot sunk in the snow."""
    n = 11
    th = 2 * np.pi * (np.arange(n) + rng.uniform(-0.3, 0.3, n)) / n
    rr = radius * R * rng.uniform(0.75, 1.1, n)
    ring = []
    for t, r in zip(th, rr):
        x, y = u * R + r * np.cos(t), v * R + r * np.sin(t)
        ring.append(_inside(c, c + [x, y, _snow_z(x, y) - 0.03 * R]))
    base = _at(c, u, v, -0.03 * R)
    apex = base + [rng.uniform(-0.1, 0.1) * radius * R, rng.uniform(-0.1, 0.1) * radius * R,
                   height * R]
    _cone(spec, torso, name, apex, ring, base, ROCK)
    fractions = np.where(np.arange(n) % 2, 0.30, 0.46) + rng.uniform(-0.05, 0.05, n)
    _snowcap(spec, torso, f"{name}_snow", apex, ring, fractions, push=0.05)


def _tree(spec, torso, name, base, h, rng, star=False):
    """A trunk under four ragged tiers, each with snow on its upper branches."""
    import mujoco

    trunk = 0.16 * h
    add(torso, f"{name}_trunk", mujoco.mjtGeom.mjGEOM_CYLINDER, (0.06 * h,), BARK,
        fromto=np.concatenate([base - [0.0, 0.0, 0.02 * h], base + [0.0, 0.0, trunk + 0.1 * h]]))
    tier, branches = 0.38 * h, 7
    for k in range(4):
        z0 = base[2] + trunk + k * (h - trunk - tier) / 3
        r = 0.42 * h * (1 - 0.2 * k)
        phase = rng.uniform(0, 2 * np.pi)
        th = phase + np.pi * np.arange(2 * branches) / branches
        tips = np.arange(2 * branches) % 2 == 0
        rr = np.where(tips, r, 0.68 * r)
        ring = np.column_stack([base[0] + rr * np.cos(th), base[1] + rr * np.sin(th),
                                z0 - np.where(tips, 0.12 * tier, 0.0)])
        apex = np.array([base[0], base[1], z0 + tier])
        _cone(spec, torso, f"{name}_tier{k}", apex, ring,
              [base[0], base[1], z0 + 0.25 * tier], PINE)
        _snowcap(spec, torso, f"{name}_snow{k}", apex, ring,
                 np.where(tips, 0.5, 0.36) + rng.uniform(-0.04, 0.04, len(ring)))
    add(torso, f"{name}_drift", mujoco.mjtGeom.mjGEOM_ELLIPSOID, (0.3 * h, 0.3 * h, 0.03 * h),
        SNOW, pos=base)
    if star:
        top = base + [0.0, 0.0, h + 0.03 * h]
        points = top + 0.07 * h * np.vstack([np.eye(3), -np.eye(3)])
        _hull(spec, torso, f"{name}_star", points, "globe_star")


def _slab(torso, name, p0, p1, half_width, half_thick, colour, up):
    """A box from p0 to p1, `half_width` across, lying on the side facing `up`."""
    import mujoco

    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    t = (p1 - p0) / np.linalg.norm(p1 - p0)
    w = np.cross(up, t)
    w /= np.linalg.norm(w)
    n = np.cross(t, w)
    if n @ up < 0:
        n, w = -n, -w
    return add(torso, name, mujoco.mjtGeom.mjGEOM_BOX,
               (np.linalg.norm(p1 - p0) / 2, half_width, half_thick), colour,
               pos=(p0 + p1) / 2 + half_thick * n, mat=np.column_stack([t, w, n]))


def _house(spec, torso, name, c, u, v, hx, hy, hz, ridge, wall):
    """Walls, gables, a snow-laden roof, a chimney, a door and lit windows; gable to the front."""
    import mujoco

    box, cyl = mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_CYLINDER
    hx, hy, hz, ridge = hx * R, hy * R, hz * R, ridge * R
    x, y = c[0] + u * R, c[1] + v * R
    z0 = min(_at(c, u + sx * hx / R, v + sy * hy / R)[2] for sx in (-1, 1) for sy in (-1, 1))
    z0 -= 0.002
    top = z0 + 2 * hz
    add(torso, name, box, (hx, hy, hz), wall, pos=(x, y, z0 + hz))
    _hull(spec, torso, f"{name}_gable",
          [(x + sx * hx, y + sy * hy, top) for sx in (-1, 1) for sy in (-1, 1)]
          + [(x + sx * hx, y, top + ridge) for sx in (-1, 1)], wall)

    # Roof boards and a thick snow layer on each slope, with rounded snow at the eaves.
    eave = 0.25 * hy
    half_len = hx + 0.15 * hy
    for side in (-1, 1):
        crest = np.array([x, y, top + ridge])
        edge = np.array([x, y + side * (hy + eave), top - eave * ridge / hy])
        up = np.array([0.0, side * ridge, hy])
        _slab(torso, f"{name}_roof{side}", edge, crest, half_len, 0.012 * R, ROOF, up)
        n = np.cross([1.0, 0.0, 0.0], crest - edge)
        n = n / np.linalg.norm(n) * np.sign(np.cross([1.0, 0.0, 0.0], crest - edge) @ up)
        _slab(torso, f"{name}_snow{side}", edge + 0.024 * R * n, crest + 0.024 * R * n,
              half_len + 0.004 * R, 0.02 * R, SNOW, up)
        lip = edge + 0.035 * R * n
        add(torso, f"{name}_eave{side}", mujoco.mjtGeom.mjGEOM_CAPSULE, (0.03 * R,), SNOW,
            fromto=(x - half_len, lip[1], lip[2], x + half_len, lip[1], lip[2]))
    ridge_z = top + ridge + 0.05 * R
    add(torso, f"{name}_ridge", mujoco.mjtGeom.mjGEOM_CAPSULE, (0.03 * R,), SNOW,
        fromto=(x - half_len, y, ridge_z, x + half_len, y, ridge_z))

    cx, cy = x - 0.45 * hx, y + 0.5 * hy
    chimney_top = top + ridge + 0.08 * R
    add(torso, f"{name}_chimney", box, (0.035 * R, 0.035 * R, (chimney_top - top) / 2), BARK,
        pos=(cx, cy, (top + chimney_top) / 2))
    add(torso, f"{name}_chimney_snow", mujoco.mjtGeom.mjGEOM_ELLIPSOID,
        (0.045 * R, 0.045 * R, 0.018 * R), SNOW, pos=(cx, cy, chimney_top))

    door_h = 1.1 * hz
    add(torso, f"{name}_door", box, (0.0003, 0.25 * hy, door_h / 2), BARK,
        pos=(x + hx + 0.0003, y, z0 + 0.002 + door_h / 2))
    add(torso, f"{name}_knob", mujoco.mjtGeom.mjGEOM_SPHERE, (0.012 * R,), "globe_brass",
        pos=(x + hx + 0.0008, y + 0.15 * hy, z0 + 0.002 + 0.5 * door_h))
    gz = top + 0.4 * ridge
    add(torso, f"{name}_round_frame", cyl, (0.3 * ridge, 0.0003), ROOF,
        fromto=(x + hx, y, gz, x + hx + 0.0005, y, gz))
    add(torso, f"{name}_round", cyl, (0.22 * ridge, 0.0003), "globe_window",
        fromto=(x + hx + 0.0002, y, gz, x + hx + 0.0007, y, gz))

    wz, ww, wh = z0 + 1.15 * hz, 0.18 * hx, 0.26 * hz
    for side in (-1, 1):
        for k, wx in enumerate((x - 0.45 * hx, x + 0.45 * hx)):
            face = y + side * hy
            tag = f"{name}_win{side}{k}"
            add(torso, f"{tag}_frame", box, (ww + 0.0008, 0.0003, wh + 0.0008), ROOF,
                pos=(wx, face + side * 0.0003, wz))
            add(torso, tag, box, (ww, 0.0003, wh), "globe_window",
                pos=(wx, face + side * 0.0005, wz))
            add(torso, f"{tag}_bar", box, (0.00025, 0.0002, wh), ROOF,
                pos=(wx, face + side * 0.0008, wz))
            add(torso, f"{tag}_sill", box, (ww + 0.0012, 0.0012, 0.0006), SNOW,
                pos=(wx, face + side * 0.0012, wz - wh - 0.0010))
    for k, (dx, dy) in enumerate(((1, 0), (-1, 0), (0, 1), (0, -1))):
        add(torso, f"{name}_drift{k}", mujoco.mjtGeom.mjGEOM_ELLIPSOID,
            (hx if dy else 0.25 * hx, hy if dx else 0.25 * hy, 0.12 * hz), SNOW,
            pos=(x + dx * hx, y + dy * hy, z0 + 0.002))


def _snowman(spec, torso, ground):
    """Three balls, coal face and buttons, a carrot nose, twig arms, scarf and top hat."""
    import mujoco

    sphere, cyl = mujoco.mjtGeom.mjGEOM_SPHERE, mujoco.mjtGeom.mjGEOM_CYLINDER
    caps = mujoco.mjtGeom.mjGEOM_CAPSULE

    def on(centre, r, az, el, out=0.97):
        return centre + out * r * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az),
                                            np.sin(el)])

    radii = (0.085 * R, 0.065 * R, 0.05 * R)
    z = ground[2] - 0.15 * radii[0]
    centres = []
    for k, r in enumerate(radii):
        z += 0.92 * r
        centres.append(ground[:2].tolist() + [z])
        add(torso, f"snowman{k}", mujoco.mjtGeom.mjGEOM_ELLIPSOID, (r, r, 0.92 * r), SNOW,
            pos=centres[-1])
        z += 0.92 * r * 0.72
    mid, head = np.array(centres[1]), np.array(centres[2])
    rm, r = radii[1], radii[2]

    for side in (-1, 1):
        add(torso, f"snowman_eye{side}", sphere, (0.12 * r,), COAL,
            pos=on(head, r, side * 0.38, 0.3))
    for k, a in enumerate(np.linspace(-0.45, 0.45, 5)):
        add(torso, f"snowman_mouth{k}", sphere, (0.07 * r,), COAL,
            pos=on(head, r, a, -0.32 + 0.12 * (a / 0.45) ** 2))
    for k, el in enumerate((0.35, 0.0, -0.35)):
        add(torso, f"snowman_button{k}", sphere, (0.11 * rm,), COAL, pos=on(mid, rm, 0.0, el))
    nose = head + [0.85 * r, 0.0, 0.0]
    th = 2 * np.pi * np.arange(8) / 8
    _hull(spec, torso, "snowman_nose",
          [*(nose + 0.16 * r * np.column_stack([np.zeros(8), np.cos(th), np.sin(th)])),
           head + [1.9 * r, 0.0, -0.08 * r]], CARROT)

    for side in (-1, 1):
        start = on(mid, rm, side * 1.5, 0.25, 0.85)
        reach = np.array([0.25, side * 0.8, 0.55])
        end = start + 1.3 * rm * reach / np.linalg.norm(reach)
        add(torso, f"snowman_arm{side}", caps, (0.06 * rm,), BARK,
            fromto=np.concatenate([start, end]))
        for k, bend in enumerate(((0.3, 0.1, 0.3), (-0.1, 0.25, -0.1))):
            twig = np.array(bend) * [1, side, 1] * rm
            add(torso, f"snowman_twig{side}{k}", caps, (0.04 * rm,), BARK,
                fromto=np.concatenate([end, end + twig]))

    neck = head - [0.0, 0.0, 0.85 * r]
    ring = np.column_stack([np.cos(np.linspace(0, 2 * np.pi, 32, endpoint=False)),
                            np.sin(np.linspace(0, 2 * np.pi, 32, endpoint=False)),
                            np.zeros(32)])
    _ribbon(spec, torso, "snowman_scarf", neck + 0.85 * r * ring, ring, 0.2 * r, 0.1 * r, SCARF,
            closed=True, prefix="globe")
    out = np.array([np.cos(0.6), np.sin(0.6), 0.0])
    s = np.linspace(0.0, 1.0, 8)[:, None]
    tail = neck + 0.95 * r * out + s * [0.25 * r, 0.2 * r, -1.1 * r] + s**2 * 0.3 * r * out
    _ribbon(spec, torso, "snowman_scarf_tail", tail, np.tile(out, (len(tail), 1)), 0.17 * r,
            0.08 * r, SCARF, prefix="globe")

    brim = head + [0.0, 0.0, 0.8 * r]
    add(torso, "snowman_brim", cyl, (0.95 * r, 0.06 * r), COAL, pos=brim)
    add(torso, "snowman_band", cyl, (0.64 * r, 0.12 * r), SCARF, pos=brim + [0, 0, 0.2 * r])
    add(torso, "snowman_hat", cyl, (0.62 * r, 0.5 * r), COAL, pos=brim + [0, 0, 0.55 * r])


def _lamp(spec, torso, ground):
    """An iron street lamp with a glowing lantern and snow on its cap."""
    import mujoco

    cyl, caps = mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE
    x, y, z = ground
    add(torso, "lamp_foot", cyl, (0.035 * R, 0.02 * R), IRON, pos=(x, y, z))
    top = z + 0.36 * R
    add(torso, "lamp_pole", cyl, (0.012 * R,), IRON, fromto=(x, y, z, x, y, top))
    half, tall = 0.032 * R, 0.045 * R
    mid = top + tall
    add(torso, "lamp_glow", mujoco.mjtGeom.mjGEOM_BOX, (half, half, tall), "globe_window",
        pos=(x, y, mid))
    for k, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
        add(torso, f"lamp_post{k}", caps, (0.005 * R,), IRON,
            fromto=(x + sx * half, y + sy * half, top, x + sx * half, y + sy * half, mid + tall))
    cap = mid + tall
    _hull(spec, torso, "lamp_cap",
          [(x + sx * 1.4 * half, y + sy * 1.4 * half, cap) for sx in (-1, 1) for sy in (-1, 1)]
          + [(x, y, cap + 0.04 * R)], IRON)
    add(torso, "lamp_snow", mujoco.mjtGeom.mjGEOM_ELLIPSOID, (1.2 * half, 1.2 * half, 0.015 * R),
        SNOW, pos=(x, y, cap + 0.02 * R))


def _disc_frame(n):
    """A rotation whose z axis is `n`."""
    a = np.cross(n, [0.0, 0.0, 1.0] if abs(n[2]) < 0.9 else [1.0, 0.0, 0.0])
    a /= np.linalg.norm(a)
    return np.column_stack([a, np.cross(n, a), n])


def _specks(torso, c, rng):
    """Flat flakes lying just inside the glass, so none stands proud of it."""
    import mujoco

    placed = 0
    while placed < SPECKS:
        n = rng.normal(size=3)
        n /= np.linalg.norm(n)
        if n[2] < -0.6:      # below here the shell hides the glass
            continue
        add(torso, f"speck{placed}", mujoco.mjtGeom.mjGEOM_CYLINDER,
            (rng.uniform(0.0005, 0.0014), 0.00006), SNOW, pos=c + (R - 0.00012) * n,
            mat=_disc_frame(n))
        placed += 1


def _flakes(torso, c, rng):
    """Flat flakes caught mid-fall, tumbling at random angles."""
    import mujoco

    placed = 0
    while placed < FLAKES:
        rel = rng.uniform(-RIN, RIN, 3)
        if np.linalg.norm(rel) > 0.9 * RIN or rel[2] < _snow_z(*rel[:2]) + 0.06 * R:
            continue
        n = rng.normal(size=3)
        add(torso, f"flake{placed}", mujoco.mjtGeom.mjGEOM_CYLINDER,
            (rng.uniform(0.0005, 0.0010), 0.00006), SNOW, pos=c + rel,
            mat=_disc_frame(n / np.linalg.norm(n)))
        placed += 1

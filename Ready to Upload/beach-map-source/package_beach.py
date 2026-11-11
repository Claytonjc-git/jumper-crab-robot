"""Package the existing Train beach through the separate Design producer."""
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image
from shellflow.map_package import export_environment, compose_default
from shellflow.package_io import file_records, read_archive
from shellflow.packages import verify_package

ROOT = Path(r"C:\Users\clayt\Documents\Jumper Crab")
DESIGN = Path(r"C:\Users\clayt\AppData\Local\Temp\jumper-design-ninja")
SOURCE = ROOT / "outputs/beach-map-source/dressed/beach"
WORK = ROOT / "outputs/beach-map-source"
PROFILE = DESIGN / "robots/jumper/profile.json"
OUTPUT = ROOT / "Ready to Upload/beach.map"
DESIGN_COMMIT = "918d192857fcea1678c2a40b339501004a94d838"

files = {p.relative_to(SOURCE).as_posix(): p.read_bytes()
         for p in SOURCE.rglob("*") if p.is_file()}
manifest = json.loads(files["scene-package.json"])
manifest["title"] = "Sunset Beach"
manifest["world"]["attach"] = {"prefix": "scn_"}
manifest["spawn"] = {"position": [0, 0, 0], "yaw": 0}
manifest["exportedBy"]["designCommit"] = DESIGN_COMMIT
manifest["files"] = file_records({n: b for n, b in files.items()
                                 if n != "scene-package.json"})
files["README.md"] = (
    "# Sunset Beach\n\n"
    "The existing video beach: flat sand, six original sunset skybox faces, "
    "low warm directional sunlight, sea, palm tree, umbrella, surfboard, sandcastle, "
    "beach ball and shells. Props are visual only, with no mass or collisions, "
    "as in the source video. Ground elevation is 0 m; spawn yaw is 0 degrees.\n\n"
    "The included original Jumper is a display default. Train replay keeps its own "
    "robot and controller; do not replace its mechanics with the map default. "
    "The map contains no motion controller.\n\n"
    "Native MuJoCo scene visuals and assets are preserved. Consumers must honor "
    "the scene skybox, materials, light, and headlight; MJCF attachment alone retains "
    "the host's global visual settings. No be-web/1 visual override is supplied, "
    "because its fixed daytime lighting would change this existing sunset. "
    "Web visual fidelity and physical hardware are untested.\n\n"
    f"Design producer commit: {DESIGN_COMMIT}\n"
).encode()
files["scene-package.json"] = (json.dumps(manifest, indent=2) + "\n").encode()

model = mujoco.MjModel.from_xml_string(files["scene.xml"].decode(), assets=files)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
model.vis.global_.offwidth, model.vis.global_.offheight = 960, 540
camera = mujoco.MjvCamera()
camera.lookat[:] = [-1.4, 0, 0.7]
camera.distance, camera.azimuth, camera.elevation = 7, 20, -14
with mujoco.Renderer(model, height=540, width=960) as renderer:
    renderer.update_scene(data, camera)
    image = renderer.render()
    assert image.std() > 10 and np.isfinite(image).all(), "Blank beach render"
    Image.fromarray(image).save(WORK / "beach-preview.png")

report = export_environment(files, OUTPUT,
                            DESIGN / "library/skins/jumper-original.skin",
                            PROFILE, preview=WORK / "beach-preview.png")
final = read_archive(OUTPUT)
for name, content in files.items():
    if name != "scene-package.json":
        assert final[name] == content, f"Environment resource changed: {name}"
assert all("\\" not in name for name in final), "Nonportable archive path"
assert final["preview/preview.png"] == (WORK / "beach-preview.png").read_bytes()
verification = verify_package(OUTPUT, platform_profile=PROFILE,
                              capabilities={"rigid", "jumper"}, mujoco=True)
combined_model, combined_data, composition = compose_default(final, profile=PROFILE)
combined_model.vis.global_.offwidth, combined_model.vis.global_.offheight = 960, 540
with mujoco.Renderer(combined_model, height=540, width=960) as renderer:
    renderer.update_scene(combined_data, camera)
    Image.fromarray(renderer.render()).save(WORK / "beach-combined.png")
summary = {"export": report, "verification": verification, "composition": composition,
           "design_commit": DESIGN_COMMIT,
           "source_scene_sha256": hashlib.sha256(files["scene.xml"]).hexdigest(),
           "native_preview_only": True, "web_acceptance_tested": False}
(WORK / "validation.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))

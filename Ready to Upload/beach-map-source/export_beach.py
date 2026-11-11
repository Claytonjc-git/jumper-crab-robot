"""Export the existing video set without changing its source or physics."""
import dataclasses
import hashlib
import importlib.util
import json
from pathlib import Path

from PIL import Image
import scenes.beach as beach
from scenes.tools.export_web_scene import export

ROOT = Path.cwd()
WORK = ROOT / "outputs/beach-map-source"
props_path = ROOT / "outputs/frontflip-short-v4/beach_props.py"
module_spec = importlib.util.spec_from_file_location("beach_video_props", props_path)
props = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(props)
from scenes.skygen import FACE_ORDER
faces = props._ball_faces((0.5, 0.45, 0.95))
ball_files = []
for index, face in enumerate(FACE_ORDER):
    path = WORK / f"beach_ball_{face}.png"
    Image.fromarray(faces[index * 256:(index + 1) * 256]).save(path)
    ball_files.append(str(path))

def decorate(spec):
    props.decorate(spec)
    ball = spec.texture("prop_ball")
    ball.data = b""
    ball.cubefiles = ball_files
    decorative = [g for g in spec.geoms if g.name.startswith("prop_")]
    assert len(decorative) > 60
    assert all(g.mass == 0 and g.contype == 0 and g.conaffinity == 0
               for g in decorative), "Beach dressing changed physics"

original_scene = beach.scene
beach.scene = lambda: dataclasses.replace(original_scene(), decorate=decorate)
output = export("beach", WORK / "dressed")
manifest_path = output / "scene-package.json"
manifest = json.loads(manifest_path.read_bytes())
manifest["exportedBy"]["videoDressingSourceSha256"] = hashlib.sha256(props_path.read_bytes()).hexdigest()
manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
print(output)

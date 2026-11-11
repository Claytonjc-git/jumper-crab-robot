"""Verify the delivered map survives the supported import/repack/reload path."""
import json
from pathlib import Path

from shellflow.packages import import_package, repack_package, verify_package
from shellflow.package_io import read_archive

ROOT = Path(r"C:\Users\clayt\Documents\Jumper Crab")
PROFILE = Path(r"C:\Users\clayt\AppData\Local\Temp\jumper-design-ninja\robots\jumper\profile.json")
WORK = ROOT / "outputs/beach-map-source"
MAP = ROOT / "Ready to Upload/beach.map"
options = {"platform_profile": PROFILE, "capabilities": {"rigid", "jumper"}}
import_package(MAP, WORK / "roundtrip-import", mujoco=True, **options)
returned = WORK / "roundtrip.map"
repack_package(MAP, returned)
assert read_archive(MAP) == read_archive(returned), "Round trip changed package bytes"
report = verify_package(returned, mujoco=True, **options)
(WORK / "roundtrip-validation.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({"roundtrip_ok": report["ok"],
                  "internal_files_identical": True,
                  "native_reload_ok": report["native_mujoco"]["passed"]}))

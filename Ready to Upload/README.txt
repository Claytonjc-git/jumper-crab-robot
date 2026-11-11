JUMPER FRONTFLIP + SUNSET BEACH

frontflip.app
  Updated: regular Jumper controls plus frontflip as an added task.
  Starts in locomotion. Stop moving, then Ctrl+Space or Menu+A selects frontflip;
  it returns to locomotion automatically. Bare Space/A keeps the original jump.
  See FRONTFLIP CONTROLS.txt for a quick guide. Full English/Chinese manuals
  are inside the app. All 12 original tasks remain; 13 tasks total.

beach.map
  Sunset Beach with the palm, umbrella, surfboard, sandcastle, beach ball,
  shells, sea, sand and original sunset skybox from the video.
  Decorations are visual only, without mass or collisions. Spawn is ground
  height 0 m, heading +X. Includes an original Jumper display default and a
  robot-free preview. Train replay retains its own robot/controller.

Covers/
  Cover images and transparent mid-flip renders for your community uploads.
  jumper-frontflip-1448x1086.png matches your requested cover dimensions.

VALIDATION AND LIMITS
  App: native/browser reference checks, shortcuts, regular-control preservation,
  completion/automatic return, tilt handling, archive hashes, both manuals.
  A 14-second physics diagnostic flipped, landed upright, returned and walked.
  That diagnostic omitted only the world's fell_over termination: stock Train
  currently resets the world midflip instead of honoring the app's per-mode
  exception. Controller safety remains active, with tilt exempt only in flip.

  Map: package hashes, trusted robot profile, native loading, spawn support,
  import/repack/reload with identical internal bytes. Stock Train's importer
  currently fails to resolve the included assets/beach_ball_right.png after
  attachment; Design's reference composer loads it correctly.

  Website import, Web map appearance and physical hardware remain untested.
  The app has no board runtime or frontflip RKNN model. The map preserves
  original MJCF sunset visuals without a be-web/1 daytime-lighting override.
  Different robot mechanical revisions are not certified interchangeable.

PROVENANCE
  Frontflip policy: Run 5 model_1500, checkpoint SHA256
  43b747b3cd26ade877fd900a72fcf22c4ab247d2074e2765bd15b4a940b2f296
  Regular modes and runtime: existing original Jumper app.
  Runtime commit: ce71f79b191766f0208dca34516564a37816c729 (reused unchanged)
  Train commit: 178d27e7006626773916e7585bda5202ba9df86f
  Design map producer commit: 918d192857fcea1678c2a40b339501004a94d838

  App build and detailed checks: out/frontflip-regular-v2/
  Map checks: outputs/beach-map-source/
  SHA256SUMS.txt records the two current upload files.

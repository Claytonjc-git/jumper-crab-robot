# Sunset Beach

The existing video beach: flat sand, six original sunset skybox faces, low warm directional sunlight, sea, palm tree, umbrella, surfboard, sandcastle, beach ball and shells. Props are visual only, with no mass or collisions, as in the source video. Ground elevation is 0 m; spawn yaw is 0 degrees.

The included original Jumper is a display default. Train replay keeps its own robot and controller; do not replace its mechanics with the map default. The map contains no motion controller.

Native MuJoCo scene visuals and assets are preserved. Consumers must honor the scene skybox, materials, light, and headlight; MJCF attachment alone retains the host's global visual settings. No be-web/1 visual override is supplied, because its fixed daytime lighting would change this existing sunset. Web visual fidelity and physical hardware are untested.

Design producer commit: 918d192857fcea1678c2a40b339501004a94d838

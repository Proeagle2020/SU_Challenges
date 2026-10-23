Webots R2025a — Fixed-wing educational demo (fixed package)

IMPORTANT: This version uses a self-contained WBT world with inline geometry and no EXTERNPROTO path. This avoids common relative-path mistakes when the world is moved. The PROTO is included separately for reuse.

Install/run:
1. Extract this ZIP to a normal folder.
2. Open worlds/fixed_wing_training.wbt from inside Webots using File > Open World.
3. Keep the controllers/fixed_wing_training/fixed_wing_training.py folder structure.
4. Press Play. If the controller is not found, open Tools > Preferences or check the Webots console for the controller path.
5. To use the reusable PROTO in another world, copy protos/FixedWingTraining.proto into that project's protos folder and use an EXTERNPROTO path relative to the .wbt file.

This is a simplified educational physics demo, not an aerodynamic or flight-dynamics validated model.

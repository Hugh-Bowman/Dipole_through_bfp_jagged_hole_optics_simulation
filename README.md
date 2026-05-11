# Dipole through BFP jagged hole optics simulation

Simulation of dipole emission from a gold nanorod imaged by a polarisation camera, including circular and real (image-derived) hole profiles.

## Run (macOS/Linux)

```bash
cd Dipole_through_jagged_hole_simulation
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python simulate_X_Y_rod_sim_only.py
```

The program loads the hole image from:
- `image_of_hole/cleaned_circle_with_hole.png`

You can replace `cleaned_circle_with_hole.png` with your own BFP PNG image and the script should process it, but this is not guaranteed for all image qualities/formats.
Always inspect the generated debug outputs in `image_of_hole/mask_debug` to confirm the annulus fit and hole mask are correct before trusting simulation results.

If needed (permissions/cache issue on some systems), run with:

```bash
MPLCONFIGDIR=.mplconfig python simulate_X_Y_rod_sim_only.py
```

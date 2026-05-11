# Dipole Through Jagged Hole Simulation

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

If needed (permissions/cache issue on some systems), run with:

```bash
MPLCONFIGDIR=.mplconfig python simulate_X_Y_rod_sim_only.py
```

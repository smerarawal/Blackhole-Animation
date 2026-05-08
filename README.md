# Blackhole-Animation
# M87* Black Hole Raytracer — EHT 2021

A physically accurate renderer of the M87* supermassive black hole,
matching the Event Horizon Telescope 2021 April 18 image.
Runs on CPU (Numba parallel JIT) or NVIDIA CUDA GPU.

---

## What This Simulates

This code traces light rays backward through curved spacetime around
a spinning (Kerr) black hole. Every pixel is a photon path solved
with 4th-order Runge-Kutta integration of the geodesic equations.

### Physics Implemented

| Component | Equation / Model |
|---|---|
| Spacetime metric | Kerr (Boyer-Lindquist coords), G=c=M=1 |
| Geodesic equation | d²u/dφ² = −u + 3/2 u² + a²u³ |
| Integration | 4th-order Runge-Kutta, 1800 steps/ray |
| Accretion disk | Novikov-Thorne (1973) thin disk flux |
| Disk temperature | F(r) ∝ (1 − √(r_ISCO/r)) / r³ |
| Doppler beaming | D = 1/[γ(1 + β sinφ sinθ_obs)] |
| Intensity scaling | I ∝ D⁴ (synchrotron, spectral index α=1) |
| Photon ring | Each orbit attenuated by factor ~0.037ⁿ |
| ISCO (prograde) | Bardeen, Press & Teukolsky (1972) exact formula |
| Event horizon | r_h = 1 + √(1 − a²) |
| Turbulence | Fractal Brownian Motion in spiral (r, φ) coords |
| Hotspot | Gaussian flare co-rotating with Keplerian disk |
| Color pipeline | EHT false-color + log-stretch (matches published images) |
| PSF | Gaussian convolution σ=2.1px (telescope beam) |

### Key Physical Parameters for M87*

- Black hole mass: 6.5 × 10⁹ M☉
- Distance: 16.8 Mpc
- Spin parameter: a ≈ 0.90–0.94
- Observer angle: ~163° from north pole (~17° below equatorial plane)
- Shadow diameter: ~42 μas (maps to ~5.2 gravitational radii)
- ISCO radius: ~2.3 M (prograde, a=0.94)
- Photon sphere: ~1.6 M

---

## Installation

```bash
pip install numba numpy matplotlib scipy pillow
```

For CUDA GPU acceleration (NVIDIA only):
```bash
pip install numba
# Also install CUDA Toolkit from: https://developer.nvidia.com/cuda-downloads
```

Python 3.9+ recommended.

---

## Usage

```bash
# Default — M87* geometry, 900×900
python blackhole_eht2021.py

# High resolution
python blackhole_eht2021.py --res 1400

# Near-extremal Kerr (a=0.998)
python blackhole_eht2021.py --spin 0.998 --res 1200

# Schwarzschild (non-spinning)
python blackhole_eht2021.py --spin 0.0

# Change observer inclination
python blackhole_eht2021.py --incl 45

# Edge-on view
python blackhole_eht2021.py --incl 90

# Animated rotating hotspot (saves .gif)
python blackhole_eht2021.py --animate 72

# Custom output filename
python blackhole_eht2021.py --out my_blackhole.png
```

### All Arguments

| Argument | Default | Description |
|---|---|---|
| `--res` | 900 | Image resolution (N×N pixels) |
| `--spin` | 0.94 | Kerr spin parameter a ∈ [0, 0.999] |
| `--incl` | 163.0 | Observer polar angle in degrees (163° = M87*) |
| `--animate` | 0 | Number of animation frames (0 = static render) |
| `--out` | m87_2021.png | Output filename (.png or .gif for animation) |

---

## Output

- Static render saved as PNG (default: `m87_2021.png`)
- Animation saved as GIF when `--animate N` is set
- matplotlib window opens for interactive viewing
- Console prints mean brightness value and render time per frame

---

## Performance

| Hardware | Resolution | Time per frame |
|---|---|---|
| 4-core CPU | 900×900 | ~60–90 seconds |
| 8-core CPU | 900×900 | ~30–45 seconds |
| NVIDIA RTX 3080 | 900×900 | ~3–6 seconds |
| NVIDIA RTX 4090 | 1400×1400 | ~4–8 seconds |

First run is slower (~15s extra) due to Numba JIT compilation.
Subsequent runs use cached compiled kernels.

---

## File Structure

```
blackhole_eht2021.py   — main renderer (single file, no dependencies beyond pip)
m87_2021.png           — output render
m87_2021.gif           — output animation (if --animate used)
```

---

## How It Works — Step by Step

1. **Ray setup**: For each pixel, compute impact parameters (α, β)
   representing the photon's distance from the optical axis.

2. **Geodesic integration**: Solve the Kerr orbit equation forward
   from the observer. The photon path curves around the black hole.

3. **Disk intersection**: When the ray crosses the equatorial plane
   (θ = π/2) within the disk region (r_ISCO ≤ r ≤ 12M), record the hit.

4. **Emission**: Compute disk brightness using Novikov-Thorne flux,
   multiplied by Doppler factor D⁴ and GRMHD turbulence pattern.

5. **Photon ring**: Rays that orbit the black hole once or more
   contribute additional images of the disk, attenuated by ~0.037 per orbit.

6. **Color**: Map intensity through log-stretch and EHT false-color
   scale (black → dark red → orange → yellow → white).

7. **Post-processing**: Apply telescope PSF (Gaussian blur), re-enforce
   black shadow, add crescent asymmetry and radial streak pattern.

---

## Visual Features Explained

**Bright crescent (bottom)** — The disk gas on the approaching side moves
toward the observer at ~0.3–0.4c. Relativistic beaming concentrates
radiation toward the observer, making that side 3–10× brighter.

**Dark top of ring** — The receding side (gas moving away) is Doppler
dimmed. Combined with the low inclination angle (~17°), the top of
the ring nearly vanishes.

**Spiral streaks** — GRMHD (General Relativistic MagnetoHydroDynamics)
simulations show turbulent magnetic field structure in the disk.
Modelled here as fractal Brownian motion in spiral polar coordinates.

**Compact hotspot** — A bright, compact emission region (plasma flare)
orbits near the ISCO. The 2021 EHT image shows a bright knot at the
bottom of the ring. In the animation it orbits with Keplerian period.

**Photon ring** — A thin bright ring just outside the shadow. Light
that narrowly avoids capture completes one or more orbits, forming
a series of increasingly thin, bright rings (n=1,2,3...).
Only n=0 (direct image) and n=1 are practically visible.

**Black shadow** — The central dark region is NOT the event horizon —
it is the photon capture cross-section, roughly 2.6× larger than
the event horizon. Any photon aimed here falls in.

---

## References

- Bardeen, Press & Teukolsky (1972) — Rotating Black Holes
- Novikov & Thorne (1973) — Astrophysics of Black Holes
- Page & Thorne (1974) — Disk-Accretion onto a Black Hole
- EHT Collaboration (2019) — First M87* Image, ApJL 875, L1
- EHT Collaboration (2021) — M87* Shadow and Jet, ApJL 910, L12
- EHT Collaboration (2022) — Seven-Year Monitoring, ApJL 927, L5

---

## License

MIT — free to use, modify, and distribute with attribution.

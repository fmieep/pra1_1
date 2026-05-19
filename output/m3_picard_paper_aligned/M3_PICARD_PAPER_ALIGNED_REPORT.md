# M3 Picard Paper-Aligned Experiment Report

Date: 2026-05-19

This report keeps the existing baseline checkpoints untouched and compares them with the recommended paper-aligned candidate: `jacobi_5_5 + eta_guard_1p1`.

## Parameters Kept

- `nonlinear_tol=2e-6`
- `linear_tol_factor=3e-3`
- `damping_norm=linf`
- `picard_mass_mode=q_derivative`
- current Wilson limiter, Robin boundary, and midpoint freeze-state

## Candidate Parameters

- `mg_smoother=jacobi`
- `mg_pre_smooths=5`
- `mg_post_smooths=5`
- `eta_guard_factor=1.1`

## Generated Figures

- `m3_picard_eta0.1_temperature_contours.png`
- `m3_picard_eta0.5_temperature_contours.png`
- `m3_picard_eta0.1_linear_history.png`
- `m3_picard_eta0.1_nonlinear_history.png`
- `m3_picard_eta0.5_linear_history.png`
- `m3_picard_eta0.5_nonlinear_history.png`
- `m3_picard_average_iterations_comparison.png`

## Summary Table

| eta | grid | variant | steps | avg linear | paper linear | avg nonlinear | paper nonlinear | max nonlinear | max eta | T max |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 32 | aligned_guard_1p1 | 103 | 4.2330 | 3.2600 | 4.2330 | 4.2600 | 8 | 0.1023 | 2.1863 |
| 0.1 | 32 | baseline | 103 | 4.2233 | 3.2600 | 4.2233 | 4.2600 | 8 | 0.1023 | 2.1863 |
| 0.1 | 64 | aligned_guard_1p1 | 129 | 5.4806 | 4.0900 | 5.4806 | 5.6900 | 9 | 0.1046 | 4.6924 |
| 0.1 | 64 | baseline | 129 | 5.4806 | 4.0900 | 5.4806 | 5.6900 | 9 | 0.1046 | 4.6924 |
| 0.1 | 128 | aligned_guard_1p1 | 258 | 8.1899 | 7.0600 | 8.1899 | 8.0500 | 12 | 0.1062 | 8.2653 |
| 0.1 | 128 | baseline | 258 | 8.1899 | 7.0600 | 8.1899 | 8.0500 | 12 | 0.1062 | 8.2653 |
| 0.5 | 32 | aligned_guard_1p1 | 140 | 8.1571 | 5.4300 | 8.1571 | 6.4300 | 28 | 0.5273 | 8.8005 |
| 0.5 | 32 | baseline | 137 | 8.6204 | 5.4300 | 8.1533 | 6.4300 | 28 | 0.5719 | 8.8007 |
| 0.5 | 64 | aligned_guard_1p1 | 197 | 12.5685 | 9.7500 | 12.5685 | 9.4900 | 30 | 0.5480 | 8.9765 |
| 0.5 | 64 | baseline | 194 | 18.9124 | 9.7500 | 12.6753 | 9.4900 | 34 | 0.5688 | 8.9767 |
| 0.5 | 128 | aligned_guard_1p1 | 304 | 22.9704 | 22.7000 | 18.5625 | 14.7000 | 34 | 0.5435 | 9.0455 |
| 0.5 | 128 | baseline | 301 | 32.6944 | 22.7000 | 18.7409 | 14.7000 | 41 | 0.5812 | 9.0456 |

## Conclusion

The paper-aligned candidate should be kept as a separate comparison branch, not as a replacement for the baseline checkpoints.

For linear iterations, the candidate is closer to the paper on the difficult `eta=0.5, 128x128` case. For nonlinear iterations, the spike is reduced, but the average Picard iteration count remains above the paper.

Do not repeat this experiment unless the limiter, boundary discretization, nonlinear residual, or time-step controller changes.

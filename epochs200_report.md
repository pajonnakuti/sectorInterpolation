# 25 vs 200 epochs

28/28 models trained for 200 epochs. The 200-epoch models are installed in `models/`; the 25-epoch ones are in `models_25ep/`.

Mean over 28 models: validation RMSE 0.679 → 0.597 °C, test RMSE (Jul–Dec 2016) 0.747 → 0.667 °C (-10.7%).

| Satellite | Sector | val 25 | val 200 | test 25 | test 200 | test change | notes |
|---|---|---|---|---|---|---|---|
| NOAA-18 | 1 Gujarat | 0.856 | 0.668 | 0.669 | 0.508 | -24% | best epoch 189, 31 min (30-day composite 1.11) |
| NOAA-18 | 2 Maharashtra | 0.517 | 0.449 | 0.524 | 0.474 | -10% | best epoch 144, 24 min (30-day composite 1.09) |
| NOAA-18 | 3 Goa | 0.546 | 0.492 | 0.758 | 0.637 | -16% | best epoch 140, 12 min (30-day composite 1.38) |
| NOAA-18 | 4 Karnataka | 0.664 | 0.612 | 0.636 | 0.610 | -4% | best epoch 95, 15 min (30-day composite 1.24) |
| NOAA-18 | 5 Kerala | 0.716 | 0.672 | 0.850 | 0.761 | -10% | best epoch 176, 23 min (30-day composite 1.45) |
| NOAA-18 | 6 South Tamil Nadu | 0.831 | 0.731 | 1.023 | 0.951 | -7% | best epoch 191, 20 min (30-day composite 1.69) |
| NOAA-18 | 7 Tamil Nadu | 0.700 | 0.639 | 0.927 | 0.855 | -8% | best epoch 198, 19 min (30-day composite 1.69) |
| NOAA-18 | 8 South Andhra Pradesh | 0.906 | 0.795 | 0.873 | 0.841 | -4% | best epoch 90, 20 min (30-day composite 1.56) |
| NOAA-18 | 9 North Andhra Pradesh | 0.731 | 0.618 | 0.722 | 0.639 | -11% | best epoch 167, 18 min (30-day composite 1.39) |
| NOAA-18 | 10 Odisha | 0.737 | 0.590 | 0.767 | 0.611 | -20% | best epoch 189, 20 min (30-day composite 1.43) |
| NOAA-18 | 11 West Bengal | 0.703 | 0.569 | 0.751 | 0.629 | -16% | best epoch 156, 10 min (30-day composite 1.63) |
| NOAA-18 | 12 Lakshadweep | 0.641 | 0.584 | 0.856 | 0.748 | -13% | best epoch 198, 19 min (30-day composite 1.43) |
| NOAA-18 | 13 North Andaman | 0.686 | 0.613 | 1.012 | 0.887 | -12% | best epoch 194, 20 min (30-day composite 1.67) |
| NOAA-18 | 14 South Andaman | 0.707 | 0.600 | 1.215 | 1.035 | -15% | best epoch 177, 13 min (30-day composite 2.14) |
| NOAA-19 | 1 Gujarat | 0.635 | 0.543 | 0.537 | 0.465 | -13% | best epoch 152, 37 min (30-day composite 0.96) |
| NOAA-19 | 2 Maharashtra | 0.488 | 0.435 | 0.442 | 0.425 | -4% | best epoch 176, 22 min (30-day composite 0.82) |
| NOAA-19 | 3 Goa | 0.556 | 0.485 | 0.559 | 0.512 | -8% | best epoch 147, 15 min (30-day composite 0.98) |
| NOAA-19 | 4 Karnataka | 0.539 | 0.490 | 0.604 | 0.566 | -6% | best epoch 190, 19 min (30-day composite 1.03) |
| NOAA-19 | 5 Kerala | 0.659 | 0.637 | 0.794 | 0.758 | -5% | best epoch 100, 26 min (30-day composite 1.18) |
| NOAA-19 | 6 South Tamil Nadu | 0.820 | 0.743 | 0.953 | 0.856 | -10% | best epoch 186, 23 min (30-day composite 1.36) |
| NOAA-19 | 7 Tamil Nadu | 0.735 | 0.683 | 0.840 | 0.804 | -4% | best epoch 196, 22 min (30-day composite 1.39) |
| NOAA-19 | 8 South Andhra Pradesh | 0.567 | 0.505 | 0.658 | 0.605 | -8% | best epoch 182, 22 min (30-day composite 1.11) |
| NOAA-19 | 9 North Andhra Pradesh | 0.698 | 0.632 | 0.633 | 0.566 | -11% | best epoch 151, 24 min (30-day composite 1.15) |
| NOAA-19 | 10 Odisha | 0.659 | 0.563 | 0.586 | 0.514 | -12% | best epoch 166, 23 min (30-day composite 1.18) |
| NOAA-19 | 11 West Bengal | 0.691 | 0.512 | 0.607 | 0.455 | -25% | best epoch 156, 17 min (30-day composite 1.42) |
| NOAA-19 | 12 Lakshadweep | 0.628 | 0.587 | 0.623 | 0.593 | -5% | best epoch 184, 26 min (30-day composite 0.99) |
| NOAA-19 | 13 North Andaman | 0.708 | 0.651 | 0.747 | 0.693 | -7% | best epoch 128, 25 min (30-day composite 1.19) |
| NOAA-19 | 14 South Andaman | 0.700 | 0.632 | 0.741 | 0.678 | -8% | best epoch 191, 12 min (30-day composite 1.32) |

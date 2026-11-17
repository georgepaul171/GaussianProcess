## Full posterior (joint-posterior methods only)

**d = 1**

| kernel | β | n | pts/ℓ^d | ref | c* (nnz/row) | winner: wall-clock | s | err/stat-err | var ratio | coverage | winner: flops | band eligible? (s) | latent Vecchia eligible? (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| m32 | 0 | 1e+04 | 251 | exact | 4 (1910) | vecchia_latent_m10 | 0.252 | 0.00526 | 1 | 0.968 | vecchia_latent_m10 | no | yes (0.252) |
| m32 | 0 | 1e+05 | 1e+03 | none | none valid | (none eligible) | - | - | - | - | (none eligible) | no | no |
| m32 | 0 | 1e+06 | 3.98e+03 | none | none valid | (none eligible) | - | - | - | - | (none eligible) | no | no |
| m32 | 0.5 | 1e+04 | 316 | exact | 4 (2374) | vecchia_latent_m10 | 0.233 | 0.0063 | 0.999 | 0.378 | vecchia_latent_m10 | no | yes (0.233) |
| m32 | 0.5 | 1e+05 | 1e+03 | none | none valid | (none eligible) | - | - | - | - | (none eligible) | no | no |
| m32 | 0.5 | 1e+06 | 3.16e+03 | none | none valid | (none eligible) | - | - | - | - | (none eligible) | no | no |
| m32 | 1 | 1e+04 | 10 | exact | 2.5 (51) | vecchia_latent_m10 | 0.229 | 0.00838 | 1 | 0.632 | vecchia_latent_m10 | no | yes (0.229) |
| m32 | 1 | 1e+05 | 10 | band c5 | 2.5 (51) | vecchia_latent_m10 | 2.51 | 0.0102 | 1 | 0.63 | vecchia_latent_m10 | no | yes (2.51) |
| m32 | 1 | 1e+06 | 10 | band c5 | 2.5 (51) | vecchia_latent_m30 | 124 | 0.00667 | 1 | 0.685 | vecchia_latent_m30 | no | yes (124) |
| se | 0 | 1e+04 | 251 | exact | 2.5 (1217) | sgpr_M100 | 0.0467 | 0.000431 | 1 | 0.922 | sgpr_M100 | yes (2.7) | yes (1.06) |
| se | 0 | 1e+05 | 1e+03 | none | none valid | (none eligible) | - | - | - | - | (none eligible) | no | no |
| se | 0 | 1e+06 | 3.98e+03 | none | none valid | (none eligible) | - | - | - | - | (none eligible) | no | no |
| se | 0.5 | 1e+04 | 316 | exact | 2.5 (1521) | sgpr_M100 | 0.0449 | 7.33e-06 | 1 | 0.148 | sgpr_M100 | yes (3.47) | yes (1.02) |
| se | 0.5 | 1e+05 | 1e+03 | none | none valid | (none eligible) | - | - | - | - | (none eligible) | no | no |
| se | 0.5 | 1e+06 | 3.16e+03 | none | none valid | (none eligible) | - | - | - | - | (none eligible) | no | no |
| se | 1 | 1e+04 | 10 | exact | 2 (41) | band_c2.5 | 0.102 | 0.00553 | 1 | 0.51 | vecchia_latent_m10 | yes (0.102) | yes (0.226) |
| se | 1 | 1e+05 | 10 | band c5 | 2 (41) | band_c2.5 | 1.08 | 0.00531 | 1 | 0.483 | vecchia_latent_m10 | yes (1.08) | yes (2.44) |
| se | 1 | 1e+06 | 10 | band c5 | 2 (41) | band_c2.5 | 21.9 | 0.0054 | 1 | 0.552 | band_c2.5 | yes (21.9) | yes (120) |

**d = 2**

| kernel | β | n | pts/ℓ^d | ref | c* (nnz/row) | winner: wall-clock | s | err/stat-err | var ratio | coverage | winner: flops | band eligible? (s) | latent Vecchia eligible? (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| m32 | 0 | 1e+04 | 51.8 | exact | 4 (1996) | vecchia_latent_m30 | 4.52 | 0.0282 | 0.995 | 0.987 | vecchia_latent_m30 | no | yes (4.52) |
| m32 | 0 | 1e+05 | 139 | none | none valid | (none eligible) | - | - | - | - | (none eligible) | no | no |
| m32 | 0.5 | 1e+04 | 316 | exact | 4 (7540) | sgpr_M1000 | 1.16 | 0.0283 | 1.02 | 0.558 | vecchia_latent_m30 | no | yes (4.55) |
| m32 | 0.5 | 1e+05 | 1e+03 | none | none valid | (none eligible) | - | - | - | - | (none eligible) | no | no |
| m32 | 1 | 1e+04 | 10 | exact | 3 (261) | vecchia_latent_m30 | 4.97 | 0.0221 | 0.997 | 0.668 | vecchia_latent_m30 | no | yes (4.97) |
| m32 | 1 | 1e+05 | 10 | band c5 | none valid | vecchia_latent_m30 | 86 | 0.0286 | 0.998 | 0.67 | vecchia_latent_m30 | no | yes (86) |
| se | 0 | 1e+04 | 51.8 | exact | 2.5 (866) | sgpr_M1000 | 1.18 | 0.00499 | 1 | 0.955 | sgpr_M1000 | yes (19.5) | no |
| se | 0 | 1e+05 | 139 | none | none valid | (none eligible) | - | - | - | - | (none eligible) | no | no |
| se | 0.5 | 1e+04 | 316 | exact | 2.5 (4034) | sgpr_M1000 | 1.28 | 3.31e-07 | 1 | 0.19 | sgpr_M1000 | yes (281) | no |
| se | 0.5 | 1e+05 | 1e+03 | none | none valid | (none eligible) | - | - | - | - | (none eligible) | no | no |
| se | 1 | 1e+04 | 10 | exact | 2 (120) | band_c2.5 | 2.9 | 0.0114 | 1 | 0.512 | band_c2.5 | yes (2.9) | no |
| se | 1 | 1e+05 | 10 | band c5 | 2 (125) | band_c2.5_metis | 69.2 | 0.0113 | 1 | 0.483 | band_c2.5 | yes (69.2) | no |

## Pointwise prediction (all methods incl. NN-Vecchia prediction)

**d = 1**

| kernel | β | n | pts/ℓ^d | ref | c* (nnz/row) | winner: wall-clock | s | err/stat-err | var ratio | coverage | winner: flops | band eligible? (s) | latent Vecchia eligible? (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| m32 | 0 | 1e+04 | 251 | exact | 4 (1910) | vecchia_latent_m10 | 0.252 | 0.00526 | 1 | 0.968 | vecchia_latent_m10 | no | yes (0.252) |
| m32 | 0 | 1e+05 | 1e+03 | none | none valid | (none eligible) | - | - | - | - | (none eligible) | no | no |
| m32 | 0 | 1e+06 | 3.98e+03 | none | none valid | (none eligible) | - | - | - | - | (none eligible) | no | no |
| m32 | 0.5 | 1e+04 | 316 | exact | 4 (2374) | vecchia_latent_m10 | 0.233 | 0.0063 | 0.999 | 0.378 | vecchia_latent_m10 | no | yes (0.233) |
| m32 | 0.5 | 1e+05 | 1e+03 | none | none valid | (none eligible) | - | - | - | - | (none eligible) | no | no |
| m32 | 0.5 | 1e+06 | 3.16e+03 | none | none valid | (none eligible) | - | - | - | - | (none eligible) | no | no |
| m32 | 1 | 1e+04 | 10 | exact | 2.5 (51) | vecchia_pred_m30 | 0.107 | 0.00586 | 1 | 0.632 | vecchia_pred_m30 | no | yes (0.229) |
| m32 | 1 | 1e+05 | 10 | band c5 | 2.5 (51) | vecchia_pred_m30 | 0.133 | 0.00847 | 1 | 0.63 | vecchia_pred_m30 | no | yes (2.51) |
| m32 | 1 | 1e+06 | 10 | band c5 | 2.5 (51) | vecchia_pred_m30 | 0.378 | 0.00846 | 1 | 0.685 | vecchia_pred_m30 | no | yes (124) |
| se | 0 | 1e+04 | 251 | exact | 2.5 (1217) | sgpr_M100 | 0.0467 | 0.000431 | 1 | 0.922 | sgpr_M100 | yes (2.7) | yes (1.06) |
| se | 0 | 1e+05 | 1e+03 | none | none valid | (none eligible) | - | - | - | - | (none eligible) | no | no |
| se | 0 | 1e+06 | 3.98e+03 | none | none valid | (none eligible) | - | - | - | - | (none eligible) | no | no |
| se | 0.5 | 1e+04 | 316 | exact | 2.5 (1521) | sgpr_M100 | 0.0449 | 7.33e-06 | 1 | 0.148 | sgpr_M100 | yes (3.47) | yes (1.02) |
| se | 0.5 | 1e+05 | 1e+03 | none | none valid | (none eligible) | - | - | - | - | (none eligible) | no | no |
| se | 0.5 | 1e+06 | 3.16e+03 | none | none valid | (none eligible) | - | - | - | - | (none eligible) | no | no |
| se | 1 | 1e+04 | 10 | exact | 2 (41) | band_c2.5 | 0.102 | 0.00553 | 1 | 0.51 | vecchia_latent_m10 | yes (0.102) | yes (0.226) |
| se | 1 | 1e+05 | 10 | band c5 | 2 (41) | vecchia_pred_m50 | 0.264 | 0.0199 | 1 | 0.49 | vecchia_pred_m50 | yes (1.08) | yes (2.44) |
| se | 1 | 1e+06 | 10 | band c5 | 2 (41) | vecchia_pred_m50 | 0.558 | 0.0193 | 1 | 0.552 | vecchia_pred_m50 | yes (21.9) | yes (120) |

**d = 2**

| kernel | β | n | pts/ℓ^d | ref | c* (nnz/row) | winner: wall-clock | s | err/stat-err | var ratio | coverage | winner: flops | band eligible? (s) | latent Vecchia eligible? (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| m32 | 0 | 1e+04 | 51.8 | exact | 4 (1996) | vecchia_latent_m30 | 4.52 | 0.0282 | 0.995 | 0.987 | vecchia_latent_m30 | no | yes (4.52) |
| m32 | 0 | 1e+05 | 139 | none | none valid | (none eligible) | - | - | - | - | (none eligible) | no | no |
| m32 | 0.5 | 1e+04 | 316 | exact | 4 (7540) | sgpr_M1000 | 1.16 | 0.0283 | 1.02 | 0.558 | vecchia_latent_m30 | no | yes (4.55) |
| m32 | 0.5 | 1e+05 | 1e+03 | none | none valid | (none eligible) | - | - | - | - | (none eligible) | no | no |
| m32 | 1 | 1e+04 | 10 | exact | 3 (261) | vecchia_latent_m30 | 4.97 | 0.0221 | 0.997 | 0.668 | vecchia_latent_m30 | no | yes (4.97) |
| m32 | 1 | 1e+05 | 10 | band c5 | none valid | vecchia_latent_m30 | 86 | 0.0286 | 0.998 | 0.67 | vecchia_latent_m30 | no | yes (86) |
| se | 0 | 1e+04 | 51.8 | exact | 2.5 (866) | sgpr_M1000 | 1.18 | 0.00499 | 1 | 0.955 | sgpr_M1000 | yes (19.5) | no |
| se | 0 | 1e+05 | 139 | none | none valid | (none eligible) | - | - | - | - | (none eligible) | no | no |
| se | 0.5 | 1e+04 | 316 | exact | 2.5 (4034) | sgpr_M1000 | 1.28 | 3.31e-07 | 1 | 0.19 | sgpr_M1000 | yes (281) | no |
| se | 0.5 | 1e+05 | 1e+03 | none | none valid | (none eligible) | - | - | - | - | (none eligible) | no | no |
| se | 1 | 1e+04 | 10 | exact | 2 (120) | band_c2.5 | 2.9 | 0.0114 | 1 | 0.512 | band_c2.5 | yes (2.9) | no |
| se | 1 | 1e+05 | 10 | band c5 | 2 (125) | band_c2.5_metis | 69.2 | 0.0113 | 1 | 0.483 | band_c2.5 | yes (69.2) | no |

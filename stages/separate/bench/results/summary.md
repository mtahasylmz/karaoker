# Bench results

| model | n | vocals SDR (median) | instr SDR (median) | avg RTF | errors |
|---|---:|---:|---:|---:|---:|
| `htdemucs` | 3 | 9.34 | 10.27 | 0.59 | 0 |
| `htdemucs_ft` | 3 | 9.50 | 10.38 | 2.01 | 0 |
| `mel_band_roformer_kim` | 3 | 11.95 | 12.84 | 0.50 | 0 |
| `bs_roformer_ep317` | 3 | 11.37 | 12.28 | 1.02 | 0 |

## Per-fixture detail

| model | fixture | wall (s) | rtf | voc SDR | inst SDR | error |
|---|---|---:|---:|---:|---:|---|
| `htdemucs` | Actions_-_One_Minute_Smile | 4.4 | 0.64 | 9.34 | 10.04 |  |
| `htdemucs` | Al_James_-_Schoolboy_Facination | 3.9 | 0.57 | 9.34 | 10.27 |  |
| `htdemucs` | Signe_Jakobsen_-_What_Have_You_Done_To_Me | 3.9 | 0.57 | 11.66 | 14.60 |  |
| `htdemucs_ft` | Actions_-_One_Minute_Smile | 13.9 | 2.04 | 8.84 | 10.19 |  |
| `htdemucs_ft` | Al_James_-_Schoolboy_Facination | 13.6 | 2.00 | 9.50 | 10.38 |  |
| `htdemucs_ft` | Signe_Jakobsen_-_What_Have_You_Done_To_Me | 13.5 | 1.98 | 11.59 | 13.99 |  |
| `mel_band_roformer_kim` | Actions_-_One_Minute_Smile | 3.6 | 0.53 | 11.95 | 12.84 |  |
| `mel_band_roformer_kim` | Al_James_-_Schoolboy_Facination | 3.2 | 0.48 | 11.43 | 12.23 |  |
| `mel_band_roformer_kim` | Signe_Jakobsen_-_What_Have_You_Done_To_Me | 3.3 | 0.48 | 12.68 | 16.42 |  |
| `bs_roformer_ep317` | Actions_-_One_Minute_Smile | 7.0 | 1.03 | 11.18 | 12.28 |  |
| `bs_roformer_ep317` | Al_James_-_Schoolboy_Facination | 6.9 | 1.01 | 11.37 | 12.21 |  |
| `bs_roformer_ep317` | Signe_Jakobsen_-_What_Have_You_Done_To_Me | 6.9 | 1.02 | 12.70 | 15.34 |  |
# Quick ConvGRU decoupling screening

Blur-Mid/Blur-Max are diagnostic pressure conditions, not formal corruption severities.

| Model | Condition | mIoU | wIoU | mVC8 | mVC16 |
|---|---|---:|---:|---:|---:|
| Old ConvGRU | Clean | 0.545946 | 0.871496 | 0.953803 | 0.953853 |
| Semantic-only | Clean | 0.518544 | 0.852978 | 0.943019 | 0.937799 |
| Full-Decoupled | Clean | 0.549283 | 0.871021 | 0.954942 | 0.955539 |
| Old ConvGRU | Blur-Mid | 0.482770 | 0.828425 | 0.915455 | 0.890840 |
| Semantic-only | Blur-Mid | 0.443640 | 0.810379 | 0.909850 | 0.881528 |
| Full-Decoupled | Blur-Mid | 0.476822 | 0.832197 | 0.927734 | 0.913331 |
| Old ConvGRU | Blur-Max | 0.434220 | 0.772734 | 0.859036 | 0.806612 |
| Semantic-only | Blur-Max | 0.403796 | 0.753964 | 0.862857 | 0.815876 |
| Full-Decoupled | Blur-Max | 0.432093 | 0.792668 | 0.891965 | 0.857104 |

{
  "Full - semantic-only": {
    "Clean \u0394mIoU": 0.030739369857386123,
    "Blur-Mid \u0394mIoU": 0.03318178882643308,
    "Blur-Max \u0394mIoU": 0.028297319293292822,
    "Blur-Mid \u0394mVC16": 0.031803510662278045,
    "Blur-Max \u0394mVC16": 0.04122868558937143
  },
  "Full - old": {
    "Clean \u0394mIoU": 0.003336985047453722,
    "Blur-Mid \u0394mIoU": -0.0059477207541108745,
    "Blur-Max \u0394mIoU": -0.002126374889050653,
    "Blur-Mid \u0394mVC16": 0.022490909400630033,
    "Blur-Max \u0394mVC16": 0.05049262075711525
  }
}

TEMPORAL STATE: GO
DECOUPLED ARCHITECTURE: GO

TEMPORAL STATE: GO
DECOUPLED ARCHITECTURE: GO

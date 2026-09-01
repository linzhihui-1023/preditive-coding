# Quick ConvGRU decoupling screening

Blur-Mid/Blur-Max are diagnostic pressure conditions, not formal corruption severities.

| Model | Condition | mIoU | wIoU | mVC8 | mVC16 |
|---|---|---:|---:|---:|---:|
| Old ConvGRU | Clean | 0.546021 | 0.871514 | 0.953817 | 0.953871 |
| Semantic-only | Clean | 0.524196 | 0.857198 | 0.948125 | 0.944373 |
| Full-Decoupled | Clean | 0.553422 | 0.871101 | 0.954660 | 0.954408 |
| Old ConvGRU | Blur-Mid | 0.482855 | 0.828449 | 0.915467 | 0.890851 |
| Semantic-only | Blur-Mid | 0.434174 | 0.816094 | 0.915194 | 0.891837 |
| Full-Decoupled | Blur-Mid | 0.482208 | 0.831811 | 0.925922 | 0.910568 |
| Old ConvGRU | Blur-Max | 0.434258 | 0.772731 | 0.859021 | 0.806576 |
| Semantic-only | Blur-Max | 0.417280 | 0.765262 | 0.871194 | 0.827113 |
| Full-Decoupled | Blur-Max | 0.441097 | 0.791106 | 0.888273 | 0.852445 |

{
  "Full - semantic-only": {
    "Clean \u0394mIoU": 0.02922592586501871,
    "Blur-Mid \u0394mIoU": 0.048033740164752314,
    "Blur-Max \u0394mIoU": 0.023817135659136457,
    "Blur-Mid \u0394mVC16": 0.01873068880477513,
    "Blur-Max \u0394mVC16": 0.025331902983173227
  },
  "Full - old": {
    "Clean \u0394mIoU": 0.007401494821091981,
    "Blur-Mid \u0394mIoU": -0.0006477645319076797,
    "Blur-Max \u0394mIoU": 0.006839621664544715,
    "Blur-Mid \u0394mVC16": 0.019717275502889264,
    "Blur-Max \u0394mVC16": 0.045868713072221734
  }
}

TEMPORAL STATE: GO
DECOUPLED ARCHITECTURE: GO

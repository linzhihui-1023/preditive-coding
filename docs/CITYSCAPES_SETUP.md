# Cityscapes robustness setup

The local Cityscapes root is `/home/lin/datasets/cityscapes` (or
`CITYSCAPES_DATASET`). The repository does not contain Cityscapes images,
labels, credentials, cookies, or zip archives.

## Official download

Create an account at the official Cityscapes site, then run this from the
Predify Conda environment:

```bash
cd /home/lin/predify2021_selective_adaptation
CITYSCAPES_DATASET=/home/lin/datasets/cityscapes scripts/prepare_cityscapes.sh
```

The script uses the official `cityscapesscripts` downloader with resume and
requests only `leftImg8bit_trainvaltest.zip` and `gtFine_trainvaltest.zip`.
It preserves the archive directory layout. Do not put credentials in this
repository.

## Checks

After both archives are downloaded and extracted:

```bash
python -m predify2021.mce_scores.check_cityscapes_setup \
  --root /home/lin/datasets/cityscapes
```

The check reads one validation pair, maps official `labelIds` through
`cityscapesScripts`, and applies online ImageNet-C Gaussian blur at levels
1--5. It does not run a model or generate a corruption dataset.

## Published protocol

`predify2021/datasets/cityscapes_corruption_protocol.json` records the 19
corruptions used by Kamann and Rother: the 16 ImageNet-C-derived Cityscapes
transformations plus intensity-dependent noise, PSF blur, and geometric
distortion. The first 16 entries carry the severity values from the cited
`imagecorruptions` implementation. The three Kamann-Rother additions are
listed with their published descriptions and explicitly marked as incomplete
until the authors' numeric artifacts are available; no substitute values are
invented.

Local semantic metrics must use the official Cityscapes validation split.
Test labels are not public and must not be treated as local ground truth.

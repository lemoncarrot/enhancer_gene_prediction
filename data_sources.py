"""Shared filenames and source URLs for the enhancer-to-gene pipeline."""

import os
from pathlib import Path


ABC = "https://raw.githubusercontent.com/broadinstitute/ABC-Enhancer-Gene-Prediction/main/reference"
ENCODE = "https://www.encodeproject.org/files"

RAW_FILES = {
    "crispr": {
        "filename": "EPCrisprBenchmark_combined_data.training_K562.GRCh38.tsv.gz",
        "url": "https://raw.githubusercontent.com/EngreitzLab/CRISPR_comparison/main/resources/crispr_data/EPCrisprBenchmark_combined_data.training_K562.GRCh38.tsv.gz",
    },
    "genes": {
        "filename": "CollapsedGeneBounds.hg38.bed",
        "url": f"{ABC}/hg38/CollapsedGeneBounds.hg38.bed",
    },
    "blocklist": {
        "filename": "GRCh38_unified_blacklist.bed",
        "url": f"{ABC}/hg38/GRCh38_unified_blacklist.bed",
    },
    "atac_peaks": {
        "filename": "ENCFF558BLC.bed.gz",
        "url": f"{ENCODE}/ENCFF558BLC/@@download/ENCFF558BLC.bed.gz",
    },
    "atac_bam_1": {
        "filename": "ENCFF512VEZ.bam",
        "url": f"{ENCODE}/ENCFF512VEZ/@@download/ENCFF512VEZ.bam",
        "postprocess": "bam_index",
    },
    "atac_bam_2": {
        "filename": "ENCFF987XOV.bam",
        "url": f"{ENCODE}/ENCFF987XOV/@@download/ENCFF987XOV.bam",
        "postprocess": "bam_index",
    },
    "genome_fasta_gz": {
        "filename": "hg38.fa.gz",
        "url": "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz",
        "postprocess": "gunzip",
    },
    "genome_fasta": {
        "filename": "hg38.fa",
    },
}

# This large Hi-C file is read remotely rather than downloaded into raw/.
HIC_URL = "https://encode-public.s3.amazonaws.com/2022/05/15/0571c671-3645-4f92-beae-51dfd3f42c36/ENCFF621AIY.hic"


def data_directory():
    """Return the configured data root (relative to the working directory by default)."""
    return Path(os.environ.get("E2G_DATA_DIR", "data"))


def raw_dir(data_dir=None):
    return Path(data_dir if data_dir is not None else data_directory()) / "raw"


def processed_dir(data_dir=None):
    return Path(data_dir if data_dir is not None else data_directory()) / "processed"


def raw_path(raw_dir, key):
    """Return the local path for a named raw input."""
    return Path(raw_dir) / RAW_FILES[key]["filename"]

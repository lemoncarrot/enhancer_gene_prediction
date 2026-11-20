"""Download the raw inputs listed in data_sources.py.

Run from the project root: python download_data.py
Requires curl and samtools on PATH.
"""

import subprocess
from pathlib import Path

from data_sources import RAW_FILES, raw_dir, raw_path


def main():
    raw = raw_dir()
    raw.mkdir(parents=True, exist_ok=True)

    for key, source in RAW_FILES.items():
        if "url" not in source:
            continue  # hg38.fa is created by decompressing hg38.fa.gz

        path = raw_path(raw, key)
        if not path.is_file():
            partial = path.with_name(path.name + ".part")
            subprocess.run(
                ["curl", "-fL", "-C", "-", "-o", str(partial), source["url"]],
                check=True,
            )
            partial.replace(path)

        if source.get("postprocess") == "bam_index":
            if not Path(str(path) + ".bai").is_file():
                subprocess.run(["samtools", "index", str(path)], check=True)
        elif source.get("postprocess") == "gunzip":
            if not raw_path(raw, "genome_fasta").is_file():
                subprocess.run(["gunzip", "-k", str(path)], check=True)

    print(f"done: {raw.resolve()}")


if __name__ == "__main__":
    main()

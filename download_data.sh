#!/usr/bin/env bash
# Downloads raw data to ${E2G_DATA_DIR:-data}/raw (~10 GB). Re-runnable: skips finished files, resumes partial ones.
# Run inside the e2g conda env (needs samtools).
#
# Sources:
#   CRISPR benchmark   K562 CRISPRi enhancer-gene pairs with regulation labels (ENCODE-rE2G, Gschwind et al. 2023)
#   ABC gene list      RefSeq genes, one TSS each; defines promoters and the benchmark's distances
#   ENCODE blocklist   Artifact-prone regions excluded from candidate elements
#   K562 ATAC peaks    MACS2 peaks (ENCSR483RKN); seeds the candidate elements (graph nodes)
#   K562 ATAC BAMs     Aligned reads, 2 replicates (~6 GB); read counts give node activity
#   hg38               UCSC GRCh38 reference genome; node DNA sequences
#   K562 Hi-C          ENCFF621AIY.hic, intact Hi-C (34 GB); NOT downloaded, streamed in main.ipynb
set -euo pipefail
RAW="${E2G_DATA_DIR:-data}/raw"
mkdir -p "$RAW" && cd "$RAW"

get() { [ -f "$2" ] || { curl -fL -C - -o "$2.part" "$1" && mv "$2.part" "$2"; }; }
ENCODE=https://www.encodeproject.org/files
ABC=https://raw.githubusercontent.com/broadinstitute/ABC-Enhancer-Gene-Prediction/main/reference

get https://raw.githubusercontent.com/EngreitzLab/CRISPR_comparison/main/resources/crispr_data/EPCrisprBenchmark_combined_data.training_K562.GRCh38.tsv.gz \
    EPCrisprBenchmark_combined_data.training_K562.GRCh38.tsv.gz
get $ABC/hg38/CollapsedGeneBounds.hg38.bed CollapsedGeneBounds.hg38.bed
get $ABC/hg38/GRCh38_unified_blacklist.bed GRCh38_unified_blacklist.bed
get $ENCODE/ENCFF558BLC/@@download/ENCFF558BLC.bed.gz ENCFF558BLC.bed.gz
for acc in ENCFF512VEZ ENCFF987XOV; do
    get $ENCODE/$acc/@@download/$acc.bam $acc.bam
    [ -f $acc.bam.bai ] || samtools index $acc.bam
done
get https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz hg38.fa.gz
[ -f hg38.fa ] || gunzip -k hg38.fa.gz
echo "done: $(pwd)"

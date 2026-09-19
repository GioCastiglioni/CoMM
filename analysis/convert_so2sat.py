"""Convert the GeoBench So2Sat tortilla into per-split .npy arrays.

The `.tortilla` reader (`tacoreader`) needs Python >= 3.9 and the training
environment is 3.8, so the format is parsed directly here and the result is
written as memory-mappable arrays, the same route CREMA-D's features take.

Layout of the file: an 18-byte header (magic `#y`, then the footer's offset and
length as little-endian u64), the data blobs, and a Parquet footer holding the
index. Each top-level row is itself a tortilla whose two entries are the S1 and
S2 GeoTIFFs, so the parse recurses once.

Run under an environment with pyarrow (e.g. the `CropCon` env), not under the
training environment:

    conda activate CropCon
    python analysis/convert_so2sat.py --root ~/workspace/datasets/so2sat
"""
import argparse
import io
import json
import os
import struct

import numpy as np
import pyarrow.parquet as pq
import rasterio

SPLITS = ("train", "validation", "test")

# The Local Climate Zone order GeoBench itself uses. Sorting the labels instead
# would renumber the classes and stop per-class results matching published ones.
CLASSES = (
    "Compact high-rise", "Compact middle-rise", "Compact low-rise",
    "Open high-rise", "Open middle-rise", "Open low-rise",
    "Lightweight low-rise", "Large low-rise", "Sparsely built", "Heavy industry",
    "Dense Trees", "Scattered trees", "Bush, scrub", "Low plants",
    "Bare rock or paved", "Bare soil or sand", "Water",
)


def read_index(fh, base=0):
    """Return the Parquet index of the tortilla starting at `base`."""
    fh.seek(base)
    head = fh.read(18)
    if head[:2] != b"#y":
        raise ValueError(f"not a tortilla at offset {base}: magic {head[:2]!r}")
    offset, length = struct.unpack("<Q", head[2:10])[0], struct.unpack("<Q", head[10:18])[0]
    fh.seek(base + offset)
    return pq.read_table(io.BytesIO(fh.read(length))).to_pandas()


def read_raster(fh, offset, length):
    fh.seek(offset)
    with rasterio.io.MemoryFile(fh.read(length)) as mem, mem.open() as src:
        return src.read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--name", default="geobench_so2sat.tortilla")
    ap.add_argument("--limit", type=int, default=0, help="convert only N samples per split, for timing")
    args = ap.parse_args()

    src = os.path.join(args.root, args.name)
    out = os.path.join(args.root, "arrays")
    os.makedirs(out, exist_ok=True)

    with open(src, "rb") as fh:
        top = read_index(fh)
        seen = set(top["labels"].unique().tolist())
        if seen != set(CLASSES):
            raise ValueError(f"labels do not match the expected class set: {seen ^ set(CLASSES)}")
        classes = list(CLASSES)
        cls_idx = {c: i for i, c in enumerate(classes)}
        # The s1 band names follow GeoBench's declared `default_order` and its
        # "# order is vv, vh" comment. They are not verified against the pixels:
        # the two `normalization_stats` definitions in the source imply opposite
        # assignments, and nothing else in the distribution settles it. Nothing in
        # this pipeline depends on the names.
        meta = {"source": src, "classes": classes, "modalities": ["s1", "s2"],
                "bands": {"s1": ["VV", "VH"],
                          "s2": ["B02", "B03", "B04", "B05", "B06",
                                 "B07", "B08", "B8A", "B11", "B12"]},
                "splits": {}}

        for split in SPLITS:
            rows = top[top["tortilla:data_split"] == split].reset_index(drop=True)
            if args.limit:
                rows = rows.iloc[: args.limit]
            n = len(rows)
            s1 = s2 = None
            labels = np.zeros(n, dtype=np.int64)
            for i, row in rows.iterrows():
                base = int(row["tortilla:offset"])
                sub = read_index(fh, base)
                arrs = {}
                for _, sr in sub.iterrows():
                    arrs[sr["tortilla:id"]] = read_raster(
                        fh, base + int(sr["tortilla:offset"]), int(sr["tortilla:length"]))
                if s1 is None:
                    s1 = np.zeros((n, *arrs["s1"].shape), dtype=np.float32)
                    s2 = np.zeros((n, *arrs["s2"].shape), dtype=np.float32)
                s1[i], s2[i] = arrs["s1"], arrs["s2"]
                labels[i] = cls_idx[row["labels"]]
                if (i + 1) % 2000 == 0:
                    print(f"  {split} {i + 1}/{n}", flush=True)

            for tag, arr in (("s1", s1), ("s2", s2), ("labels", labels)):
                np.save(os.path.join(out, f"{split}_{tag}.npy"), arr)
            meta["splits"][split] = {"n": n, "s1": list(s1.shape[1:]), "s2": list(s2.shape[1:])}
            print(f"{split}: {n} samples  s1{s1.shape}  s2{s2.shape}", flush=True)

            # Statistics come from the training split only and are measured rather
            # than copied, which also sidesteps an ambiguity in the source: GeoBench's
            # `so2sat.py` defines `normalization_stats` twice and the two definitions
            # disagree on whether VV or VH owns each value. Measuring per channel is
            # correct under either reading, since channel i gets channel i's own
            # statistics; only the band *name* below is affected.
            if split == "train":
                meta["norm"] = {
                    m: {"mean": a.mean(axis=(0, 2, 3)).tolist(),
                        "std": a.std(axis=(0, 2, 3)).tolist()}
                    for m, a in (("s1", s1), ("s2", s2))}

    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()

"""Write dataset/catalog.json, the file every entry point reads its dataset paths from.

Paths default to a fixed layout under one root directory (see LAYOUT and the README);
any of them can be overridden with --path.

    python scripts/configure_catalog.py --root /data/remm
    python scripts/configure_catalog.py --root /data/remm --path trifeatures=/scratch/trifeatures
    python scripts/configure_catalog.py --root /data/remm --dry-run
"""
import argparse
import json
import os

CATALOG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset", "catalog.json")

LAYOUT = {
    "trifeatures": "trifeatures",
    "mosi": "multibench/mosi_data.pkl",
    "mosei": "multibench/mosei_senti_data.pkl",
    "humor": "multibench/humor.pkl",
    "sarcasm": "multibench/sarcasm.pkl",
    "visionandtouch": "multibench/triangle_real_data",
    "mimic": "mimic/im.pk",
    "so2sat": "so2sat/arrays",
    "hateful_memes": "hateful_memes",
    "mmimdb": "mmimdb",
    "crema_d": "crema_d",
    "sen1floods11": "sen1floods11",
}
# Image-text datasets read their split files from `metadata`, which is the same directory.
WITH_METADATA = ("hateful_memes", "mmimdb")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="directory holding the datasets")
    ap.add_argument("--path", action="append", default=[], metavar="NAME=PATH",
                    help="override the path of one dataset (repeatable)")
    ap.add_argument("--dry-run", action="store_true", help="print the catalog without writing it")
    args = ap.parse_args()

    paths = {name: os.path.join(os.path.abspath(args.root), rel) for name, rel in LAYOUT.items()}
    for item in args.path:
        name, sep, path = item.partition("=")
        if not sep or name not in LAYOUT:
            ap.error(f"--path expects NAME=PATH with NAME in {sorted(LAYOUT)}, got {item!r}")
        paths[name] = os.path.abspath(path)

    catalog = {}
    for name, path in paths.items():
        catalog[name] = {"path": path, "metadata": path} if name in WITH_METADATA else {"path": path}
        print(f"{name:15s} {path}{'' if os.path.exists(path) else '   (not found)'}")

    if args.dry_run:
        return
    with open(CATALOG, "w") as f:
        json.dump(catalog, f, indent=4)
        f.write("\n")
    print("written:", CATALOG)


if __name__ == "__main__":
    main()

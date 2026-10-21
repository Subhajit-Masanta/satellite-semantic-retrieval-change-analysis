"""Dataset acquisition helper.

    python download_data.py            # print where to get everything
    python download_data.py --check    # verify what is already on disk

The datasets are NOT auto-downloaded: LEVIR-CD lives behind Google Drive and
OSCD behind an institutional mirror, both of which need a browser. This script
tells you exactly where to click and what layout to end up with, then verifies
you got it right.
"""

import argparse
from pathlib import Path

DATA = Path("data")

DATASETS = {
    "LEVIR-CD": {
        "role": "PRIMARY -- all headline Precision@k / mAP / F1 / IoU numbers",
        "size": "637 pairs, 1024x1024, ~0.5 m/px, approx 3.8 GB",
        "labels": "pixel-level binary change masks (expert annotated)",
        "urls": [
            "https://justchenhao.github.io/LEVIR/           (project page)",
            "https://www.kaggle.com/datasets/mdrifaturrahman33/levir-cd-change-detection",
        ],
        "layout": "data/LEVIR-CD/{train,val,test}/{A,B,label}/*.png",
        "note": "A = time 1, B = time 2, label = ground-truth change mask. "
                "Pairs are already co-registered -- that is deliberate, general "
                "registration is out of scope (see PROJECT_PLAN.md).",
    },
    "OSCD": {
        "role": "SECONDARY -- proves transfer to genuine Sentinel-2 multispectral",
        "size": "24 pairs, Sentinel-2, 2015-2018, small (~100s of MB)",
        "labels": "pixel-level change ground truth, urban change",
        "urls": [
            "https://rcdaudt.github.io/oscd/                (project page)",
            "https://ieee-dataport.org/open-access/oscd-onera-satellite-change-detection",
            "https://huggingface.co/datasets/blanchon/OSCD_MSI   (easiest mirror)",
        ],
        "layout": "data/OSCD/<city>/{imgs_1,imgs_2}/ + labels",
        "note": "Real Sentinel-2, so this is the dataset that answers 'does it "
                "work on actual satellite data, not aerial crops?'",
    },
    "Sentinel-2 / Landsat / Bhuvan": {
        "role": "QUALITATIVE DEMO ONLY -- never enters a metrics table",
        "size": "2-3 hand-picked AOI tile pairs",
        "labels": "NONE -- this is why they are qualitative only",
        "urls": [
            "https://dataspace.copernicus.eu/        (Sentinel-1/2, free account)",
            "https://earthexplorer.usgs.gov/         (Landsat C2, free account)",
            "https://bhuvan-app3.nrsc.gov.in/data/download/   (NRSC/ISRO)",
            "https://bhoonidhi.nrsc.gov.in/          (newer NRSC portal)",
        ],
        "layout": "data/live/<aoi_name>/{t1,t2}.tif",
        "note": "These are the imagery sources PS SIH26227 section 7.1 names. "
                "Ingesting them shows applicability to the intended operational "
                "data. With no ground truth they stay qualitative -- see the "
                "metric validity rule in PROJECT_PLAN.md.",
    },
}


def show():
    print("=" * 74)
    print("DATASETS -- Semantic Retrieval & Change Analysis (SIH26227)")
    print("=" * 74)
    for name, d in DATASETS.items():
        print(f"\n### {name}")
        print(f"  role   : {d['role']}")
        print(f"  size   : {d['size']}")
        print(f"  labels : {d['labels']}")
        print("  get it :")
        for u in d["urls"]:
            print(f"     - {u}")
        print(f"  layout : {d['layout']}")
        print(f"  note   : {d['note']}")
    print("\n" + "=" * 74)
    print("Person A: start the LEVIR-CD download FIRST -- it is ~3.8 GB and it")
    print("is the one task that blocks integration if left late.")
    print("=" * 74)


def check():
    print("Checking data/ ...\n")
    if not DATA.exists():
        print(f"  {DATA.resolve()} does not exist yet.")
        print("  Run  python download_data.py  for where to get the datasets.")
        return 1

    ok = True

    levir = DATA / "LEVIR-CD"
    if levir.is_dir():
        found = False
        for split in ("train", "val", "test", ""):
            root = levir / split if split else levir
            a, b, l = root / "A", root / "B", root / "label"
            if a.is_dir() and b.is_dir():
                na, nb = len(list(a.glob("*.png"))), len(list(b.glob("*.png")))
                nl = len(list(l.glob("*.png"))) if l.is_dir() else 0
                flag = "OK " if na == nb and na > 0 else "!! "
                print(f"  {flag}LEVIR-CD/{split or '.'}: A={na} B={nb} label={nl}")
                if na != nb:
                    print("      A and B counts differ -- pairs will not line up.")
                    ok = False
                found = True
        if not found:
            print("  !! LEVIR-CD/ exists but has no A/ B/ subfolders.")
            ok = False
    else:
        print("  -- LEVIR-CD not present (PRIMARY dataset, needed for all numbers)")
        ok = False

    oscd = DATA / "OSCD"
    print(f"  {'OK ' if oscd.is_dir() else '-- '}OSCD "
          f"{'present' if oscd.is_dir() else 'not present (secondary)'}")

    live = DATA / "live"
    if live.is_dir():
        n = len(list(live.glob("*/t1.*")))
        print(f"  OK live/: {n} AOI pair(s) for qualitative demo")
    else:
        print("  -- live/ not present (qualitative demo tiles, optional)")

    print("\nReady for real runs." if ok else
          "\nNot ready: get LEVIR-CD first. Meanwhile everything runs on "
          "synthetic data:\n  python gates.py --synthetic\n  python main.py --demo")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="verify what is on disk")
    args = ap.parse_args()
    raise SystemExit(check() if args.check else (show() or 0))

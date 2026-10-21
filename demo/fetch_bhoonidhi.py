"""Search and download Resourcesat LISS-IV scenes from the Bhoonidhi API.

Login comes from the environment, never from a file in the project:
    set BHOONIDHI_USER=...   set BHOONIDHI_PASS=...

    python demo/fetch_bhoonidhi.py list 2023          # Online scenes Jan-Mar 2023 over Vellore
    python demo/fetch_bhoonidhi.py get 2023 <item_id> [<item_id> ...]

Only products with Online=Y can be fetched through the API. Online=N ones
(red download icon in the web cart) have to be downloaded from Browse & Order.
"""
import json
import os
import sys
from pathlib import Path

import requests

API = "https://bhoonidhi-api.nrsc.gov.in"
COLLECTIONS = ["ResourceSat-2_LISS4-MX70_L2", "ResourceSat-2A_LISS4-MX70_L2"]
BBOX = [79.04, 12.82, 79.22, 13.00]  # Vellore city + VIT
OUT = Path(__file__).resolve().parent.parent / "data" / "bhoonidhi"


def token():
    user, pw = os.environ.get("BHOONIDHI_USER"), os.environ.get("BHOONIDHI_PASS")
    if not user or not pw:
        sys.exit("Set BHOONIDHI_USER and BHOONIDHI_PASS first.")
    r = requests.post(f"{API}/auth/token", timeout=60,
                      json={"userId": user, "password": pw, "grant_type": "password"})
    if r.status_code != 200:
        sys.exit(f"Login failed: {r.status_code} {r.text[:300]}")
    return {"Authorization": "Bearer " + r.json()["access_token"]}


def search(headers, year):
    body = {
        "collections": COLLECTIONS,
        "datetime": f"{year}-01-01T00:00:00Z/{year}-03-31T23:59:59Z",
        "bbox": BBOX,
        "filter": {"args": [{"property": "Online"}, "Y"], "op": "eq"},
        "filter-lang": "cql2-json",
        "limit": 100,
    }
    r = requests.post(f"{API}/data/search", json=body, headers=headers, timeout=60)
    if r.status_code != 200:
        sys.exit(f"Search failed: {r.status_code} {r.text[:300]}")
    return r.json().get("features", [])


def download(headers, year, collection, item_id):
    folder = OUT / str(year)
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / f"{item_id}.zip"
    with requests.get(f"{API}/download", headers=headers, timeout=(30, 900), stream=True,
                      params={"id": item_id, "collection": collection}) as r:
        if r.status_code != 200:
            print(f"  {item_id}: HTTP {r.status_code} {r.text[:200]}")
            return
        with open(dest, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    print(f"  saved {dest} ({dest.stat().st_size / 1e6:.0f} MB)")


def main():
    if len(sys.argv) < 3 or sys.argv[1] not in ("list", "get"):
        sys.exit(__doc__)
    headers = token()
    year = sys.argv[2]
    feats = search(headers, year)
    if sys.argv[1] == "list":
        print(f"{len(feats)} online scenes in Jan-Mar {year}")
        for i, ft in enumerate(feats):
            p = ft.get("properties", {})
            if i == 0:
                print("property names:", sorted(p))
            print(ft["collection"], ft["id"], json.dumps(p)[:200])
        return
    wanted = set(sys.argv[3:])
    for ft in feats:
        if ft["id"] in wanted:
            download(headers, year, ft["collection"], ft["id"])


if __name__ == "__main__":
    main()

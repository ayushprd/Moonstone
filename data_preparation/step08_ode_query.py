"""Query ODE REST API for WAC monthly mosaic catalog (Tier 1.5 prep).

Discovers all available WAC monthly global mosaics (SDPWMG products)
and saves a catalog for future multi-angular stack downloads.

Usage:
    python step08_ode_query.py
"""

import argparse
import json
import sys
from pathlib import Path

import requests

from config import ODE_API_URL, RAW_DIR


def query_ode_catalog(output_path: Path = None) -> dict:
    """Query ODE REST API for all WAC monthly mosaics.

    Returns parsed catalog with product details.
    """
    if output_path is None:
        output_path = RAW_DIR / "wac_monthly_catalog.json"

    print("=== ODE REST API: WAC Monthly Global Mosaics (SDPWMG) ===")
    print(f"  URL: {ODE_API_URL}")

    # Add results=m to get full metadata (default returns count only)
    url = ODE_API_URL + "&results=m"
    try:
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  ERROR: API query failed: {e}")
        return {}

    # Parse response
    ode = data.get("ODEResults", {})
    count = ode.get("Count", 0)
    print(f"  Total products: {count}")

    products = ode.get("Products", {}).get("Product", [])
    if isinstance(products, dict):
        products = [products]

    # Extract key fields from each product
    catalog = {"query_url": url, "total_count": count, "products": []}

    for p in products:
        entry = {
            "product_name": p.get("Product_name", ""),
            "description": p.get("Description", ""),
            "external_url": p.get("External_url", ""),
            "label_url": p.get("LabelURL", ""),
            "files_url": p.get("FilesURL", ""),
            "product_url": p.get("ProductURL", ""),
            "start_time": p.get("UTC_start_time", ""),
            "stop_time": p.get("UTC_stop_time", ""),
            "observation_time": p.get("Observation_time", ""),
            "resolution": p.get("Map_resolution_text", ""),
        }
        catalog["products"].append(entry)

    # Save catalog
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(catalog, f, indent=2)

    print(f"  Catalog saved: {output_path}")
    print(f"  Products cataloged: {len(catalog['products'])}")

    # Print summary
    if catalog["products"]:
        print(f"\n  Products:")
        for i, p in enumerate(catalog["products"][:10]):
            name = p["product_name"]
            start = p["start_time"][:10] if p["start_time"] else "?"
            stop = p["stop_time"][:10] if p["stop_time"] else "?"
            print(f"    {i+1}. {name} ({start} to {stop})")
        if len(catalog["products"]) > 10:
            print(f"    ... and {len(catalog['products']) - 10} more")

        # Date range
        starts = [p["start_time"] for p in catalog["products"] if p["start_time"]]
        stops = [p["stop_time"] for p in catalog["products"] if p["stop_time"]]
        if starts:
            print(f"\n  Date range: {min(starts)[:10]} to {max(stops)[:10]}")
            print(f"  Total mosaic products: {len(catalog['products'])}")

    return catalog


def main():
    parser = argparse.ArgumentParser(
        description="Query ODE for WAC monthly mosaic catalog"
    )
    parser.add_argument(
        "--output", type=Path,
        default=RAW_DIR / "wac_monthly_catalog.json",
        help="Output catalog JSON path"
    )
    args = parser.parse_args()

    catalog = query_ode_catalog(args.output)

    if not catalog.get("products"):
        print("\n  WARNING: No products found. The API may have returned an unexpected format.")
        print("  Check the raw JSON response for debugging.")

    print("\n=== ODE query complete ===")


if __name__ == "__main__":
    main()

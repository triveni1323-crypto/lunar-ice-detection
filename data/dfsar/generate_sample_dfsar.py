import csv
from pathlib import Path


def generate_sample_dfsar_csv(output_path="data/dfsar/sample_dfsar_radar.csv"):
    rows = []
    for y in range(4):
        for x in range(5):
            base_backscatter = -14.0 - y * 0.7 + (x * 0.3)
            if x >= 3 and y >= 1:
                base_backscatter += 2.0
            slope = 6 + x * 0.5 + y * 0.8
            shadow = 0.75 if (x + y) >= 3 else 0.35
            roughness = 0.18 + (x * 0.04) + (y * 0.05)
            ice_label = 1 if (x + y) >= 3 else 0
            rows.append({
                "x": x,
                "y": y,
                "backscatter_db": round(base_backscatter, 2),
                "slope_deg": round(slope, 2),
                "shadow_probability": round(shadow, 2),
                "roughness": round(roughness, 2),
                "ice_label": ice_label,
            })

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["x", "y", "backscatter_db", "slope_deg", "shadow_probability", "roughness", "ice_label"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Created DFSAR sample dataset at: {path}")


if __name__ == "__main__":
    generate_sample_dfsar_csv()

from pathlib import Path


class DataLoader:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)

    def list_files(self, subdir=""):
        folder = self.data_dir / subdir
        if folder.exists():
            return sorted(str(p) for p in folder.iterdir())
        return []

    def load_raster(self, file_path):
        try:
            import rasterio
            with rasterio.open(file_path) as src:
                data = src.read()
                meta = src.meta.copy()
                return data, meta
        except Exception as e:
            print(f"Error loading raster: {e}")
            return None, None

    def load_csv(self, file_path):
        import pandas as pd
        return pd.read_csv(file_path)

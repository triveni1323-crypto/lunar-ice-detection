import matplotlib.pyplot as plt


def plot_ice_map(ice_map):
    plt.figure(figsize=(8, 8))
    plt.imshow(ice_map, cmap="viridis")
    plt.colorbar(label="Ice Probability")
    plt.title("Lunar Ice Probability Map")
    plt.show()


def plot_terrain_profile(data):
    plt.figure(figsize=(8, 4))
    plt.plot(data)
    plt.title("Terrain Profile")
    plt.xlabel("Pixel Index")
    plt.ylabel("Elevation / Intensity")
    plt.show()

import pystac_client
import rasterio 
from rasterio.windows import from_bounds
from rasterio.warp import transform_geom
from shapely.geometry import box, shape, mapping
import numpy as np
import matplotlib.pyplot as plt
from rasterio.plot import show


# Define a bounding box for an AOI (Ottawa) in EPSG:4326
bbox=[-75.8860,45.3157,-75.5261,45.5142]
bbox_crs = "EPSG:4326"

# Link to ccmeo datacube stac-api
stac_root = "https://datacube.services.geo.ca/stac/api"
catalog = pystac_client.Client.open(stac_root)

search = catalog.search(
    collections=['mrdem-30'], 
    bbox=bbox,
    ) 

# Get the link to the data asset for mrdem-30 dtm
links = []
for page in search.pages():
    for item in page:
        links.append(item.assets['dtm'].href)

# Read AOI from the first COG
with rasterio.open(links[0]) as src:
    # Diagnostics: print dataset CRS and bounds
    print("Dataset CRS:", src.crs)
    print("Dataset bounds:", src.bounds)
    print("AOI bbox (input CRS", bbox_crs, "):", bbox)

    # Transform bbox from bbox_crs -> src.crs (previous code had args reversed)
    geom4326 = mapping(box(*bbox))
    transformed_geom = transform_geom(bbox_crs, src.crs, geom4326)
    transformed_bbox = shape(transformed_geom).bounds
    print("Transformed bbox (dataset CRS):", transformed_bbox)
    # Define the window to read the values
    window = from_bounds(transformed_bbox[0], transformed_bbox[1], 
                         transformed_bbox[2], transformed_bbox[3], 
                         src.transform)
    print("Computed window:", window)
    print("Window width x height:", getattr(window, 'width', None), getattr(window, 'height', None))
    # Read value from file
    rst = src.read(1, window=window)
    print("Read array shape:", rst.shape)
    # Compute basic stats if array not empty
    if rst.size:
        try:
            print("data min/max:", np.nanmin(rst), np.nanmax(rst))
        except Exception:
            print("Could not compute min/max (array may be masked)")
    else:
        print("Read array is empty (zero size)")

    # Copy and update the source metadata to write the output tiff
    metadata = src.meta.copy()
    metadata.update({
        'height': window.height,
        'width': window.width,
        'transform': rasterio.windows.transform(window, src.transform)
    }) 

# Perform analysis ...

# Display the raster data as a visual
fig, ax = plt.subplots(figsize=(12, 10))
show(rst, ax=ax, transform=rasterio.windows.transform(window, src.transform), 
     cmap='terrain', title='MRDEM-30 DTM - AOI (Ottawa)')
plt.colorbar(ax.images[0], ax=ax, label='Elevation (meters)')
plt.tight_layout()
plt.show()

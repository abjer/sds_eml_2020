import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import matplotlib.pyplot as plt

from scipy.spatial import Voronoi

# The function below is taken from https://gist.github.com/pv/8036995

def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.
    
    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.
    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.
        
    
    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()*2

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    
    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)


# Making a wrapper function for geopandas

def voronoi_wrapper(points, origin_geom):
    '''
    Returns a series of Voronoi shapely objects contruced
    from the centroids.
    '''
    
    vor = Voronoi(points)
    
    regions, vertices = voronoi_finite_polygons_2d(vor)
    
    def _polygon_intersect(points):
        poly = shapely.geometry.Polygon(points)
        poly_intersect_origin = poly.intersection(origin_geom)
        return poly_intersect_origin
    
    poly_list = [_polygon_intersect(vertices[region]) for region in regions]
    
    return gpd.GeoSeries(poly_list, index=points.index)

from shapely.geometry import box, Point

def make_vor_plot(n_points=6):

    f, ax = plt.subplots(1, 2, figsize=(12, 6))

    coords = pd.DataFrame(np.random.rand(n_points,2))
    gpd.GeoSeries(coords.apply(Point,axis=1)).plot(ax=ax[0])
    ax[0].set_xlim(0,1)
    ax[0].set_ylim(0,1)

    coords_vor = gpd.GeoDataFrame(geometry=voronoi_wrapper(coords, box(0,0,1,1)),
                                  data={'number':range(n_points)})
    coords_vor.plot(ax=ax[1], column='number')
    

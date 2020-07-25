

import matplotlib.pyplot as plt

import rasterio
import shapely
import geopandas as gpd

from copy import copy

from pandas import Series
from shapely.geometry import Polygon
from geopandas import GeoDataFrame


def plot_raster_polygon(raster,polygon,ax=None,**kwargs): #TODO
    '''

    :param raster: A rasterio dataset object opened in 'r' mode
    :param polygon: shapely.Polygon
    :param ax: matplotlib axes
    :param kwargs: Passed first to rasterio.plot.show() and then to the matplotlib imshow or contour method depending on contour argument.
    See full lists at:
    https://rasterio.readthedocs.io/en/latest/api/rasterio.plot.html#rasterio.plot.show and
    http://matplotlib.org/api/axes_api.html?highlight=imshow#matplotlib.axes.Axes.imshow or
    http://matplotlib.org/api/axes_api.html?highlight=imshow#matplotlib.axes.Axes.contour
    :return: A matplotlib Axes
    '''
    out_image, out_transform = rasterio.mask.mask(raster, gpd.GeoSeries(polygon), crop=True, filled=False)
    out_meta = raster.meta
    # out_meta.update({'driver': 'GTiff',
    #                  "height": out_image.shape[1],
    #                  "width": out_image.shape[2],
    #                  "transform": out_transform})
    return rasterio.plot.show(out_image, ax=ax, transform=out_transform,**kwargs)


def _polygon_from_treeObj(treeObj):
    if isinstance(treeObj, GeoDataFrame):
        l = len(treeObj)
        if l == 1:
            tree = treeObj.iloc[0].geometry
        else:
            raise TypeError(f"The treeObj was a GeoDataFrame with length {l}. "
                            f"Only a single tree as a Polygon, single-row-GDF or Series"
                            f" containing a Polygon are supported.")
    elif isinstance(treeObj, Series):
        tree = treeObj.geometry
    elif isinstance(treeObj, Polygon):
        tree = treeObj
    else:
        raise TypeError("Only GDF, Series or Polygon supported.")
    return tree

def plot_tree(raster,tree,ax=None, simplify=False, simplify_tolerance=0,
    box_scale = 1.5,
    param_polygon={},param_box={},param_boundary={}):
    '''

    :param raster: Rasterio dataset object
    :param tree: Shapely tree polygon, or Series/single-row GeoDF containing geometry field
    :param ax: Matplotlib Axes object
    :param simplify: Simplify polygon, defualt is False
    :param simplify_tolerance: Maximum distance of the simplified polygon from the original.
    :param box_scale: How much wider should the box surrounding tree be.
        1.0 means original envelope, default 1.5
    :param param_polygon: Paramaters passed to the polygon raster plotting function.
        > plot_raster_polygon
            > rasterio.plot.show
                > matplotlib.imshow
                > matplotlib.contour
    :param param_box: Parameters passed to the box plotting function, same as above.
    :param param_boundary: Parameters passed to the tree outline plot, can influence the line or point parameters
        > geopandas.plotting.plot_series
    :return: A Matplotlib Axes containing all the plots overlaid
    :raises TypeError:
    '''
    show = False
    if ax is None:
        show = True
        ax = plt.gca()

    tree = _polygon_from_treeObj(tree)

    if simplify:
        tree=tree.simplify(simplify_tolerance)


    # plot tree pixels
    plot_raster_polygon(raster, tree, ax=ax, **param_polygon)
    # plot outline
    par_boundary = dict(color='r', alpha=1, linewidth=1)
    par_boundary.update(param_boundary)
    gpd.GeoSeries(tree.boundary).plot(ax=ax,**par_boundary)
    # box around plot
    box = shapely.affinity.scale(tree.envelope, xfact=box_scale, yfact=box_scale)
    plot_raster_polygon(raster, box, ax=ax, alpha=0.6, **param_box)

    if show:
        plt.show()
    return ax

def plot_tree_hist(raster,tree,ax=None,bins=50,**hist_kwargs):
    '''
    Show histogram of tree pixel values.

    :param raster: Rasterio dataset object
    :param tree: Shapely tree polygon, or Series/single-row GeoDF containing geometry field
    :param ax: Matplotlib Axes object
    :param bins: Histogram bins, as a number or other format supported by matplotlib histogram.
    :param hist_kwargs: Passed to matplotlib.hist
    :return: Matplotlib Axes
    '''
    show = False
    if ax is None:
        show = True
        ax = plt.gca()

    tree=_polygon_from_treeObj(tree)
    out_image, out_transform = rasterio.mask.mask(raster,gpd.GeoSeries(tree), crop=True, filled=False)

    ax.hist(out_image.flatten(),bins=bins,**hist_kwargs)

    if show:
        plt.show()
    return ax

def plot_raster(raster,ax=None,**kwargs):
    show = False
    if ax is None:
        show = True
        ax = plt.gca()

    ax.imshow(raster,**kwargs)


    if show:
        plt.show()
    return ax
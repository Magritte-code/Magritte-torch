import matplotlib.pyplot    as plt                  # Mpl plotting
import matplotlib                                   # Mpl
import plotly.graph_objects as go                   # Plotly plotting
from magrittetorch.model.model import Model         # Magritte model
from magrittetorch.model.image import ImageType     # Image type
from magrittetorch.tools import radiativetransferutils as rtutils
import numpy                as np                   # Data structures
import os                                           # Creating directories
import warnings                                     # Hide warnings
warnings.filterwarnings('ignore')                   # especially for yt
from typing import Optional, Union, List, Tuple     # Type hinting

from matplotlib.gridspec  import GridSpec           # Plot layout
from plotly.subplots      import make_subplots      # Plotly subplots
from astropy              import constants, units   # Unit conversions
from astropy.units        import Unit, Quantity #type hint
from astropy.io           import fits               # Fits file handling
from scipy.interpolate    import griddata           # Grid interpolation
from palettable.cubehelix import cubehelix2_16      # Nice colormap
from tqdm                 import tqdm               # Progress bars
from ipywidgets           import interact           # Interactive plots
from ipywidgets.embed     import embed_minimal_html # Store interactive plots
from math                 import floor, ceil        # Math helper functions
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


# Port matplotlib colormaps to plotly
cubehelix2_16_rgb = []
for i in range(0, 255):
    cubehelix2_16_rgb.append(
        matplotlib.colors.colorConverter.to_rgb(
            cubehelix2_16.mpl_colormap(
                matplotlib.colors.Normalize(vmin=0, vmax=255)(i)
            )
        )
    )

def matplotlib_to_plotly(cmap, pl_entries):
    """
    Converts matplotlib cmap to plotly colorscale.
    """
    h = 1.0/(pl_entries-1)
    pl_colorscale = []

    for k in range(pl_entries):
        C = list(map(np.uint8, np.array(cmap(k*h)[:3])*255))
        pl_colorscale.append([k*h, 'rgb'+str((C[0], C[1], C[2]))])
    return pl_colorscale

# Plotly compatible colorscale
cubehelix2_16_plotly = matplotlib_to_plotly(cubehelix2_16.mpl_colormap, 255)

# Plotly standard config
modeBarButtonsToRemove = ['select2d', 'lasso2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'hoverClosestCartesian', 'hoverCompareCartesian', 'zoom3d', 'pan3d', 'resetCameraDefault3d', 'resetCameraLastSave3d', 'hoverClosest3d', 'orbitRotation', 'tableRotation', 'zoomInGeo', 'zoomOutGeo', 'resetGeo', 'hoverClosestGeo', 'sendDataToCloud', 'hoverClosestGl2d', 'hoverClosestPie', 'toggleHover', 'resetViews', 'toggleSpikelines', 'resetViewMapbox']


def image_mpl(
        model: Model,
        image_nr: int   =  -1,
        zoom: float     = 1.3,
        npix_x: int     = 256,
        npix_y: int     = 256,
        x_unit: Unit    = units.au,
        v_unit: Unit    = units.km/units.s,
        method: str     = 'nearest'
    ):
    """
    Create plots of the channel maps of a synthetic observation (image) with matplotlib.

    Parameters
    ----------
    model : object
        Magritte model object.
    image_nr : int
        Number of the synthetic observation to plot. (Use -1 to indicate the last one.)
    zoom : float
        Factor with which to zoom in on the middel of the image.
    npix_x : int
        Number of pixels in the image in the horizontal (x) direction.
    npix_y : int
        Number of pixels in the image in the vertical (y) direction.
    x_unit : astropy.Unit object
        Unit of length for the horixontal (x) axis.
    y_unit : astropy.Unit object
        Unit of length for the vertical (y) axis.
    method : str
        Method to interpolate the scattered intensity data onto a regular image grid.

    Returns
    -------
    None
    """
    # Check if there are images
    if (len(model.images) < 1):
        print('No images in model.')
        return

    # Get path of image directory
    im_dir = os.path.dirname(os.path.abspath(model.io.save_location)) + '/images/'

    # If no image directory exists yet
    if not os.path.exists(im_dir):
        # Create image directory
        os.makedirs(im_dir)
        print('Created image directory:', im_dir)

    # Extract data of last image
    imx = model.images[image_nr].imX.get_astropy() #dims: [image.npix]
    imy = model.images[image_nr].imY.get_astropy() #dims: [image.npix]
    imI = model.images[image_nr].I.get_astropy() #dims: [image.npix, image.nfreqs]

    # Extract the number of frequency bins
    nfreqs = model.images[image_nr].nfreqs.get()

    # Set image boundaries
    deltax = (np.max(imx) - np.min(imx))/zoom
    midx = (np.max(imx) + np.min(imx))/2.0
    deltay = (np.max(imy) - np.min(imy))/zoom
    midy = (np.max(imy) + np.min(imy))/2.0

    x_min, x_max = midx - deltax/2.0, midx + deltax/2.0
    y_min, y_max = midy - deltay/2.0, midy + deltay/2.0

    # Create image grid values
    xs: Quantity = np.linspace(x_min, x_max, npix_x)
    ys: Quantity = np.linspace(y_min, y_max, npix_y)

    # Extract the spectral / velocity data
    freqs = model.images[image_nr].freqs.get_astropy() #dims: [image.nfreqs]
    f_ij  = np.mean(freqs)
    velos = (f_ij - freqs) / f_ij * constants.c

    # Interpolate the scattered data to an image (regular grid)
    # Euhm, the units are not preserved here, so I have to add them back in; for optical depths, these are the wrong units, though the units will not be used for anything
    Is = np.zeros((nfreqs)) * units.W / units.m**2 / units.sr / units.Hz
    zs: Quantity = np.zeros((nfreqs, npix_x, npix_y)) * units.W / units.m**2 / units.sr / units.Hz
    for f in range(nfreqs):
        # Nearest neighbor interpolate scattered image data
        zs[f] = griddata(
            (imx, imy),
            imI[:,f],
            (xs[None,:], ys[:,None]),
            method=method,
            fill_value = 0.0 #for non-nearest neighbor interpolation, otherwise the ceil/floor functions will complain
        )   * units.W / units.m**2 / units.sr / units.Hz # This scipy function does not preserve the units
        Is[f] = np.sum(zs[f])
    Is = Is / np.max(Is) # Normalized intensity, thus unitless

    # Put zero/negative values to the smallest positive value
    zs[zs<=0.0] = np.min(zs[zs>0.0])
    # Put nan values to smallest positive value
    zs[np.isnan(zs)] = np.min(zs[zs>0.0])

    # Get the logarithm of the data (matplotlib has a hard time handling logarithmic data.)
    log_zs     = np.log10(zs.value)
    log_zs_min = np.min(log_zs)
    log_zs_max = np.max(log_zs)

    lzmin = ceil (log_zs_min)
    lzmax = floor(log_zs_max)

    lz_25 = ceil (log_zs_min + 0.25*(log_zs_max - log_zs_min))
    lz_50 = ceil (log_zs_min + 0.50*(log_zs_max - log_zs_min))
    lz_75 = floor(log_zs_min + 0.75*(log_zs_max - log_zs_min))

    ticks  = [lzmin, lz_25, lz_50, lz_75, lzmax]
    levels = np.linspace(log_zs_min, log_zs_max, 250)

    figs = []
    gs   = GridSpec(1,2, wspace=.1, width_ratios=[2, 1])

    for f in tqdm(reversed(range(nfreqs)), total=nfreqs):
        fig = plt.figure(dpi=300)
        ax1 = fig.add_subplot(gs[0])
        ax  = ax1.contourf(
            xs.to(x_unit),
            ys.to(x_unit),
            log_zs[f],
            cmap=cubehelix2_16.mpl_colormap,
            levels=levels
        )
        ax0 = inset_axes(
                  ax1,
                  width="100%",
                  height="5%",
                  loc='lower left',
                  bbox_to_anchor=(0, 1.025, 1, 1),
                  bbox_transform=ax1.transAxes,
                  borderpad=0
        )

        cbar = fig.colorbar(ax, cax=ax0, orientation="horizontal")
        ax0.xaxis.set_ticks_position('top')
        ax0.xaxis.set_label_position('top')
        ax0.xaxis.set_ticks         (ticks)
        ax0.xaxis.set_ticklabels    ([f'$10^{{{t}}}$' for t in ticks])

        ax1.set_aspect('equal')
        ax1.set_xlabel(f'image x [{x_unit}]', labelpad = 10)
        ax1.set_ylabel(f'image y [{x_unit}]', labelpad = 10)

        ax2 = fig.add_subplot(gs[1])
        ax2.plot(velos.to(v_unit).value, Is/np.max(Is))
        ax2.yaxis.set_label_position("right")
        ax2.yaxis.tick_right()
        ax2.axvline(velos[f].to(v_unit).value, c='red')
        ax2.set_xlabel(f'velocity [{v_unit}]', labelpad=10)
        asp = 2*np.diff(ax2.get_xlim())[0] / np.diff(ax2.get_ylim())[0]
        ax2.set_aspect(asp)

        if   (model.images[image_nr].imageType.get() == ImageType.Intensity):
            ax0.set_xlabel('Intensity [W m$^{-2}$ sr$^{-1}$ Hz$^{-1}$]', labelpad=11)
            ax2.set_ylabel('Relative intensity',                         labelpad=15)
        elif (model.images[image_nr].imageType.get() == ImageType.OpticalDepth):
            ax0.set_xlabel('Optical depth [.]',      labelpad=11)
            ax2.set_ylabel('Relative optical depth', labelpad=15)

        plt.savefig(f"{im_dir}/image_{f:0>3d}.png", bbox_inches='tight')

        figs.append(fig)

        plt.close()

    # Create a widget for plots
    widget = interact(lambda v: figs[v], v=(0, len(figs)-1))

    return widget


def image_plotly(
        model: Model,
        image_nr: int   =  -1,
        zoom: float     = 1.3,
        npix_x: int     = 300,
        npix_y: int     = 300,
        x_unit: Unit    = units.au,
        v_unit: Unit    = units.km/units.s,
        method: str     = 'nearest',
        width: float    = 620,   # Yields approx square channel map
        height: float   = 540    # Yields approx square channel map
    ):
    """
    Create plots of the channel maps of a synthetic observation (image) with plotly.

    Parameters
    ----------
    model : object
        Magritte model object.
    image_nr : int
        Number of the synthetic observation to plot. (Use -1 to indicate the last one.)
    zoom : float
        Factor with which to zoom in on the middel of the image.
    npix_x : int
        Number of pixels in the image in the horizontal (x) direction.
    npix_y : int
        Number of pixels in the image in the vertical (y) direction.
    x_unit : astropy.units object
        Unit of length for the horixontal (x) axis.
    y_unit : astropy.units object
        Unit of length for the vertical (y) axis.
    method : str
        Method to interpolate the scattered intensity data onto a regular image grid.
    width : float
        Width of the resulting figure.
    height : float
        Height of the resulting figure.

    Returns
    -------
    None
    """
    # Check if there are images
    if (len(model.images) < 1):
        print('No images in model.')
        return

    # Get path of image directory
    im_dir = os.path.dirname(os.path.abspath(model.io.save_location)) + '/images/'

    # If no image directory exists yet
    if not os.path.exists(im_dir):
        # Create image directory
        os.makedirs(im_dir)
        print('Created image directory:', im_dir)

    # Extract data of last image
    imx = model.images[image_nr].imX.get_astropy() #dims: [image.npix]
    imy = model.images[image_nr].imY.get_astropy() #dims: [image.npix]
    imI = model.images[image_nr].I.get_astropy() #dims: [image.npix, image.nfreqs]

    # Extract the number of frequency bins
    nfreqs = model.images[image_nr].nfreqs.get()

    # Set image boundaries
    deltax = (np.max(imx) - np.min(imx))/zoom
    midx = (np.max(imx) + np.min(imx))/2.0
    deltay = (np.max(imy) - np.min(imy))/zoom
    midy = (np.max(imy) + np.min(imy))/2.0

    x_min, x_max = midx - deltax/2.0, midx + deltax/2.0
    y_min, y_max = midy - deltay/2.0, midy + deltay/2.0

    # Create image grid values
    xs: Quantity = np.linspace(x_min, x_max, npix_x)
    ys: Quantity = np.linspace(y_min, y_max, npix_y)

    # Extract the spectral / velocity data
    freqs = model.images[image_nr].freqs.get_astropy() #dims: [image.nfreqs]
    f_ij  = np.mean(freqs)
    velos = (f_ij - freqs) / f_ij * constants.c

    # Interpolate the scattered data to an image (regular grid)
    # Euhm, the units are not preserved here, so I have to add them back in; for optical depths, these are the wrong units, though the units will not be used for anything
    Is = np.zeros((nfreqs)) * units.W / units.m**2 / units.sr / units.Hz
    zs: Quantity = np.zeros((nfreqs, npix_x, npix_y)) * units.W / units.m**2 / units.sr / units.Hz
    for f in range(nfreqs):
        # Nearest neighbor interpolate scattered image data
        zs[f] = griddata(
            (imx, imy),
            imI[:,f],
            (xs[None,:], ys[:,None]),
            method=method,
            fill_value = 0.0 #for non-nearest neighbor interpolation, otherwise the ceil/floor functions will complain
        ) * units.W / units.m**2 / units.sr / units.Hz # This scipy function does not preserve the units
        Is[f] = np.sum(zs[f])
    Is = Is / np.max(Is) # Normalized intensity, thus unitless

    # Put zero-values to the smallest non-zero value
    zs[zs<=0.0] = np.min(zs[zs>0.0])
    # Put nan values to smallest positive value
    zs[np.isnan(zs)] = np.min(zs[zs>0.0])

    # Get the logarithm of the data (matplotlib has a hard time handling logarithmic data.)
    log_zs     = np.log10(zs.value)
    log_zs_min = np.min(log_zs)
    log_zs_max = np.max(log_zs)

    if   (model.images[image_nr].imageType.get() == ImageType.Intensity):
        # Create plotly plot
        fig = make_subplots(
            rows               = 1,
            cols               = 2,
            column_widths      = [0.7, 0.3],
            horizontal_spacing = 0.05,
            subplot_titles     = ['Intensity', ''],
        )
    elif (model.images[image_nr].imageType.get() == ImageType.OpticalDepth):
        # Create plotly plot
        fig = make_subplots(
            rows               = 1,
            cols               = 2,
            column_widths      = [0.7, 0.3],
            horizontal_spacing = 0.05,
            subplot_titles     = ['Optical depth', ''],
        )


    fig.add_vrect(
        row        = 1,
        col        = 1,
        x0         = -1.0e+99,
        x1         = +1.0e+99,
        line_width = 0,
        fillcolor  = "black"
    )

    # Convert to given units
    xs = xs.to(x_unit)
    ys = ys.to(x_unit)

    # Convert to given units
    x_max = np.max(xs)
    x_min = np.min(xs)
    y_max = np.max(ys)
    y_min = np.min(ys)

    # Build up plot
    for f in reversed(range(nfreqs)):
        fig.add_trace(
            go.Heatmap(
                x          = xs       .astype(float),
                y          = ys       .astype(float),
                z          = log_zs[f].astype(float),
                visible    = False,
                hoverinfo  = 'none',
                zmin       = log_zs_min.astype(float),
                zmax       = log_zs_max.astype(float),
                colorscale = cubehelix2_16_plotly,
                showscale  = False
            ),
            row = 1,
            col = 1
        )

        fig.add_trace(
            go.Scatter(
                x          = (velos)        .astype(float),
                y          = (Is/np.max(Is)).astype(float),
                visible    = False,
                hoverinfo  = 'none',
                line_color = '#1f77b4',
                showlegend = False
            ),
            row = 1,
            col = 2
        )

        fig.add_trace(
            go.Scatter(
                x          = np.array([velos[f], velos[f]], dtype=float),
                y          = np.array([-1.0e+10, +1.0e+10], dtype=float),
                visible    = False,
                hoverinfo  = 'none',
                line_color = 'red',
                showlegend = False
            ),
            row = 1,
            col = 2
        )

    # Boxes around plots (more liek mpl)
    fig.update_xaxes(
        showline=True, linewidth=2, linecolor='rgba(1,1,1,1)', mirror=True, ticks='outside'
    )
    fig.update_yaxes(
        showline=True, linewidth=2, linecolor='rgba(1,1,1,1)', mirror=True, ticks='outside'
    )

    # Make figure limits a bit wider than the plotted frequency range
    v_min = float(min(velos)) * 1.05
    v_max = float(max(velos)) * 1.05

    # Black background for channel map
    fig.add_vrect(
        row        = 1,
        col        = 1,
        x0         = 1000.0*x_min,  # large enough so you don't see edges
        x1         = 1000.0*x_max,  # large enough so you don't see edges
        line_width = 0,
        fillcolor  = "black",
        layer="below"
    )

    # Plot axes
    fig.update_xaxes(
        row        = 1,
        col        = 1,
        title_text = f'image x [{x_unit}]',
        range    = [x_min, x_max],
        showgrid = False,
        zeroline = False
    )
    fig.update_yaxes(
        row         = 1,
        col         = 1,
        title_text  = f'image y [{x_unit}]',
        scaleanchor = "x",
        scaleratio  = 1,
        showgrid    = False,
        zeroline    = False
    )
    fig.update_xaxes(
        row        = 1,
        col        = 2,
        title_text = f'velocity [{v_unit}]',
        range      = [v_min, v_max]
    )

    if   (model.images[image_nr].imageType.get() == ImageType.Intensity):
        fig.update_yaxes(
            row        = 1,
            col        = 2,
            title_text = "Relative intensity",
            side       = 'right',
            range      = [-0.05, +1.05]
        )
    elif (model.images[image_nr].imageType.get() == ImageType.OpticalDepth):
        fig.update_yaxes(
            row        = 1,
            col        = 2,
            title_text = "Relative opacity",
            side       = 'right',
            range      = [-0.05, +1.05]
        )

    # Subplot titles are annotations
    fig.update_annotations(
        font_size = 16,
        borderpad = 7
    )

    fig.update_layout(
        width        = width,
        height       = height,
        plot_bgcolor = 'rgba(0,0,0,0)',
        dragmode     = 'pan',
        font         = dict(family="Calibri", size=14, color='black')
    )

    # Make 3 middle traces visible
    fig.data[3*nfreqs//2  ].visible = True
    fig.data[3*nfreqs//2+1].visible = True
    fig.data[3*nfreqs//2+2].visible = True

    # Create and add slider
    steps = []
    for f in range(nfreqs):
        step = dict(
            method = "restyle",
            args = [
                {"visible": [False] * len(fig.data)},
                {"title": "Channel map: " + str(f)}
            ],
            label = ''
        )
        # Toggle f'th trace to "visible"
        step["args"][0]["visible"][3*f:3*f+3] = [True, True, True]
        steps.append(step)

    sliders = [
        dict(
            active     = nfreqs//2,
            pad        = {"t": 75},
            steps      = steps,
            tickcolor  = 'white',
            transition = {'duration': 0}
        )
    ]

    fig.update_layout(
        sliders=sliders
    )

    # Config for modebar buttons
    config = {
        "modeBarButtonsToRemove": modeBarButtonsToRemove,
        "scrollZoom": True
    }

    # Save figure as html file
    fig.write_html(f"{im_dir}/image.html", config=config)

    return fig.show(config=config)


def plot_velocity_1D(model: Model, xscale: str='log', yscale: str='linear'):
    """
    Plot the velocity profile of the model (in 1D, radially).

    Parameters
    ----------
    model : object
        Magritte model object.
    xscale : str
        Scale of the xaxis ("linear", "log", "symlog", "logit", ...)
    yscale : str
        Scale of the yaxis ("linear", "log", "symlog", "logit", ...)

    Returns
    -------
    None
    """
    rs = np.linalg.norm(model.geometry.points.position.get_astropy(), axis=1)
    vs = np.linalg.norm(model.geometry.points.velocity.get_astropy(), axis=1)

    fig = plt.figure(dpi=300)
    plt.plot  (rs, vs)
    plt.xlabel('radius [m]',     labelpad=10)
    plt.ylabel('velocity [m/s]', labelpad=10)
    plt.xscale(xscale)
    plt.yscale(yscale)
    plt.show  ()


def plot_temperature_1D(model: Model, xscale: str='log', yscale: str ='linear'):
    """
    Plot the temperature profile of the model (in 1D, radially).

    Parameters
    ----------
    model : object
        Magritte model object.
    xscale : str
        Scale of the xaxis ("linear", "log", "symlog", "logit", ...)
    yscale : str
        Scale of the yaxis ("linear", "log", "symlog", "logit", ...)

    Returns
    -------
    None
    """
    rs   = np.linalg.norm(model.geometry.points.position.get_astropy(), axis=1)
    temp = model.thermodynamics.temperature.gas.get_astropy()

    fig = plt.figure(dpi=300)
    plt.plot  (rs, temp)
    plt.xlabel('radius [m]',      labelpad=10)
    plt.ylabel('temperature [K]', labelpad=10)
    plt.xscale(xscale)
    plt.yscale(yscale)
    plt.show  ()


def plot_turbulence_1D(model: Model, xscale: str = 'log', yscale: str = 'linear'):
    """
    Plot the (micro) turbulence profile of the model (in 1D, radially).

    Parameters
    ----------
    model : object
        Magritte model object.
    xscale : str
        Scale of the xaxis ("linear", "log", "symlog", "logit", ...)
    yscale : str
        Scale of the yaxis ("linear", "log", "symlog", "logit", ...)

    Returns
    -------
    None
    """
    rs    = np.linalg.norm(model.geometry.points.position.get_astropy(), axis=1)
    vturb = model.thermodynamics.turbulence.vturb.get_astropy()

    fig = plt.figure(dpi=300)
    plt.plot  (rs, vturb)
    plt.xlabel('radius [m]',       labelpad=10)
    plt.ylabel('turbulence [m/s]', labelpad=10)
    plt.xscale(xscale)
    plt.yscale(yscale)
    plt.show  ()


def plot_number_densities_1D(model: Model, xscale: str = 'log', yscale: str = 'log'):
    """
    Plot the number densities of all species in the model (in 1D, radially).

    Parameters
    ----------
    model : object
        Magritte model object.
    xscale : str
        Scale of the xaxis ("linear", "log", "symlog", "logit", ...)
    yscale : str
        Scale of the yaxis ("linear", "log", "symlog", "logit", ...)

    Returns
    -------
    None
    """
    rs   = np.linalg.norm(model.geometry.points.position.get_astropy(), axis=1)
    abns = model.chemistry.species.abundance.get_astropy()
    syms = model.chemistry.species.symbol.get()

    for s in range(model.parameters.nspecs.get()):
        fig = plt.figure(dpi=300)
        plt.plot  (rs, abns[:,s])
        plt.xlabel('radius [m]',                               labelpad=10)
        plt.ylabel(f'{syms[s]} number density [m$^{{{-3}}}$]', labelpad=10)
        plt.xscale(xscale)
        plt.yscale(yscale)
        plt.show  ()


def plot_populations_1D(model: Model, lev_max: int=7, xscale: str='log', yscale: str='log'):
    """
    Plot the relative populations in the model (in 1D, radially).

    Parameters
    ----------
    model : object
        Magritte model object.
    lev_max : int
        Number of levels to plot.
    xscale : str
        Scale of the xaxis ("linear", "log", "symlog", "logit", ...)
    yscale : str
        Scale of the yaxis ("linear", "log", "symlog", "logit", ...)

    Returns
    -------
    None
    """
    rs: Quantity = np.linalg.norm(model.geometry.points.position.get_astropy(), axis=1)
    npoints = model.parameters.npoints.get()

    for lspec in model.sources.lines.lineProducingSpecies:
        nlev     = lspec.linedata.nlev.get()
        pops     = np.array(lspec.population).reshape((npoints,nlev))
        pops_tot = np.array(lspec.population_tot)

        plt.figure(dpi=300)

        for i in range(min([lev_max, nlev])):
            plt.plot  (rs.to(units.m), pops[:,i]/pops_tot, label=f'i={i}')
            plt.ylabel('fractional level populations [.]', labelpad=10)
            plt.xlabel('radius [m]',                       labelpad=10)
            plt.xscale(xscale)
            plt.yscale(yscale)
            plt.legend()
        plt.show()

def save_fits(
        model: Model,
        filename: Optional[str]   = None,
        image_nr: int   =  -1,
        zoom: float     = 1.3,
        npix_x: int     = 256,
        npix_y: int     = 256,
        method: str     = 'nearest',
        dpc: float      = 1.0,
        coord: Optional[str] = None,
        f_rest: Quantity = 0.0,
        square: bool    = False
    ) -> None:
    """
    Save channel maps of synthetic observation (image) as a fits file.

    Parameters
    ----------
    model : object
        Magritte model object.
    image_nr : int
        Number of the synthetic observation to plot. (Use -1 to indicate the last one.)
    zoom : float
        Factor with which to zoom in on the middel of the image.
    npix_x : int
        Number of pixels in the image in the horizontal (x) direction.
    npix_y : int
        Number of pixels in the image in the vertical (y) direction.
    method : str
        Method to interpolate the scattered intensity data onto a regular image grid.
    dpc : float
        Distance of source in parsec.
    coord : str
        Image centre coordinates.
    f_rest : Quantity
        Rest frequency of the transition.
    square : bool
        True if square pixels are required.

    Returns
    -------
    None
    """
    # Check if there are images
    if (len(model.images) < 1):
        print('No images in model.')
        return

    # # Check if 3D or a projection surface was used
    # if (model.parameters.dimension() != 3 and model.images[image_nr].imagePointPosition == ImagePointPosition.AllModelPoints):
    #     raise ValueError('save_fits only works for 3D models or models imaged using the new imager. Please use save_fits_1D for 1D models.')

    if not filename:
        # Get path of image directory
        im_dir = os.path.dirname(os.path.abspath(model.io.save_location)) + '/images/'
        # If no image directory exists yet
        if not os.path.exists(im_dir):
            # Create image directory
            os.makedirs(im_dir)
            print('Created image directory:', im_dir)
        # Define filename
        filename = f"{im_dir}image.fits"

    # Remove fits file if it already exists
    if os.path.isfile(filename):
        os.remove(filename)

    # Extract image data
    imx = model.images[image_nr].imX.get_astropy()
    imy = model.images[image_nr].imY.get_astropy() #units: [m]
    imI = model.images[image_nr].I.get_astropy() #units: [W/m^2/sr/Hz]

    # Extract the number of frequency bins
    nfreqs = model.images[image_nr].nfreqs.get()

    # Set image boundaries
    deltax = (np.max(imx) - np.min(imx))/zoom
    midx = (np.max(imx) + np.min(imx))/2.0
    deltay = (np.max(imy) - np.min(imy))/zoom
    midy = (np.max(imy) + np.min(imy))/2.0

    x_min, x_max = midx - deltax/2.0, midx + deltax/2.0
    y_min, y_max = midy - deltay/2.0, midy + deltay/2.0

    # Rescale if square pixels are required
    if square:
        pix_size_x = (x_max - x_min) / npix_x
        pix_size_y = (y_max - y_min) / npix_y

        if   pix_size_x > pix_size_y:
            y_max *= pix_size_x / pix_size_y
            y_min *= pix_size_x / pix_size_y

        elif pix_size_x < pix_size_y:
            x_max *= pix_size_y / pix_size_x
            x_min *= pix_size_y / pix_size_x

    # Create image grid values
    xs: Quantity = np.linspace(x_min, x_max, npix_x)
    ys: Quantity = np.linspace(y_min, y_max, npix_y)

    # Extract the spectral / velocity data
    freqs = model.images[image_nr].freqs.get_astropy()
    f_cen: Quantity = np.mean(freqs)

    # If no rest frequency is given,
    # default to cetral frequency in the image.
    if f_rest == 0.0:
        f_rest = f_cen

    velos: Quantity = (freqs - f_rest) / f_rest * constants.c
    v_cen: Quantity = (f_cen - f_rest) / f_rest * constants.c

    dpix_x: Quantity = np.mean(np.diff(xs))
    dpix_y: Quantity = np.mean(np.diff(ys))
    dvelos: Quantity = np.diff(velos)

    if (np.abs(rtutils.relative_error(np.max(dvelos), np.min(dvelos))) > 1.0e-9):
        print('WARNING: No regularly spaced frequency bins!')
        dvelo: Optional[float] = None
    else:
        dvelo = np.mean(dvelos).to(units.m/units.s).value#type: ignore

    # Interpolate the scattered data to an image (regular grid)
    zs: Quantity = np.zeros((nfreqs, npix_x, npix_y))
    for f in range(nfreqs):
        # Nearest neighbor interpolate scattered image data
        zs[f] = griddata((imx, imy), imI[:,f], (xs[None,:], ys[:,None]), method=method) * \
            units.W / units.m**2 / units.sr / units.Hz # This scipy function does not preserve the units

    # Convert intensity from J/s/m/m/ster to Jy/pixel
    zs = zs * dpix_x * dpix_y / (dpc * units.parsec)**2 / 1.0e-26

    if coord is None:
        target_ra  = 0.0
        target_dec = 0.0
    else:
        # Decode the image center cooridnates
        # Check first whether the format is OK
        # (From RADMC3D. Thanks C. Dullemond!)
        dum = coord

        ra = []
        delim = ['h', 'm', 's']
        for i in delim:
            ind = dum.find(i)
            if ind <= 0:
                msg = 'coord keyword has a wrong format. The format should be coord="0h10m05s -10d05m30s"'
                raise ValueError(msg)
            ra.append(float(dum[:ind]))
            dum = dum[ind + 1:]

        dec = []
        delim = ['d', 'm', 's']
        for i in delim:
            ind = dum.find(i)
            if ind <= 0:
                msg = 'coord keyword has a wrong format. The format should be coord="0h10m05s -10d05m30s"'
                raise ValueError(msg)
            dec.append(float(dum[:ind]))
            dum = dum[ind + 1:]

        target_ra = (ra[0] + ra[1] / 60. + ra[2] / 3600.) * 15.
        if dec[0] >= 0:
            target_dec = (dec[0] + dec[1] / 60. + dec[2] / 3600.)
        else:
            target_dec = (dec[0] - dec[1] / 60. - dec[2] / 3600.)

    # Convert pixel sizes to degrees
    deg_dpix_x = dpix_x / units.au / dpc / 3600.0
    deg_dpix_y = dpix_y / units.au / dpc / 3600.0

    # Construct the fits header
    hdr = fits.Header()
    hdr['SIMPLE']   = 'T'              # (T=true) indeed a simple fits file
    hdr['BITPIX']   = -64              # number of bits per word in the data (64 bit IEEE floating point format)
    hdr['NAXIS']    = 3                # dimensions (or number of axis) of the file
    hdr['NAXIS1']   = npix_x           # number of pixels along x axis (hor)
    hdr['NAXIS2']   = npix_y           # number of pixels along y-axis (ver)
    hdr['NAXIS3']   = nfreqs           # number of pixels along velocity-axis

    hdr['EXTEND']   = 'T'              # Extendible fits file (T=true for safety, though not required)
    hdr['CROTA1']   = 0.0              # Rotation of axis 1
    hdr['CROTA2']   = 0.0              # Rotation of axis 2.
    hdr['EPOCH']    = 2000.0           # Equinox of celestial coordinate system (deprecated)
    hdr['EQUINOX']  = 2000.0           # Equinox of celestial coordinate system (new)
    hdr['SPECSYS']  = 'LSRK'           # Not sure...
    hdr['RESTFREQ'] = f_rest.to(units.Hz).value # rest frequency of the transition [Hz]
    hdr['VELREF']   = 257              # Not sure...

    hdr['CTYPE1']   = 'RA---SIN'
    hdr['CDELT1']   = -deg_dpix_x.value # pixel size in degrees along x-axis
    hdr['CRPIX1']   = (npix_x-1)/2.0   # pixel index of centre x=0
    hdr['CRVAL1']   = target_ra        # image centre coordinate (x/ra)
    hdr['CUNIT1']   = 'DEG'            # x-axis unit

    hdr['CTYPE2']   = 'DEC--SIN'
    hdr['CDELT2']   = deg_dpix_y.value # pixel size in degrees along y-axis
    hdr['CRPIX2']   = (npix_y-1)/2.0   # pixel index of centre y=0
    hdr['CRVAL2']   = target_dec       # image centre coordinate (y/dec)
    hdr['CUNIT2']   = 'DEG'            # y-axis unit

    hdr['CTYPE3']   = 'VELO-LSR'
    hdr['CDELT3']   = dvelo            # pixel size in m/s along velocity-axis
    hdr['CRPIX3']   = (nfreqs-1)/2.0   # pixel index of centre
    hdr['CRVAL3']   = v_cen.to(units.m/units.s).value # centre value
    hdr['CUNIT3']   = 'M/S'            # velocity-axis unit

    hdr['BTYPE']    = 'INTENSITY'
    hdr['BSCALE']   = 1.0
    hdr['BZERO']    = 0.0
    hdr['BUNIT']    = 'JY/PIXEL'

    fits.writeto(filename, data=zs.value, header=hdr)

    print('Written file to:', filename)

    return
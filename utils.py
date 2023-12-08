import csv

import numpy as np
from astropy.io import fits
from scipy.interpolate import interp1d

from extern.interpolate import SmoothSpline

waveStart = 3000
waveEnd = 9999
waveNum = 7000
waveGrid = np.linspace(waveStart, waveEnd, num=waveNum)
normWavelength = 4999

def interpOntoGrid(waveGrid ,wavelength ,flux):
    """
    Description:
        一种将谱通量和方差置于同一位置的方法
          波长网格作为模板（5 km/s 等距箱） (5 km/s equally spaced bins)
    """
    # 将通量和方差插值到波长网格上
    interpFlux = np.interp(waveGrid, wavelength, flux, right=np.nan, left=np.nan)

    wavelength = waveGrid
    flux = interpFlux

    return wavelength ,flux

def read_obs_line_index(x):
    """
    计算一条光谱的线指数
    :param flux: 光谱的流量向量
    :param wave: 光谱的波长向量
    :return: 线指数, np.array类型
    """
    # 理论模板
    # elements = [(3835, 3836),
    #             (3889, 3890),
    #             (3933, 3934),
    #             (3969, 3970),
    #             (4101, 4102),
    #             (4856, 4857),
    #             (4858, 4859),
    #             (5183, 5184),
    #             (5240, 5241),
    #             (5276, 5277),
    #             (5316, 5317),
    #             (5321, 5322),
    #             (5323, 5324),
    #             (5336, 5337),
    #             (5368, 5369),
    #             (5589, 5590),
    #             (5657, 5658),
    #             (5891, 5892),
    #             (8467, 8468), ]
    # 观测模板
    # elements = [(3832, 3833),
    #             (3838, 3839),
    #             (3839, 3840),
    #             (3870, 3871),
    #             (3871, 3872),
    #             (3932, 3933),
    #             (3936, 3937),
    #             (3970, 3971),
    #             (4099, 4100),
    #             (4179, 4180),
    #             (4215, 4216),
    #             (4566, 4567),
    #             (5183, 5184),
    #             (5185, 5186),
    #             (5252, 5253),
    #             (5783, 5784),
    #             (6566, 6567),
    #             (8544, 8545), ]

    baseline = 3900
    line_index = x[:, :,[3932-baseline, 3933-baseline, 3936-baseline, 3937-baseline, 3970-baseline, 3971-baseline, 4099-baseline, 4100-baseline,
                         4179-baseline, 4180-baseline, 4215-baseline, 4216-baseline, 4566-baseline, 4567-baseline, 5183-baseline, 5184-baseline, 5185-baseline, 5186-baseline, 5252-baseline, 5253-baseline, 5783-baseline, 5784-baseline, 6566-baseline, 6567-baseline, 8544-baseline, 8545-baseline]]


    return line_index


def read_syn_line_index(x):
    """
    计算一条光谱的线指数
    :param flux: 光谱的流量向量
    :param wave: 光谱的波长向量
    :return: 线指数, np.array类型
    """
    # 理论模板
    # elements = [(3835, 3836),
    #             (3889, 3890),
    #             (3933, 3934),
    #             (3969, 3970),
    #             (4101, 4102),
    #             (4856, 4857),
    #             (4858, 4859),
    #             (5183, 5184),
    #             (5240, 5241),
    #             (5276, 5277),
    #             (5316, 5317),
    #             (5321, 5322),
    #             (5323, 5324),
    #             (5336, 5337),
    #             (5368, 5369),
    #             (5589, 5590),
    #             (5657, 5658),
    #             (5891, 5892),
    #             (8467, 8468), ]
    # 观测模板
    # elements = [(3832, 3833),
    #             (3838, 3839),
    #             (3839, 3840),
    #             (3870, 3871),
    #             (3871, 3872),
    #             (3932, 3933),
    #             (3936, 3937),
    #             (3970, 3971),
    #             (4099, 4100),
    #             (4179, 4180),
    #             (4215, 4216),
    #             (4566, 4567),
    #             (5183, 5184),
    #             (5185, 5186),
    #             (5252, 5253),
    #             (5783, 5784),
    #             (6566, 6567),
    #             (8544, 8545), ]

    baseline = 3900
    line_index = x[:, :,
                 [3932 - baseline, 3933 - baseline, 3936 - baseline, 3937 - baseline, 3970 - baseline, 3971 - baseline,
                  4099 - baseline, 4100 - baseline,
                  4179 - baseline, 4180 - baseline, 4215 - baseline, 4216 - baseline, 4566 - baseline, 4567 - baseline,
                  5183 - baseline, 5184 - baseline, 5185 - baseline, 5186 - baseline, 5252 - baseline, 5253 - baseline,
                  5783 - baseline, 5784 - baseline, 6566 - baseline, 6567 - baseline, 8544 - baseline, 8545 - baseline]]

    return line_index

baseline = 3900
syn_line_index = [3932 - baseline, 3933 - baseline, 3936 - baseline, 3937 - baseline, 3970 - baseline, 3971 - baseline,
                  4099 - baseline, 4100 - baseline,
                  4179 - baseline, 4180 - baseline, 4215 - baseline, 4216 - baseline, 4566 - baseline, 4567 - baseline,
                  5183 - baseline, 5184 - baseline, 5185 - baseline, 5186 - baseline, 5252 - baseline, 5253 - baseline,
                  5783 - baseline, 5784 - baseline, 6566 - baseline, 6567 - baseline, 8544 - baseline, 8545 - baseline]

def normalize_spectrum_null(wave):
    return np.ones_like(wave) * np.nan, np.ones_like(wave) * np.nan



def normalize_spectrum_spline(wave, flux, p=1E-6, q=0.5, lu=(-1, 3), binwidth=30, niter=2):
    """ A double smooth normalization of a spectrum

    Converted from Chao Liu's normSpectrum.m
    Updated by Bo Zhang

    Parameters
    ----------
    wave: ndarray (n_pix, )
        wavelegnth array
    flux: ndarray (n_pix, )
        flux array
    p: float
        smoothing parameter between 0 and 1:
        0 -> LS-straight line
        1 -> cubic spline interpolant
    q: float in range of [0, 1]
        percentile, between 0 and 1
    lu: float tuple
        the lower & upper exclusion limits
    binwidth: float
        width of each bin
    niter: int
        number of iterations
    Returns
    -------
    flux_norm: ndarray
        normalized flux
    flux_cont: ndarray
        continuum flux

    Example
    -------


    """
    # wave = wave[0:1800]
    # flux = flux[0:1800]
    if np.sum(np.logical_and(np.isfinite(flux), flux > 0)) <= 10:
        return normalize_spectrum_null(wave)

    _wave = np.copy(wave)
    _flux = np.copy(flux)
    ind_finite = np.isfinite(flux)
    wave = _wave[ind_finite]
    flux = _flux[ind_finite]
    _flux_norm = np.copy(_flux)
    _flux_cont = np.copy(_flux)

    # default config is even weight
    var = np.ones_like(flux)

    # check q region
    # assert 0. <= q <= 1.

    nbins = np.int(np.ceil((wave[-1] - wave[0]) / binwidth) + 1)
    bincenters = np.linspace(wave[0], wave[-1], nbins)

    # iteratively smoothing
    ind_good = np.isfinite(flux)
    for _ in range(niter):

        flux_smoothed1 = SmoothSpline(wave[ind_good], flux[ind_good], p=p, var=var[ind_good])(wave)
        # residual
        res = flux - flux_smoothed1

        # determine sigma
        stdres = np.zeros(nbins)
        for ibin in range(nbins):
            ind_this_bin = ind_good & (np.abs(wave - bincenters[ibin]) <= binwidth)
            if 0 <= q <= 0:
                stdres[ibin] = np.std(
                    res[ind_this_bin] - np.percentile(res[ind_this_bin], 100 * q))
            else:
                stdres[ibin] = np.std(res[ind_this_bin])
        stdres_interp = interp1d(bincenters, stdres, kind="linear")(wave)
        if 0 <= q <= 1:
            res1 = (res - np.percentile(res, 100 * q)) / stdres_interp
        else:
            res1 = res / stdres_interp
        ind_good = ind_good & (res1 > lu[0]) & (res1 < lu[1])

        # assert there is continuum pixels
        try:
            assert np.sum(ind_good) > 0
        except AssertionError:
            Warning("@normalize_spectrum_iter: unable to find continuum!")
            ind_good = np.ones(wave.shape, dtype=np.bool)

    # final smoothing
    flux_smoothed2 = SmoothSpline(
        wave[ind_good], flux[ind_good], p=p, var=var[ind_good])(wave)
    # normalized flux
    flux_norm = flux / flux_smoothed2

    _flux_norm[ind_finite] = flux_norm
    _flux_cont[ind_finite] = flux_smoothed2

    _flux = _flux_norm

    # norm = np.linalg.norm(_flux)
    # _flux = _flux / norm


    return _flux

def rv(fname):

    filenameA = r'E:\Aa学习\研二上\实验\数据处理\dr8csv\final_data_dr8_v2.csv'
    osid_obs = []
    rv_obs = []
    with open(filenameA, mode='r', encoding='utf-8-sig') as file:
        reader = csv.reader(file)
        f = next(reader)
        for row in reader:
            osid_obs.append(row[0])
            rv_obs.append(float(row[4]))
    fitsFile = fits.open(fname)

    obsid = fitsFile[0].header['OBSID']
    for index_A in range(len(osid_obs)):
        if int(osid_obs[index_A]) == int(obsid):
            shift = rv_obs[index_A]
            break
        if index_A == len(osid_obs) - 1:
            shift = 'none'
    return shift


def shiftToRest(shift,_wavelength):
    """
    Shift the observed wavelengths to the rest frame in the same grid as the templates using the radial velocity
    使用径向速度将观察到的波长移动到与模板相同的网格中的静止帧
    Input:
    Calculated radial velocity float [km/s] 计算的径向速度浮动
    """
    # 检查是否找到了 RV，如果没有，请不要移动频谱
    if np.isnan(shift):
        shift = 0.0

    _wavelength = _wavelength / (shift / (299792.458) + 1)

    return _wavelength



import pickle

import numpy as np
import os


def read_in_neural_network():
    '''
    read in the weights and biases parameterizing a particular neural network.
    You can read in existing networks from the neural_nets/ directory, or you
    can train your own networks and edit this function to read them in.
    '''

    path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'neural_nets/NN_normalized_spectra.npz')
    tmp = np.load(path)
    w_array_0 = tmp["w_array_0"]
    w_array_1 = tmp["w_array_1"]
    w_array_2 = tmp["w_array_2"]
    b_array_0 = tmp["b_array_0"]
    b_array_1 = tmp["b_array_1"]
    b_array_2 = tmp["b_array_2"]
    x_min = tmp["x_min"]
    x_max = tmp["x_max"]
    NN_coeffs = (w_array_0, w_array_1, w_array_2, b_array_0, b_array_1, b_array_2, x_min, x_max)
    tmp.close()
    return NN_coeffs

def read_in_neural_network_syn():
    '''
    read in the weights and biases parameterizing a particular neural network.
    You can read in existing networks from the neural_nets/ directory, or you
    can train your own networks and edit this function to read them in.
    '''

    tmp = np.load(r'E:\Aa学习\研二下\实验\The_Payne-master\The_Payne-master\loss\NN_normalized_spectra.npz')
    w_array_0 = tmp["w_array_0"]
    w_array_1 = tmp["w_array_1"]
    w_array_2 = tmp["w_array_2"]
    b_array_0 = tmp["b_array_0"]
    b_array_1 = tmp["b_array_1"]
    b_array_2 = tmp["b_array_2"]
    x_min = tmp["x_min"]
    x_max = tmp["x_max"]
    NN_coeffs = (w_array_0, w_array_1, w_array_2, b_array_0, b_array_1, b_array_2, x_min, x_max)
    tmp.close()
    return NN_coeffs


def load_wavelength_array():
    '''
    read in the default wavelength grid onto which we interpolate all spectra
    '''
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'other_data/apogee_wavelength.npz')
    tmp = np.load(path)
    wavelength = tmp['wavelength']
    tmp.close()
    return wavelength


def load_apogee_mask():
    '''
    read in the pixel mask with which we will omit bad pixels during spectral fitting
    The mask is made by comparing the tuned Kurucz models to the observed spectra from Arcturus
    and the Sun from APOGEE. We mask out pixels that show more than 2% of deviations.
    '''
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'other_data/apogee_mask.npz')
    tmp = np.load(path)
    mask = tmp['apogee_mask']
    tmp.close()
    return mask


def load_cannon_contpixels():
    '''
    read in the default list of APOGEE pixels for continuum fitting.
    '''
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'other_data/cannon_cont_pixels_apogee.npz')
    tmp = np.load(path)
    pixels_cannon = tmp['pixels_cannon']
    tmp.close()
    return pixels_cannon


def load_training_data():
    '''
    read in the default Kurucz training spectra for APOGEE

    Here we only consider a small number (<1000) of training spectra.
    In practice, more training spectra will be better. The default
    neural network was trained using 12000 training spectra.
    '''
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'other_data/kurucz_training_spectra.npz')
    tmp = np.load(path)
    training_labels = (tmp["labels"].T)[:800,:]
    training_spectra = tmp["spectra"][:800,:]
    validation_labels = (tmp["labels"].T)[800:,:]
    validation_spectra = tmp["spectra"][800:,:]
    tmp.close()
    return training_labels, training_spectra, validation_labels, validation_spectra

def load_training_data_syn():
    '''
    read in the default Kurucz training spectra for APOGEE

    Here we only consider a small number (<1000) of training spectra.
    In practice, more training spectra will be better. The default
    neural network was trained using 12000 training spectra.
    '''

    tmp = np.load(r'The_Payne/npz/A_data.npz')
    training_labels = tmp["training_labels"]
    training_spectra = tmp["training_spectra"]
    validation_labels = tmp["validation_labels"]
    validation_spectra = tmp["validation_spectra"]
    tmp.close()
    return training_labels, training_spectra, validation_labels, validation_spectra



def doppler_shift(wavelength, flux, dv):
    '''
    dv is in km/s
    We use the convention where a positive dv means the object is moving away.

    This linear interpolation is actually not that accurate, but is fine if you
    only care about accuracy to the level of a few tenths of a km/s. If you care
    about better accuracy, you can do better with spline interpolation.
    '''
    c = 2.99792458e5 # km/s
    doppler_factor = np.sqrt((1 - dv/c)/(1 + dv/c))
    new_wavelength = wavelength * doppler_factor
    new_flux = np.interp(new_wavelength, wavelength, flux)
    return new_flux


def get_apogee_continuum(spec, spec_err = None, cont_pixels = None):
    '''
    continuum normalize spectrum.
    pixels with large uncertainty are weighted less in the fit.
    '''
    if cont_pixels is None:
        cont_pixels = load_cannon_contpixels()
    cont = np.empty_like(spec)

    wavelength = load_wavelength_array()

    deg = 4

    # if we haven't given any uncertainties, just assume they're the same everywhere.
    if spec_err is None:
        spec_err = np.zeros(spec.shape[0]) + 0.0001

    # Rescale wavelengths
    bluewav = 2*np.arange(2920)/2919 - 1
    greenwav = 2*np.arange(2400)/2399 - 1
    redwav = 2*np.arange(1894)/1893 - 1

    blue_pixels= cont_pixels[:2920]
    green_pixels= cont_pixels[2920:5320]
    red_pixels= cont_pixels[5320:]

    # blue
    cont[:2920]= _fit_cannonpixels(bluewav, spec[:2920], spec_err[:2920],
                        deg, blue_pixels)
    # green
    cont[2920:5320]= _fit_cannonpixels(greenwav, spec[2920:5320], spec_err[2920:5320],
                        deg, green_pixels)
    # red
    cont[5320:]= _fit_cannonpixels(redwav, spec[5320:], spec_err[5320:], deg, red_pixels)
    return cont


def _fit_cannonpixels(wav, spec, specerr, deg, cont_pixels):
    '''
    Fit the continuum to a set of continuum pixels
    helper function for get_apogee_continuum()
    '''
    chpoly = np.polynomial.Chebyshev.fit(wav[cont_pixels], spec[cont_pixels],
                deg, w=1./specerr[cont_pixels])
    return chpoly(wav)

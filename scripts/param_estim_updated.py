### Importing packages

import numpy as np
import matplotlib.pyplot as plt
import bilby
import h5py
from bilby.core.prior import Uniform, PowerLaw
from bilby.gw.conversion import convert_to_lal_binary_black_hole_parameters, generate_all_bbh_parameters
bilby.core.utils.log.setup_logger(log_level='WARNING')
from gwpy.timeseries import TimeSeries
import os
from pesummary.io import read
bilby.core.utils.log.setup_logger(log_level='WARNING')


#### Importing files 

EVENT = 'B'

gps_time = 1265754805.00

sampling_frequency = 1024
reference_frequency = 10
waveform_min_frequency = 10
detector_min_frequency = 20
detector_max_frequency = 448

duration = 8

file_h1 = '/home/achakraborty/project_lensing_beyong_GWTC/strains/H-H1_GWOSC_O3b_4KHZ_R1-1265754112-4096.hdf5'
file_l1 = '/home/achakraborty/project_lensing_beyong_GWTC/strains/L-L1_GWOSC_O3b_4KHZ_R1-1265754112-4096.hdf5'
post_file = '/home/achakraborty/project_lensing_beyong_GWTC/posteriors/GW200214_223307-PYCBC-POSTERIOR-IMRPhenomXPHM.hdf'

asd_file_h1 = 'H1_asd.txt' 
asd_file_l1= 'L1_asd.txt'

asd_data_h = np.loadtxt(asd_file_h1)
asd_data_l = np.loadtxt(asd_file_l1)

freq_h = asd_data_h[:,0]
freq_l = asd_data_l[:,0]
H1_psd = asd_data_h[:,1] **2
L1_psd = asd_data_l[:,1] **2

#### Analyzing the inputs

with h5py.File(post_file, 'r') as f:
    samples = f['samples']
    maxL_idx = np.argmax(samples['loglikelihood'][:])

    z = samples['redshift'][maxL_idx]
    maxL_m1 = samples['srcmass1'][maxL_idx] * (1 + z)
    maxL_m2 = samples['srcmass2'][maxL_idx] * (1 + z)
    maxL_distance = samples['distance'][maxL_idx]
    maxL_phase = samples['coa_phase'][maxL_idx]
    maxL_delta_tc = samples['delta_tc'][maxL_idx]
    maxL_a1 = samples['spin1_a'][maxL_idx]
    maxL_a2 = samples['spin2_a'][maxL_idx]
    maxL_tilt1 = samples['spin1_polar'][maxL_idx]
    maxL_tilt2 = samples['spin2_polar'][maxL_idx]
    maxL_theta_jn = samples['inclination'][maxL_idx]
    maxL_ra = samples['ra'][maxL_idx]
    maxL_dec = samples['dec'][maxL_idx]

print(f"-> Found MaxL Detector Masses: M1={maxL_m1:.2f}, M2={maxL_m2:.2f}")

#### Modeling the Waveform

def lal_bbh_wo_lensing(frequency_array, mass_1, mass_2, luminosity_distance, a_1, tilt_1,
                       phi_12, a_2, tilt_2, phi_jl, theta_jn, phase, a, b, k, phi_0,
                       b_prime, k_prime, phi_0_prime, f_0, psi, **kwargs):

    gr_waveform = bilby.gw.source.lal_binary_black_hole(
        frequency_array=frequency_array, mass_1=mass_1, mass_2=mass_2,
        luminosity_distance=luminosity_distance, a_1=a_1, tilt_1=tilt_1, phi_12=phi_12,
        a_2=a_2, tilt_2=tilt_2, phi_jl=phi_jl, theta_jn=theta_jn, phase=phase, psi=psi, **kwargs)

    if gr_waveform is None:
        return None

    amp_mod = a * (1 + b * (np.cos(2 * np.pi * frequency_array / f_0 + phi_0) * np.exp(-frequency_array * k)))
    phase_mod = np.exp(1j * b_prime * np.cos(2 * np.pi * frequency_array / f_0 + phi_0_prime) * np.exp(-frequency_array * k_prime))
    correction = amp_mod * phase_mod

    return dict(plus=gr_waveform['plus'] * correction, cross=gr_waveform['cross'] * correction)


#### Loading data into Bilby

H1 = bilby.gw.detector.get_empty_interferometer("H1")
L1 = bilby.gw.detector.get_empty_interferometer("L1")

h1_ts = TimeSeries.read(file_h1, format='hdf5.gwosc').crop(gps_time-duration/2, gps_time+duration/2).resample(sampling_frequency)
l1_ts = TimeSeries.read(file_l1, format='hdf5.gwosc').crop(gps_time-duration/2, gps_time+duration/2).resample(sampling_frequency)

H1.set_strain_data_from_gwpy_timeseries(h1_ts)
L1.set_strain_data_from_gwpy_timeseries(l1_ts)

H1.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
    frequency_array=freq_h, psd_array=H1_psd)
L1.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
    frequency_array=freq_l, psd_array=L1_psd)

for ifo in [H1, L1]:
    ifo.minimum_frequency = detector_min_frequency
    ifo.maximum_frequency = detector_max_frequency

ifos = [H1, L1]

#### Setting up Prior

priors = bilby.core.prior.PriorDict()
priors['mass_1'] = maxL_m1
priors['mass_2'] = maxL_m2
priors['phase'] = maxL_phase
priors['geocent_time'] = gps_time + maxL_delta_tc # Trigger + offset
priors['a_1'] = maxL_a1
priors['a_2'] = maxL_a2
priors['tilt_1'] = maxL_tilt1
priors['tilt_2'] = maxL_tilt2
priors['phi_12'] = 0.0
priors['phi_jl'] = 0.0
priors['dec'] = maxL_dec
priors['ra'] = maxL_ra
priors['theta_jn'] = maxL_theta_jn
priors['luminosity_distance'] = maxL_distance


priors['psi'] = Uniform(name='psi', minimum=0.0, maximum=np.pi, boundary='periodic', latex_label='$\psi$')
priors['b'] = Uniform(name='b', minimum=0.0, maximum=0.99, latex_label='$b$')
priors['f_0'] = Uniform(name='f_0', minimum=10.0, maximum=detector_max_frequency, latex_label='$f_0$')
priors['phi_0'] = Uniform(name='phi_0', minimum=0.0, maximum=2*np.pi, boundary='periodic', latex_label=r'$\phi_0$')
priors['b_prime'] = Uniform(name='b_prime', minimum=0.0, maximum=0.99, latex_label="$b'$")
priors['phi_0_prime'] = Uniform(name='phi_0_prime', minimum=0.0, maximum=2*np.pi, boundary='periodic', latex_label=r"$\phi'_0$")


priors['a'] = 1.0
priors['k'] = 0.0
priors['k_prime'] = 0.0

#### Setting up the run 

waveform_arguments = dict(waveform_approximant='IMRPhenomXPHM', 
                          minimum_frequency = waveform_min_frequency, 
                          reference_frequency= reference_frequency, 
                          catch_waveform_errors=True)


waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    frequency_domain_source_model=lal_bbh_wo_lensing,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
    waveform_arguments= waveform_arguments
)

likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
    interferometers=ifos, waveform_generator=waveform_generator, priors=priors
)

result = bilby.run_sampler(
    likelihood=likelihood,
    priors=priors,
    conversion_function = bilby.gw.conversion.generate_all_bbh_parameters,
    sampler='dynesty',
    nlive=500,
    dlogz=0.1,
    npool=16,
    outdir=f'/home/achakraborty/project_lensing_beyong_GWTC/Lensing_Result_Event_{EVENT}',
    label=f'GW_Lensing_{EVENT}'
)

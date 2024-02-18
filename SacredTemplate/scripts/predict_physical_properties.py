import pandas as pd
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from keras.models import load_model
import math
import numpy as np
import pickle

def main():
    df_testing= pd.read_csv('..\data\database.csv')

    X_test = df_testing.loc[:, ['wavelength', 'primary_particle_size', 'equi_mobility_dia', 'q_ext', 'q_abs', 'q_sca', 'g']]

    # Normalizaing Min max
    scaling_x = pickle.load(open('..\data\inverse_scaler_x.sav', 'rb'))
    scaling_y = pickle.load(open('..\data\inverse_scaler_y.sav', 'rb'))
    X_test = scaling_x.transform(X_test)

    model = load_model('../data/inverse_best_model.hdf5')
    Y_test = model.predict(X_test)
    Y_test = scaling_y.inverse_transform(Y_test)


    # Computing others
    Y_test = pd.DataFrame(data=Y_test, columns=["fractal_dimension", "fraction_of_coating"])

    wavelength = df_testing['wavelength']
    primary_particle_size = df_testing['primary_particle_size']
    equi_mobility_dia = df_testing['equi_mobility_dia']

    q_abs = df_testing['q_abs']
    q_sca = df_testing['q_sca']
    q_ext = df_testing['q_ext']
    g = df_testing['g']

    fractal_dimension = Y_test['fractal_dimension']
    fraction_of_coating = Y_test['fraction_of_coating']
    number_of_primary_particles = (df_testing['equi_mobility_dia']/(2*0.7943*df_testing['primary_particle_size']))**(1/0.51) #Using mobility diameter formulae
    vol_equi_radius_inner=((3*number_of_primary_particles*4*math.pi*(15)**3)/(4*math.pi*3)) ** (1. / 3) #considering that particle size without coating is always 15
    vol_equi_radius_outer=df_testing['primary_particle_size']*(number_of_primary_particles ** (1. / 3))



    mie_epsilon = np.zeros_like(wavelength) + 2 #Check
    length_scale_factor = 2 * math.pi / wavelength

    m_real_bc = np.empty_like(wavelength)
    for i in range(0, len(wavelength)):
        if wavelength[i] == 467:
            m_real_bc[i] = 1.92
        elif wavelength[i] == 530:
            m_real_bc[i] = 1.96
        elif wavelength[i] == 660:
            m_real_bc[i] = 2
        else:
            m_real_bc[i] = np.nan

    m_im_bc = np.empty_like(wavelength)
    for i in range(0, len(wavelength)):
        if wavelength[i] == 467:
            m_im_bc[i] = 0.67
        elif wavelength[i] == 530:
            m_im_bc[i] = 0.65
        elif wavelength[i] == 660:
            m_im_bc[i] = 0.63
        else:
            m_im_bc[i] = np.nan

    m_real_organics = np.empty_like(wavelength)
    for i in range(0, len(wavelength)):
        if wavelength[i] == 467:
            m_real_organics[i] = 1.59
        elif wavelength[i] == 530:
            m_real_organics[i] = 1.47
        elif wavelength[i] == 660:
            m_real_organics[i] = 1.47
        else:
            m_real_organics[i] = np.nan

    m_im_organics = np.empty_like(wavelength)
    for i in range(0, len(wavelength)):
        if wavelength[i] == 467:
            m_im_organics[i] = 0.11
        elif wavelength[i] == 530:
            m_im_organics[i] = 0.04
        elif wavelength[i] == 660:
            m_im_organics[i] = 0
        else:
            m_im_organics[i] = np.nan

    volume_total = (4 * math.pi * (vol_equi_radius_outer ** 3)) / 3
    volume_bc = (4 * math.pi * (vol_equi_radius_inner ** 3)) / 3
    volume_organics = volume_total - volume_bc

    density_bc = np.zeros_like(wavelength) + 1.5 #Check
    density_organics = np.zeros_like(wavelength) + 1.1 #Check

    mass_bc = volume_bc * density_bc * (1 / 1000000000000000000000)
    mass_organics = volume_organics * density_organics * (1 / 1000000000000000000000)
    mass_total = mass_bc + mass_organics
    mr_total_bc = mass_total / mass_bc
    mr_nonbc_bc = mass_organics / mass_bc


    c_geo = (math.pi) * ((vol_equi_radius_outer) ** 2)
    c_ext = (q_ext * c_geo) / (float(1000000))
    c_abs = q_abs * c_geo / (1000000)
    c_sca = q_sca * c_geo / (1000000)
    ssa = q_sca / q_ext
    mac_total = (c_abs) / (mass_total * 1000000000000)
    mac_bc = c_abs / (mass_bc * (1000000000000))
    mac_organics = c_abs / (mass_organics * (1000000000000))

    final = np.stack((wavelength, fractal_dimension, fraction_of_coating, primary_particle_size,
                      number_of_primary_particles, vol_equi_radius_inner, vol_equi_radius_outer, equi_mobility_dia,
                      mie_epsilon, length_scale_factor, m_real_bc, m_im_bc, m_real_organics, m_im_organics,
                      volume_total, volume_bc, volume_organics, density_bc, density_organics, mass_total, mass_organics,
                      mass_bc, mr_total_bc, mr_nonbc_bc, q_ext, q_abs, q_sca, g, c_geo, c_ext, c_abs, c_sca, ssa,
                      mac_total, mac_bc, mac_organics), axis=1)

    final_dataset = pd.DataFrame(data=final, columns=['wavelength', 'fractal_dimension', 'fraction_of_coating', 'primary_particle_size',
                      'number_of_primary_particles', 'vol_equi_radius_inner', 'vol_equi_radius_outer', 'equi_mobility_dia',
                      'mie_epsilon', 'length_scale_factor', 'm_real_bc', 'm_im_bc', 'm_real_organics', 'm_im_organics',
                      'volume_total', 'volume_bc', 'volume_organics', 'density_bc', 'density_organics', 'mass_total', 'mass_organics',
                      'mass_bc', 'mr_total_bc', 'mr_nonbc_bc', 'q_ext', 'q_abs', 'q_sca', 'g', 'c_geo', 'c_ext', 'c_abs', 'c_sca', 'ssa',
                      'mac_total', 'mac_bc', 'mac_organics'])
    final_dataset.to_csv('..\data\predicted_inverse_dataset.csv', index=False)

if __name__ == "__main__":
    main()

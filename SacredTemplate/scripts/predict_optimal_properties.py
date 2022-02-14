import pandas as pd
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from keras.models import load_model
import math
import numpy as np

def main():
    df_testing= pd.read_csv('..\data\database.csv')
    df_training=pd.read_excel('..\data\database_new.xlsx')

    X_train = df_training.iloc[:, :8]
    Y_train = df_training.iloc[:, 25:28]
    X_test = df_testing.loc[:, ['wavelength', 'fractal_dimension', 'fraction_of_coating', 'primary_particle_size',
                   'number_of_primary_particles',
                   'vol_equi_radius_outer', 'vol_equi_radius_inner', 'equi_mobility_dia']]
    # Normalizaing Min max
    scaling_x = MinMaxScaler()
    scaling_y = MinMaxScaler()
    X_train = scaling_x.fit_transform(X_train)
    X_test = scaling_x.transform(X_test)
    Y_train = scaling_y.fit_transform(Y_train)

    model = load_model('best_model.hdf5')
    Y_test = model.predict(X_test)
    Y_test = scaling_y.transform(Y_test)
    #print(Y_test)

    # Computing others
    Y_test = pd.DataFrame(data=Y_test, columns=["q_abs", "q_sca", "g"])

    wavelength = df_testing['wavelength']
    fractal_dimension = df_testing['fractal_dimension']
    fraction_of_coating = df_testing['fraction_of_coating']
    primary_particle_size = df_testing['primary_particle_size']
    number_of_primary_particles = df_testing['number_of_primary_particles']
    vol_equi_radius_inner = df_testing['vol_equi_radius_inner']
    vol_equi_radius_outer = df_testing['vol_equi_radius_outer']
    equi_mobility_dia = df_testing['equi_mobility_dia']

    mie_epsilon = np.zeros_like(wavelength) + 2 #Check
    length_scale_factor = 2 * math.pi / wavelength
    m_real_bc= np.zeros_like(wavelength) + 2 #Check
    m_im_bc= np.zeros_like(wavelength) + 2 #Check
    m_real_organics= np.zeros_like(wavelength) + 2 #Check
    m_im_organics= np.zeros_like(wavelength) + 2 #Check

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

    q_abs = Y_test['q_abs']
    q_sca = Y_test['q_sca']
    q_ext = q_abs + q_sca
    g = Y_test['g']
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
    final_dataset.to_csv('..\data\predicted_forward_dataset.csv', index=False)

if __name__ == "__main__":
    main()
# Black-Carbon-Aerosol-Project
#Results for Evaluation on Mean absolute error and trained using Mean squared error:
Type of split for evaluation | Best results yet(Mean absolute error)
------------- | -------------
Random split | Mean absolute error on test set [q_abs, q_sca, g]:-   [0.00238034 0.00192617 0.00298648]
Fractal_dimension=2.1, 2.2 left out for evaluation  | Mean absolute error on test set [q_abs, q_sca, g]:-   [0.01714607 0.0415282  0.02505202]
Fraction of coating=40, 50 left out for evaluation  | Mean absolute error on test set [q_abs, q_sca, g]:-   [0.00939356 0.00609494 0.01331163]

#Results for inverse problem:(Trained using MSE and evaulated using mean absolute error)
#Standardized output values too

Type of split for evaluation | Best results yet(Mean absolute error)- Using wavelength, vol_equi_radius_outer, primary_particle_size, q_ext,q_abs,q_sca,g as input
------------- | -------------
Random split | Mean absolute error on test set [fractal_dimension, fraction_of_coating]:-   [0.02767658 0.4789153 ]




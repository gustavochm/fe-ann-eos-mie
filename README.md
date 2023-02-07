
# Supplementary information: Thermodynamically-consistent machine-learning based Equation of State for the Mie fluid

This repository is part of the supplemetary information of the article *"Thermodynamically-consistent machine-learning based Equation of State for the Mie fluid" (Submitted to Journal of Physical Chemistry B)* by Gustavo Chaparro and Erich A. Müller. This article introduces an artificial neural network (ANN) based equation of state for the Mie fluid **(FE-ANN EoS)**, this EoS is formulated as follows:

$$ A^{res} = ANN(\alpha_{vdw}, \rho, 1/T) - ANN(\alpha_{vdw}, \rho=0.0, 1/T) $$

Where, $A^{res}$ is the dimensionless residual Helmholtz free energy, $\rho$ is the dimensionless density, $T$ is the dimensionless temperature and $\alpha_{vdw}$ is defined a the Mie fluid as follows:

$$ \alpha_{vdw} = \mathcal{C}_{Mie} \left[ \left(\frac{1}{\lambda_a-3} \right) - \left(\frac{1}{\lambda_r-3} \right) \right], \qquad \mathcal{C}_{Mie} = \frac{\lambda_r}{\lambda_r- \lambda_a} \left( \frac{\lambda_r}{\lambda_r}\right)^{\frac{\lambda_a}{\lambda_r - \lambda_a}} $$

This model has been trained using thermophysical properties of the Mie fluid that include first-order derivative properties such as the compressibility factor ($Z$) and the internal energy ($U$), and second-order derivative properties such as the isobaric heat capacity ($C_V$), the thermal pressure coefficient ($\gamma_V$), the isothermal compressibility ($\rho\kappa_T$), thermal expansion coefficient ($\alpha_P$), adibatic index ($\gamma$) and the Joule-Thomson coefficient ($\mu_{JT}$).


## Use of the FE-ANN EoS

The saved model is found located at the [fe-ann-eos](./fe-ann-eos) folder. The model can be loaded using tensorflow.

The following examples of use of the FE-ANN EoS are provided in the [examples](./examples) folder.

1. Computing 1st and 2nd order derivative properties with the FE-ANN EoS.
2. Computing fluid phase equilibria with the FE-ANN EoS.



## Prerequisites

- numpy
- scipy
- matplotlib
- tensorflow (tested version > v2.5.0)

## License information

See ``LICENSE.md`` for information on the terms & conditions for usage of this software, and a DISCLAIMER OF ALL WARRANTIES.

Although not required by the license, if it is convenient for you, please cite this  if used in your work. Please also consider contributing any changes you make back, and benefit the community.
# Supplementary information: "Development of thermodynamically-consistent machine-learning Equations of State: Application to the Mie fluid"

This repository is part of the supplementary information of the article *["Development of thermodynamically-consistent machine-learning Equations of State: Application to the Mie fluid"](https://doi.org/10.1063/5.0146634)* by Gustavo Chaparro and Erich A. Müller. This article introduces an artificial neural network (ANN) based equation of state for the Mie fluid **, FE-ANN EoS**. 

The Mie potential is described as follows:

$$ U^{Mie} = \mathcal{C}_{Mie} \epsilon \left[ \left(\frac{\sigma}{r}\right)^{\lambda_r} -  \left(\frac{\sigma}{r}\right)^{\lambda_a} \right], \quad \mathcal{C}_{Mie} = \frac{\lambda_r}{\lambda_r- \lambda_a} \left( \frac{\lambda_r}{\lambda_a}\right)^{\frac{\lambda_a}{\lambda_r - \lambda_a}} $$ 

Here, $\epsilon$ represents the potential well depth, $\sigma$ is the effective monomer diameter, $r$ is the centre-to-centre distance between two Mie monomers, and $\lambda_r$ and $\lambda_a$ are the repulsive and attractive exponent, respectively.

The FE-ANN EoS models the Helmholtz free energy of the Mie fluid as follows:


$$ A^{res} = ANN(\alpha_{vdw}, \rho, 1/T) - ANN(\alpha_{vdw}, \rho=0, 1/T) $$

Where, $A^{res}$ is the dimensionless residual Helmholtz free energy, $\rho$ is the dimensionless density, $T$ is the dimensionless temperature and $\alpha_{vdw}$ is defined a the Mie fluid as follows:

$$ \alpha_{vdw} = \mathcal{C}_{Mie} \left[ \left(\frac{1}{\lambda_a-3} \right) - \left(\frac{1}{\lambda_r-3} \right) \right]$$

This model has been trained using thermophysical properties of the Mie fluid that include first-order derivative properties such as the compressibility factor ( $Z$ ) and the internal energy ( $U$ ), and second-order derivative properties such as the isobaric heat capacity ( $C_V$ ), the thermal pressure coefficient ( $\gamma_V$ ), the isothermal compressibility ( $\rho\kappa_T$ ), thermal expansion coefficient ( $\alpha_P$ ), adiabatic index ( $\gamma$ ) and the Joule-Thomson coefficient ( $\mu_{JT}$ ).


## Use of the FE-ANN EoS

The saved model is located in the [``fe-ann-eos``](./fe-ann-eos) folder. The model can be loaded using TensorFlow.

The following examples of using the FE-ANN EoS are provided in the [``examples``](./examples) folder.

1. Computing 1st and 2nd order derivative properties with the FE-ANN EoS.
2. Computing fluid phase equilibria with the FE-ANN EoS.


## Prerequisites

- NumPy
- SciPy
- matplotlib
- TensorFlow (tested on version 2.4.1)

## License information

See ``LICENSE.md`` for information on the terms & conditions for usage of this software and a DISCLAIMER OF ALL WARRANTIES.

Although not required by the license, if it is convenient for you, please cite this if used in your work. Please also consider contributing any changes you make back, and benefit the community.

# Supplementary information: Thermodynamically-consistent machine-learning based Equation of State for the Mie fluid

This repository is part of the supplemetary information of the article *"Thermodynamically-consistent machine-learning based Equation of State for the Mie fluid" (Submitted to Journal of Physical Chemistry B)* by Gustavo Chaparro and Erich A. Müller. This article introduces an artificial neural network (ANN) based equation of state for the Mie fluid **(FE-ANN EoS)**, this EoS is formulated as follows:

$$ A^{*, res} = ANN(\alpha_{vdw}, \rho^*, 1/T^*) - ANN(\alpha_{vdw}, \rho^*=0.0, 1/T^*) $$

Where, $A^{*, res}$ is the dimensionless residual Helmholtz free energy, $\rho^*$ is the dimensionless density, $T^*$ is the dimensionless temperature and $\alpha_{vdw}$ if defined for the Mie fluid as follows:

$$ \alpha_{vdw} = \mathcal{C}_{Mie} \left[ \left(\frac{1}{\lambda_a-3} \right) - \left(\frac{1}{\lambda_r-3} \right) \right], \qquad \mathcal{C}_{Mie} = \frac{\lambda_r}{\lambda_r- \lambda_a} \left( \frac{\lambda_r}{\lambda_r}\right)^{\frac{\lambda_a}{\lambda_r - \lambda_a}} $$

This model has been trained using derivative information of the Mie


## Use of the FE-ANN EoS

The saved model with tensorflow is found in the [fe-ann-eos folder](./fe-ann-eos)



## Prerequisites

- numpy
- scipy
- tensorflow 

## License information

See ``LICENSE.md`` for information on the terms & conditions for usage of this software, and a DISCLAIMER OF ALL WARRANTIES.

Although not required by the license, if it is convenient for you, please cite this  if used in your work. Please also consider contributing any changes you make back, and benefit the community.
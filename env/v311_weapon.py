"""Same Equation (7)/(8), with a transparent four-kilometre calibration."""
import math
from .v37_weapon import MADSACGaussianFiniteAmmoWeapon, NOISE_SCALE

EFFECTIVE_HIT_DISTANCE = 4000. / math.log(6.)

# Geometry and single shared Gaussian draw are reused without modification.
class MADSACFourKmWeapon(MADSACGaussianFiniteAmmoWeapon):
    pass

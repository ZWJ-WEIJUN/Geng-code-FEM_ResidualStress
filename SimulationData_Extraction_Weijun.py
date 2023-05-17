import numpy as np
from matplotlib import pyplot as plt

# with open('thinwall_test_fr_control_lp1.7.npy', 'rb') as f:
#     nodal_T2 = np.load(f)
#     nodal_A2 = np.load(f)
#     LSC = np.load(f)
#     energy_lose = np.load(f)
#     laser_energy = np.load(f)
#     median_temperature = np.load(f)
#     lp_layer = np.load(f)
#     fr_layer = np.load(f)
#     dp_error = np.load(f)
#     dp_dedl = np.load(f)


with open('thinwall_test_lp_control_lp1.7.npy', 'rb') as lp:
    nodal_T2 = np.load(lp)
    nodal_A2 = np.load(lp)
    LSC = np.load(lp)
    energy_lose = np.load(lp)
    laser_energy = np.load(lp)
    median_temperature = np.load(lp)
    lp_layer = np.load(lp)
    # fr_layer = np.load(lp)
    dp_error = np.load(lp)
    dp_dedl = np.load(lp)

print(lp_layer)
# print(fr_layer)
layer = np.arange(1, 51, dtype=np.int)
plt.plot(layer, lp_layer, label='Laser power(KW) per layer')
# plt.plot(layer, fr_layer, label='Feedrate(mm/s) per layer')
plt.show()

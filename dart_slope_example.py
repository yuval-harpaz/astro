import matplotlib.pyplot as plt
import numpy as np
# center pixel was 152.55 red and 60.04 green. this is a 154% increase, over a span of 148nm (between filter centers). when I correct it for 100nm I get 87.7% increase
r = 152.55
g = 60.04
per_nm = (r / g) ** (1/(616.6 - 468.6))
slope = per_nm ** 100 * 100 - 100
y = (per_nm ** np.arange(0,149)) * 100 - 100
plt.figure()
plt.plot(np.arange(0,149)+468.6, y)
plt.grid()
plt.xticks([468.6, 468.6+100, 616.6])
plt.yticks([y[0], y[100], y[148]])
plt.xlabel('wave length (nm)')
plt.ylabel('increase (%)')

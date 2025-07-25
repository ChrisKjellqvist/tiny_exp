
x = [0,1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
avg_err = [1.04, 0.878, 0.798, 0.69, 0.58, 0.45, 0.31, 0.19, 0.07, 0.02, 0.005, 0.001, 4e-6]
max_err = [5.66, 5.75, 5.75, 5.70, 5.71, 5.69, 5.28, 5.31, 3.41, 1.82, 0.9714, 0.507, 1e-5]

avg_err_silu = [0.23, 0.05, 0.015, 0.003]
max_err_silu = [2.27, 0.81, 0.24, 0.0655]



import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(4, 3), dpi=800)
ax.plot(x, avg_err)
ax.grid(True)
ax.set_xticks(range(0, 13, 2))

color = 'tab:red'
ax.set_xlabel('# Mantissa Bits as Index')
ax.set_ylabel('Average Error (%)', color=color)
ax.plot(x, avg_err, color=color)
ax.tick_params(axis='y', labelcolor=color)

ax2 = ax.twinx()  # instantiate a second Axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Maximum Relative Error (%)', color=color)  # we already handled the x-label with ax1
ax2.plot(x, max_err, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
fig.savefig("maxavgerr.pdf")
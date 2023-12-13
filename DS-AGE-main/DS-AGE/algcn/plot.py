import matplotlib.pyplot as plt
import numpy as np

citeseer_macro_mean = 100 * np.array([0.67554, 0.661704, 0.666086, 0.682008, 0.659772, 0.667594, 0.678556, 0.662176])
citeseer_micro_mean = 100 * np.array([0.722, 0.725, 0.7212, 0.7296, 0.7178, 0.7246, 0.727, 0.7204])
citeseer_macro_std = 100 * np.array([0.009854006799, 0.01389851899, 0.01597379823, 0.01299678691, 0.01960610033, 0.01161106283, 0.01202094755, 0.01622913214])
citeseer_micro_std = 100 * np.array([0.007176350047, 0.01031988372, 0.01177709642, 0.008619744776, 0.01091787525, 0.01692040189, 0.00316227766, 0.02403747075])

citeseer_simple_macro_mean = 100 * np.array([0.668178, 0.665818, 0.66614, 0.667492, 0.678418, 0.667046, 0.677746, 0.663292])
citeseer_simple_micro_mean = 100 * np.array([0.7266, 0.7228, 0.7206, 0.719, 0.7272, 0.724, 0.7258, 0.7242])
citeseer_simple_macro_std = 100 * np.array([0.0149845744, 0.02433581147, 0.008139625299, 0.01122404873, 0.01117057161, 0.0152960691, 0.01076299587, 0.01765044957])
citeseer_simple_micro_std = 100 * np.array([0.01397497764, 0.01764086166, 0.01283744523, 0.005431390246, 0.005449770637, 0.007483314774, 0.01025670512, 0.01709385855])

# Cora data
cora_macro_mean = 100 * np.array([0.801444, 0.816486, 0.804168, 0.806266, 0.815984, 0.81167, 0.801662, 0.810942])
cora_micro_mean = 100 * np.array([0.8136, 0.8294, 0.8194, 0.82, 0.8304, 0.8234, 0.815, 0.8212])
cora_macro_std = 100 * np.array([0.01406780651, 0.01360618352, 0.01266348965, 0.008402837616, 0.008388371713, 0.009201907411, 0.012692406, 0.008012594461])
cora_micro_std = 100 * np.array([0.01550161282, 0.01277888884, 0.01035857133, 0.007745966692, 0.009989994995, 0.008876936408, 0.008972179222, 0.008729261137])

cora_simple_macro_mean = 100 * np.array([0.797262, 0.804796, 0.802736, 0.799542, 0.801976, 0.801084, 0.798558, 0.803506])
cora_simple_micro_mean = 100 * np.array([0.809, 0.8156, 0.816, 0.8118, 0.8174, 0.8146, 0.815, 0.8126])
cora_simple_macro_std = 100 * np.array([0.005899899999, 0.006679946108, 0.008378068393, 0.01553469086, 0.005177999614, 0.007596033834, 0.009840321133, 0.0172163826])
cora_simple_micro_std = 100 * np.array([0.00790569415, 0.007021395873, 0.008746427842, 0.01645296326, 0.007127411872, 0.007092249291, 0.008972179222, 0.01933390804])
labels = ['AGE', 'AGE+F', 'AGE+S', 'AGE+E', 'AGE+F+S', 'AGE+F+E', 'AGE+S+E', 'All']

fig, axs = plt.subplots(2)

bar_width = 0.35
index = np.arange(len(labels))

axs[0].bar(index, citeseer_macro_mean - 60, bar_width, label='GCN',bottom=60)
axs[0].bar(index + bar_width, citeseer_simple_macro_mean - 60, bar_width, label='Simple GCN', bottom=60)
axs[0].errorbar(index, citeseer_macro_mean, yerr=citeseer_macro_std, fmt='none', capsize=5, capthick=2, ecolor='black')
axs[0].errorbar(index + bar_width, citeseer_simple_macro_mean, yerr=citeseer_simple_macro_std, fmt='none', capsize=5, capthick=2, ecolor='black')

axs[0].set_ylim(60, 70)

axs[0].set_xticklabels(labels)
axs[0].set_ylabel('Values')
axs[0].set_title('Comparison of Macro F1 Scores GCN vs Simple GCN on Citeseer')
axs[0].set_xticks(index + bar_width / 2, labels)
axs[0].legend()

axs[1].bar(index, citeseer_micro_mean - 65, bar_width, label='GCN',bottom=65)
axs[1].bar(index + bar_width, citeseer_simple_micro_mean - 65, bar_width, label='Simple GCN', bottom=65)
axs[1].errorbar(index, citeseer_micro_mean, yerr=citeseer_micro_std, fmt='none', capsize=5, capthick=2, ecolor='black')
axs[1].errorbar(index + bar_width, citeseer_simple_micro_mean, yerr=citeseer_simple_micro_std, fmt='none', capsize=5, capthick=2, ecolor='black')

axs[1].set_ylim(65, 75)

axs[1].set_xticklabels(labels)
axs[1].set_ylabel('Values')
axs[1].set_title('Comparison of Micro F1 Scores of GCN vs Simple GCN on Citeseer')
axs[1].set_xticks(index + bar_width / 2, labels)
axs[1].legend()


fig2, ax2 = plt.subplots(2)
ax2[0].bar(index, cora_macro_mean - 75, bar_width, label='GCN',bottom=75)
ax2[0].bar(index + bar_width, cora_simple_macro_mean - 75, bar_width, label='Simple GCN', bottom=75)
ax2[0].errorbar(index, cora_macro_mean, yerr=cora_macro_std, fmt='none', capsize=5, capthick=2, ecolor='black')
ax2[0].errorbar(index + bar_width, cora_simple_macro_mean, yerr=cora_simple_macro_std, fmt='none', capsize=5, capthick=2, ecolor='black')

ax2[0].set_ylim(75, 85)

ax2[0].set_xticklabels(labels)
ax2[0].set_ylabel('Values')
ax2[0].set_title('Comparison of Macro F1 Scores GCN vs Simple GCN on Cora')
ax2[0].set_xticks(index + bar_width / 2, labels)
ax2[0].legend()

ax2[1].bar(index, cora_micro_mean - 75, bar_width, label='GCN',bottom=75)
ax2[1].bar(index + bar_width, cora_simple_micro_mean - 75, bar_width, label='Simple GCN', bottom=75)
ax2[1].errorbar(index, cora_micro_mean, yerr=cora_micro_std, fmt='none', capsize=5, capthick=2, ecolor='black')
ax2[1].errorbar(index + bar_width, cora_simple_micro_mean, yerr=cora_simple_micro_std, fmt='none', capsize=5, capthick=2, ecolor='black')

ax2[1].set_ylim(75, 85)

ax2[1].set_xticklabels(labels)
ax2[1].set_ylabel('Values')
ax2[1].set_title('Comparison of Micro F1 Scores of GCN vs Simple GCN on Cora')
ax2[1].set_xticks(index + bar_width / 2, labels)
ax2[1].legend()
plt.tight_layout()
plt.show()
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# ------------------------
# 1. Define file paths
# ------------------------

crp_files = [
    ("fine_amount_real.csv", "Fine Amount Real Log"),
    ("fine_amount_unconditional.csv", "Fine Amount Unconditional Log"),
    ("fine_amount_transition_list.csv", "Fine Amount Transition List Log"),
    ("fine_amount_simulation.csv", "Fine Amount Simulation Log")
]

leucocytes_files = [
    ("total_payment_amount_real.csv", "Total Payement Real Log"),
    ("total_payment_amount_unconditional.csv", "Total Payement Unconditional Log"),
    ("total_payment_amount_transition-list.csv", "Total Payement Transition List Log"),
    ("total_payment_amount_simulation.csv", "Total Payement Simulation Log")
]

# ------------------------
# 2. Plot CRP comparison
# ------------------------

plt.figure(figsize=(10, 6))
for file, label in crp_files:
    df = pd.read_csv(file)
    if not df.empty and "value" in df:
        sns.kdeplot(df["value"], fill=False, label=label)

plt.title("Fine Amount Density Comparison (4 Logs)")
plt.xlabel("Fine Amount")
plt.ylabel("Density")
plt.xlim(0, 200) 
plt.legend()
plt.tight_layout()
plt.savefig("fine_amount_density_comparison.png")
plt.show()

# ------------------------
# 3. Plot Leucocytes comparison
# ------------------------

plt.figure(figsize=(10, 6))
for file, label in leucocytes_files:
    df = pd.read_csv(file)
    if not df.empty and "value" in df:
        sns.kdeplot(df["value"], fill=False, label=label)

plt.title("Total Payement Density Comparison (4 Logs)")
plt.xlabel("Total Payement")
plt.ylabel("Density")
plt.xlim(0, 100) 
plt.legend()
plt.tight_layout()
plt.savefig("total_payement_density_comparison.png")
plt.show()


"""

leucocytes_files = [
    ("leucocytes_real.csv", "eucocytes Real Log"),
    ("leucocytes_no_petri.csv", "Leucocytes Unconditional Log"),
    ("leucocytes_transition_list.csv", "Leucocytes Transition List Log"),
    ("leucocytes_simulation.csv", "Leucocytes Simulation Log")
]

"""
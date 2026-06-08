import pandas as pd
import matplotlib.pyplot as plt

# Carrega o CSV sem cabeçalho
df = pd.read_csv(
    "10-10000-1200.txt",
    header=None,
    names=["fitness", "epoca"]
)

# Ordena por época (opcional)
df = df.sort_values("epoca")

# Plota
plt.figure(figsize=(10, 6))
plt.plot(df["epoca"], df["fitness"], marker=".")

plt.title("Fitness por Época")
plt.xlabel("Época")
plt.ylabel("Fitness")
plt.grid(True)

plt.tight_layout()
plt.show()
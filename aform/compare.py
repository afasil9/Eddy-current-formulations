import re
import pandas as pd
import matplotlib.pyplot as plt

def extract_norms(filename):
    """Extracts iteration number (n) and norms of a_n, B, E, J from file."""
    data = []
    with open(filename, 'r') as f:
        text = f.read()
    
    # Find each block starting with "n is ..."
    blocks = re.split(r'\n(?=n is)', text)
    
    for block in blocks:
        n_match = re.search(r"n is (\d+)", block)
        a_match = re.search(r"Norm of a_n:\s*([0-9.eE+-]+)", block)
        b_match = re.search(r"Norm of B:\s*([0-9.eE+-]+)", block)
        e_match = re.search(r"Norm of E:\s*([0-9.eE+-]+)", block)
        j_match = re.search(r"Norm of J:\s*([0-9.eE+-]+)", block)
        
        if n_match and a_match and b_match and e_match and j_match:
            n = int(n_match.group(1))
            data.append({
                "n": n,
                "a_n": float(a_match.group(1)),
                "B": float(b_match.group(1)),
                "E": float(e_match.group(1)),
                "J": float(j_match.group(1))
            })
    
    return pd.DataFrame(data)

# --- Read both files ---
file1 = "output.log"
file2 = "output_interior.log"

df1 = extract_norms(file1)
df2 = extract_norms(file2)

# --- Plotting ---
plt.figure(figsize=(10, 6))
for norm in ["a_n", "B", "E", "J"]:
    plt.plot(df1["n"], df1[norm], marker='o', label=f"{norm} (file1)")
    plt.plot(df2["n"], df2[norm], marker='x', linestyle='--', label=f"{norm} (file2)")

plt.yscale("log")  # often norms vary exponentially — log scale helps
plt.xlabel("Iteration (n)")
plt.ylabel("Norm value (log scale)")
plt.title("Comparison of Norms between Two Simulations")
plt.legend()
plt.grid(True, which="both", ls="--", lw=0.5)
plt.tight_layout()
plt.show()

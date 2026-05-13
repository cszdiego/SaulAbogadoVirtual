"""Creates the required project directory structure."""
import os

dirs = [
    "data/raw",
    "data/processed",
    "vector_db",
    "scripts",
]

for d in dirs:
    os.makedirs(d, exist_ok=True)
    print(f"OK  {d}/")

print("\nEstructura de carpetas lista.")

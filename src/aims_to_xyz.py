# aims_to_xyz.py
from pathlib import Path
import sys

if len(sys.argv) not in (2, 3):
    raise SystemExit(
        "Usage: python aims_to_xyz.py geometry2.txt [geometry2.xyz]"
    )

input_file = Path(sys.argv[1])
output_file = (
    Path(sys.argv[2])
    if len(sys.argv) == 3
    else input_file.with_suffix(".xyz")
)

atoms = []

with input_file.open() as f:
    for line in f:
        fields = line.split()

        if fields and fields[0] == "atom":
            if len(fields) < 5:
                raise ValueError(f"Malformed atom line:\n{line}")

            x, y, z = map(float, fields[1:4])
            element = fields[4]
            atoms.append((element, x, y, z))

with output_file.open("w") as f:
    f.write(f"{len(atoms)}\n")
    f.write(f"Converted from {input_file.name}\n")

    for element, x, y, z in atoms:
        f.write(f"{element:<2s} {x:16.8f} {y:16.8f} {z:16.8f}\n")

print(f"Wrote {len(atoms)} atoms to {output_file}")
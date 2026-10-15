#!/usr/bin/env python3
"""
Analyze Pb/Ca-Br geometry in a CsPbBr3 nanocrystal

Default cutoffs:
    M-Br bond cutoff = 3.5 A
    M-Br-M angle calculation requires both M-Br distances <= 3.5 A

"""

import argparse
from pathlib import Path
from itertools import combinations
import csv

import numpy as np
import matplotlib.pyplot as plt


DEFAULT_BOND_CUTOFF = 3.5  # Angstrom
DEFAULT_BINS = 40

METALS = {"Pb", "Ca"}
ANION = "Br"


def read_xyz(filename):
    with open(filename, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    start = 0
    try:
        natoms = int(lines[0].split()[0])
        if len(lines) >= natoms + 2:
            start = 2
            atom_lines = lines[start:start + natoms]
        else:
            atom_lines = lines
    except (ValueError, IndexError):
        atom_lines = lines

    elements = []
    coords = []

    for line in atom_lines:
        fields = line.split()
        if len(fields) < 4:
            continue

        try:
            xyz = [float(fields[1]), float(fields[2]), float(fields[3])]
        except ValueError:
            continue

        elements.append(fields[0])
        coords.append(xyz)

    if not coords:
        raise ValueError(f"No atom coordinates found in {filename}")

    return elements, np.asarray(coords, dtype=float)


# Geometry utilities

def distance(a, b):
    return np.linalg.norm(a - b)


def angle(a, vertex, c):
    v1 = a - vertex
    v2 = c - vertex

    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)

    if n1 == 0 or n2 == 0:
        return np.nan

    cosang = np.dot(v1, v2) / (n1 * n2)
    cosang = np.clip(cosang, -1.0, 1.0)

    return np.degrees(np.arccos(cosang))



def find_metal_br_bonds(elements, coords, cutoff):
    """
    Identify all Pb-Br and Ca-Br bonds within cutoff.
    """
    metals = [i for i, e in enumerate(elements) if e in METALS]
    brs = [i for i, e in enumerate(elements) if e == ANION]

    bonds = []

    for m in metals:
        for b in brs:
            d = distance(coords[m], coords[b])

            if d <= cutoff:
                bonds.append({
                    "metal_index": m,
                    "metal": elements[m],
                    "br_index": b,
                    "distance_A": d,
                })

    return bonds


def build_metal_neighbors(bonds):
    """
    Build metal -> Br neighbor mapping.
    """
    neighbors = {}

    for bond in bonds:
        m = bond["metal_index"]
        neighbors.setdefault(m, []).append(
            (bond["br_index"], bond["distance_A"])
        )

    return neighbors



def make_bond_records(elements, bonds):
    records = []

    for bond in bonds:
        records.append({
            "metal_index": bond["metal_index"],
            "metal": bond["metal"],
            "br_index": bond["br_index"],
            "distance_A": bond["distance_A"],
        })

    return records


def make_angle_records(elements, coords, bonds, cutoff):
    """
    For every Br atom, find all Pb/Ca neighbors within cutoff.

    For every pair of metal neighbors around the same Br, calculate
    M-Br-M.

    outputs:
        Pb-Br-Pb angles
        Pb-Br-Ca angles
        Ca-Br-Ca angles
    """
    br_neighbors = {}

    for bond in bonds:
        br = bond["br_index"]
        m = bond["metal_index"]

        br_neighbors.setdefault(br, []).append(m)

    records = []

    for br, metals in br_neighbors.items():
        for m1, m2 in combinations(metals, 2):

            labels = sorted([elements[m1], elements[m2]])
            pair_type = f"{labels[0]}-{ANION}-{labels[1]}"

            theta = angle(
                coords[m1],
                coords[br],
                coords[m2],
            )

            records.append({
                "br_index": br,
                "metal1_index": m1,
                "metal1": elements[m1],
                "metal2_index": m2,
                "metal2": elements[m2],
                "angle_deg": theta,
                "type": pair_type,
            })

    return records


def calculate_site_data(elements, coords, neighbors):
    records = []

    for m, nbrs in neighbors.items():
        if elements[m] not in METALS:
            continue

        distances = [x[1] for x in nbrs]

        record = {
            "metal_index": m,
            "metal": elements[m],
            "coordination_number": len(nbrs),
            "mean_M_Br_A": np.mean(distances) if distances else np.nan,
            "std_M_Br_A": np.std(distances) if distances else np.nan,
            "bond_length_distortion": np.nan,
            "angular_distortion_deg": np.nan,
        }

        records.append(record)

    return records


def plot_histogram(values_by_type, xlabel, title, output_file,
                   bins=DEFAULT_BINS, xlim=None):
    plt.figure(figsize=(7, 5))

    all_values = np.concatenate(list(values_by_type.values()))

    bin_edges = np.linspace(
        all_values.min(),
        all_values.max(),
        bins + 1
    )

    for label, values in values_by_type.items():
        if len(values) == 0:
            continue

        print(label)

        if (label=='Pb-Br'):
            plt.hist(
                values, 
                bins=bin_edges,
                alpha=0.55,
                density=False,
                label=label,
                color='blue'
            )
        if (label=='Ca-Br'):
            plt.hist(
                values, 
                bins=bin_edges,
                alpha=0.55,
                density=False,
                label=label,
                color='orange'
            )
        if (label=='Pb-Br-Pb'):
            plt.hist(
                values, 
                bins=bin_edges,
                alpha=0.55,
                density=False,
                label=label,
                color='blue'
            )
        if (label=='Ca-Br-Ca'):
            plt.hist(
                values, 
                bins=bin_edges,
                alpha=0.55,
                density=False,
                label=label,
                color='orange'
            )
        if (label=='Ca-Br-Pb'):
            plt.hist(
                values, 
                bins=bin_edges,
                alpha=0.55,
                density=False,
                label=label,
                color='green'
            )

        

    plt.xlabel(xlabel)
    plt.ylabel("Number")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    #plt.xlim([2.7,3.2])
    plt.xlim(xlim)
    plt.savefig(output_file, dpi=300)
    plt.close()


def plot_site_histogram(values_by_type, xlabel, title, output_file,
                        bins=30):
    plt.figure(figsize=(7, 5))

    for label, values in values_by_type.items():
        if len(values) == 0:
            continue

        plt.hist(
            values,
            bins=bins,
            alpha=0.55,
            density=False,
            label=label,
        )

    plt.xlabel(xlabel)
    plt.ylabel("Number")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()


def write_csv(filename, records, fieldnames):
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

def analyze(filename, cutoff, bins):

    input_path = Path(filename)
    prefix = input_path.with_suffix("")

    elements, coords = read_xyz(filename)

    print(f"Read {len(elements)} atoms from {filename}")

    counts = {}
    for e in elements:
        counts[e] = counts.get(e, 0) + 1

    print("Composition:")
    for e, n in counts.items():
        print(f"  {e:>3s}: {n}")


    bonds = find_metal_br_bonds(elements, coords, cutoff)

    bond_records = make_bond_records(elements, bonds)

    bond_values = {}

    for bond in bonds:
        label = f"{bond['metal']}-Br"
        bond_values.setdefault(label, []).append(
            bond["distance_A"]
        )

    print("\nM-Br bonds:")
    for label, values in bond_values.items():
        print(
            f"  {label}: {len(values)} bonds, "
            f"mean = {np.mean(values):.4f} A, "
            f"std = {np.std(values):.4f} A, "
            f"range = {np.min(values):.4f}-{np.max(values):.4f} A"
        )

    plot_histogram(
        bond_values,
        xlabel="M–Br distance (Å)",
        title="Metal–Br bond-length distribution",
        output_file=f"{prefix}_bond_lengths.png",
        bins=bins,
        xlim=[2.7,3.2],
    )

    angle_records = make_angle_records(
        elements, coords, bonds, cutoff
    )

    angle_values = {}

    for rec in angle_records:
        angle_values.setdefault(rec["type"], []).append(
            rec["angle_deg"]
        )

    print("\nM-Br-M angles:")
    for label, values in angle_values.items():
        print(
            f"  {label}: {len(values)} angles, "
            f"mean = {np.mean(values):.3f} deg, "
            f"std = {np.std(values):.3f} deg"
        )

    plot_histogram(
        angle_values,
        xlabel="M–Br–M angle (degrees)",
        title="Metal–Br–metal angle distribution",
        output_file=f"{prefix}_bond_angles.png",
        bins=bins,
        xlim=[148,173],
    )


    write_csv(
        f"{prefix}_bond_data.csv",
        bond_records,
        [
            "metal_index",
            "metal",
            "br_index",
            "distance_A",
        ],
    )

    write_csv(
        f"{prefix}_angle_data.csv",
        angle_records,
        [
            "br_index",
            "metal1_index",
            "metal1",
            "metal2_index",
            "metal2",
            "angle_deg",
            "type",
        ],
    )


    print("\nOutput files:")
    print(f"  {prefix}_bond_lengths.png")
    print(f"  {prefix}_bond_angles.png")
    print(f"  {prefix}_bond_data.csv")
    print(f"  {prefix}_angle_data.csv")
    print(f"  {prefix}_site_data.csv")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Pb/Ca-Br geometry in a CsPbBr3 nanocrystal."
    )

    parser.add_argument(
        "xyz",
        help="XYZ geometry file",
    )

    parser.add_argument(
        "--cutoff",
        type=float,
        default=DEFAULT_BOND_CUTOFF,
        help=f"M-Br bond cutoff in Angstrom (default: {DEFAULT_BOND_CUTOFF})",
    )

    parser.add_argument(
        "--bins",
        type=int,
        default=DEFAULT_BINS,
        help=f"Number of histogram bins (default: {DEFAULT_BINS})",
    )

    args = parser.parse_args()

    analyze(
        args.xyz,
        cutoff=args.cutoff,
        bins=args.bins,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Analyze absolute octahedral tilts in a CsPbBr3 nanocrystal.

For every complete Pb/CaBr6 octahedron, the three local octahedral axes are
identified from the three approximately trans Br-M-Br pairs. Each local axis
is then compared to the crystallographic reference axes:

    x_ref = (1, 0, 0)
    y_ref = (0, 1, 0)
    z_ref = (0, 0, 1)

For each octahedron:
  1. Find its six nearest Br neighbors within --cutoff.
  2. Identify the three pairs of Br atoms that are approximately trans.
  3. Treat the vectors along those three Br-M-Br pairs as the local
     octahedral axes.
  4. Assign the three local axes to x/y/z by maximizing the total absolute
     alignment with the fixed reference axes.
  5. Calculate:
         tilt_x = acos(|u_x . x_ref|)
         tilt_y = acos(|u_y . y_ref|)
         tilt_z = acos(|u_z . z_ref|)
     and
         RMS tilt = sqrt((tilt_x^2 + tilt_y^2 + tilt_z^2)/3)

"""

import argparse
import csv
from itertools import permutations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METALS = {"Pb", "Ca"}
ANION = "Br"

# reference axes
REFERENCE_AXES = {
    "x": np.array([1.0, 0.0, 0.0]),
    "y": np.array([0.0, 1.0, 0.0]),
    "z": np.array([0.0, 0.0, 1.0]),
}


def read_xyz(filename):
    with open(filename, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        raise ValueError("Input file is empty.")

    try:
        n_atoms = int(lines[0].split()[0])
        if len(lines) >= n_atoms + 2:
            atom_lines = lines[2:2 + n_atoms]
        else:
            atom_lines = lines
    except ValueError:
        atom_lines = lines

    elements = []
    coords = []

    for line in atom_lines:
        parts = line.split()
        if len(parts) < 4:
            continue

        try:
            xyz = [float(parts[1]), float(parts[2]), float(parts[3])]
        except ValueError:
            continue

        elements.append(parts[0])
        coords.append(xyz)

    if not coords:
        raise ValueError("No atomic coordinates could be parsed.")

    return np.asarray(elements), np.asarray(coords, dtype=float)


def find_complete_octahedra(elements, coords, cutoff):
    """
    Find Pb/Ca sites with exactly six Br neighbors within cutoff.

    Returns:
        list of dictionaries with metal index and Br neighbor information.
    """
    metal_indices = np.where(np.isin(elements, list(METALS)))[0]
    br_indices = np.where(elements == ANION)[0]

    octahedra = []

    for metal_idx in metal_indices:
        distances = []

        for br_idx in br_indices:
            vec = coords[br_idx] - coords[metal_idx]
            dist = np.linalg.norm(vec)

            if dist <= cutoff:
                distances.append((br_idx, dist, vec))

        distances.sort(key=lambda x: x[1])

        # Only use complete octahedra.
        if len(distances) != 6:
            continue

        octahedra.append(
            {
                "metal_index": int(metal_idx),
                "metal": elements[metal_idx],
                "neighbors": distances,
            }
        )

    return octahedra


def identify_trans_pairs(neighbors):
    """
    Identify the three approximately trans Br-Br pairs.

    We choose the pairing that minimizes the total
    deviation of the three Br-M-Br angles from 180 degrees.

    """
    # There are 15 possible pairings of six objects into three pairs.
    # Generate all unique perfect matchings recursively.
    indices = list(range(len(neighbors)))

    def generate_pairings(items):
        if not items:
            yield []
            return

        first = items[0]
        for pos in range(1, len(items)):
            second = items[pos]
            remaining = items[1:pos] + items[pos + 1:]

            for rest in generate_pairings(remaining):
                yield [(first, second)] + rest

    best_pairs = None
    best_score = np.inf

    for pairing in generate_pairings(indices):
        score = 0.0

        for i, j in pairing:
            vi = neighbors[i][2]
            vj = neighbors[j][2]

            ni = np.linalg.norm(vi)
            nj = np.linalg.norm(vj)

            cos_theta = np.dot(vi, vj) / (ni * nj)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            theta = np.degrees(np.arccos(cos_theta))

            score += (180.0 - theta) ** 2

        if score < best_score:
            best_score = score
            best_pairs = pairing

    return best_pairs


def local_axes_from_octahedron(octahedron):

    neighbors = octahedron["neighbors"]
    pairs = identify_trans_pairs(neighbors)

    axes = []

    for pair in pairs:
        i, j = pair
        vi = neighbors[i][2]
        vj = neighbors[j][2]

        axis = vi / np.linalg.norm(vi)

        axes.append(
            {
                "vector": axis,
                "br_1": neighbors[i][0],
                "br_2": neighbors[j][0],
                "angle_deg": np.degrees(
                    np.arccos(
                        np.clip(
                            np.dot(vi, vj)
                            / (np.linalg.norm(vi) * np.linalg.norm(vj)),
                            -1.0,
                            1.0,
                        )
                    )
                ),
            }
        )

    return axes


def assign_axes_to_reference_axes(local_axes):
    """
    Assign the three local axes to x/y/z to maximize total alignment.
    """
    ref_names = ["x", "y", "z"]
    best_perm = None
    best_score = -np.inf

    for perm in permutations(range(3)):
        score = 0.0

        for ref_name, local_idx in zip(ref_names, perm):
            local_vec = local_axes[local_idx]["vector"]
            ref_vec = REFERENCE_AXES[ref_name]
            score += abs(np.dot(local_vec, ref_vec))

        if score > best_score:
            best_score = score
            best_perm = perm

    assigned = {}

    for ref_name, local_idx in zip(ref_names, best_perm):
        local_vec = local_axes[local_idx]["vector"]
        ref_vec = REFERENCE_AXES[ref_name]

        alignment = abs(np.dot(local_vec, ref_vec))
        alignment = np.clip(alignment, -1.0, 1.0)
        tilt = np.degrees(np.arccos(alignment))

        axis_data = dict(local_axes[local_idx])
        axis_data["reference_axis"] = ref_name
        axis_data["tilt_deg"] = float(tilt)
        axis_data["alignment"] = float(alignment)

        assigned[ref_name] = axis_data

    return assigned


def analyze_structure(elements, coords, cutoff):
    """Calculate absolute tilts for every complete octahedron."""
    octahedra = find_complete_octahedra(elements, coords, cutoff)

    site_records = []
    axis_records = []

    for octahedron in octahedra:
        local_axes = local_axes_from_octahedron(octahedron)
        assigned = assign_axes_to_reference_axes(local_axes)

        tilt_x = assigned["x"]["tilt_deg"]
        tilt_y = assigned["y"]["tilt_deg"]
        tilt_z = assigned["z"]["tilt_deg"]

        rms_tilt = np.sqrt(
            (tilt_x**2 + tilt_y**2 + tilt_z**2) / 3.0
        )

        metal_idx = octahedron["metal_index"]
        metal = octahedron["metal"]

        site_records.append(
            {
                "metal_index": metal_idx,
                "metal": metal,
                "tilt_x_deg": tilt_x,
                "tilt_y_deg": tilt_y,
                "tilt_z_deg": tilt_z,
                "rms_tilt_deg": rms_tilt,
                "br_neighbor_1": octahedron["neighbors"][0][0],
                "br_neighbor_2": octahedron["neighbors"][1][0],
                "br_neighbor_3": octahedron["neighbors"][2][0],
                "br_neighbor_4": octahedron["neighbors"][3][0],
                "br_neighbor_5": octahedron["neighbors"][4][0],
                "br_neighbor_6": octahedron["neighbors"][5][0],
            }
        )

        for ref_axis in ["x", "y", "z"]:
            data = assigned[ref_axis]

            axis_records.append(
                {
                    "metal_index": metal_idx,
                    "metal": metal,
                    "reference_axis": ref_axis,
                    "tilt_deg": data["tilt_deg"],
                    "alignment": data["alignment"],
                    "local_axis_x": data["vector"][0],
                    "local_axis_y": data["vector"][1],
                    "local_axis_z": data["vector"][2],
                    "br_1": data["br_1"],
                    "br_2": data["br_2"],
                    "trans_angle_deg": data["angle_deg"],
                }
            )

    return site_records, axis_records


def save_csv(records, filename):
    if not records:
        return

    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def main():
    parser = argparse.ArgumentParser(
        description="Calculate absolute Pb/Ca octahedral tilts relative to fixed x/y/z axes."
    )
    parser.add_argument("xyz", help="XYZ or coordinate-only XYZ structure")
    parser.add_argument(
        "--cutoff",
        type=float,
        default=4.0,
        help="M-Br cutoff in Angstrom (default: 4.0)",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=30,
        help="Number of histogram bins (default: 30)",
    )

    args = parser.parse_args()

    xyz_path = Path(args.xyz)
    stem = xyz_path.with_suffix("")

    elements, coords = read_xyz(xyz_path)

    site_records, axis_records = analyze_structure(
        elements, coords, args.cutoff
    )

    if not site_records:
        raise ValueError(
            "No complete six-coordinate Pb/Ca octahedra were found. "
            "Try increasing --cutoff."
        )

    # Separate site records by metal.
    pb_sites = [
        r for r in site_records if r["metal"] == "Pb"
    ]
    ca_sites = [
        r for r in site_records if r["metal"] == "Ca"
    ]

    # Save data.
    site_csv = stem.parent / f"{stem.name}_absolute_octahedral_tilts.csv"
    axis_csv = stem.parent / f"{stem.name}_absolute_octahedral_axis_data.csv"

    save_csv(site_records, site_csv)
    save_csv(axis_records, axis_csv)

    # ------------------------------------------------------------
    # Histogram of RMS absolute tilt
    # ------------------------------------------------------------
    all_rms = np.array([r["rms_tilt_deg"] for r in site_records])

    if len(all_rms) > 0:
        edges = np.linspace(
            all_rms.min(),
            all_rms.max() if all_rms.max() > all_rms.min()
            else all_rms.min() + 1.0,
            args.bins + 1,
        )
    else:
        edges = args.bins

    fig, ax = plt.subplots(figsize=(8, 5.5))

    for metal, records in [("Pb", pb_sites), ("Ca", ca_sites)]:
        if not records:
            continue

        values = np.array([r["rms_tilt_deg"] for r in records])

        ax.hist(
            values,
            bins=edges,
            alpha=0.55,
            density=False,
            label=metal,
        )

    ax.set_xlabel("RMS octahedral tilt relative to fixed axes (degrees)")
    ax.set_ylabel("Number of octahedra")
    ax.set_title("Absolute octahedral tilts")
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        stem.parent / f"{stem.name}_absolute_octahedral_tilts.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    # ------------------------------------------------------------
    # Histograms for x, y, z tilts
    # ------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(8, 12))

    for ax, ref_axis in zip(axes, ["x", "y", "z"]):
        values_by_metal = {}

        for metal in ["Pb", "Ca"]:
            values = [
                r["tilt_deg"]
                for r in axis_records
                if r["metal"] == metal
                and r["reference_axis"] == ref_axis
            ]

            if values:
                values_by_metal[metal] = np.asarray(values)

        if not values_by_metal:
            continue

        all_values = np.concatenate(list(values_by_metal.values()))

        if all_values.max() > all_values.min():
            hist_edges = np.linspace(
                all_values.min(),
                all_values.max(),
                args.bins + 1,
            )
        else:
            hist_edges = np.linspace(
                max(0.0, all_values.min() - 0.5),
                all_values.min() + 0.5,
                args.bins + 1,
            )

        for metal, values in values_by_metal.items():
            ax.hist(
                values,
                bins=hist_edges,
                alpha=0.55,
                density=False,
                label=metal,
            )

        ax.set_xlabel(
            f"Absolute tilt relative to {ref_axis}-axis (degrees)"
        )
        ax.set_ylabel("Number of octahedra")
        ax.set_title(f"{ref_axis}-axis tilt")
        ax.legend()

    fig.tight_layout()
    fig.savefig(
        stem.parent / f"{stem.name}_absolute_axis_tilts.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    # RMS tilt by site
    fig, ax = plt.subplots(figsize=(10, 5.5))

    indices = np.arange(len(site_records))
    rms = np.array([r["rms_tilt_deg"] for r in site_records])

    pb_mask = np.array([r["metal"] == "Pb" for r in site_records])
    ca_mask = np.array([r["metal"] == "Ca" for r in site_records])

    if np.any(pb_mask):
        ax.scatter(
            indices[pb_mask],
            rms[pb_mask],
            marker="o",
            label="Pb",
        )

    if np.any(ca_mask):
        ax.scatter(
            indices[ca_mask],
            rms[ca_mask],
            marker="s",
            label="Ca",
        )

    ax.set_xlabel("Metal atom index")
    ax.set_ylabel("RMS absolute octahedral tilt (degrees)")
    ax.set_title("Absolute octahedral tilt by site")
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        stem.parent / f"{stem.name}_absolute_octahedral_tilt_by_site.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    print("\nAbsolute octahedral tilt analysis")
    print("---------------------------------")
    print(f"Complete octahedra: {len(site_records)}")
    print(f"Pb octahedra:       {len(pb_sites)}")
    print(f"Ca octahedra:       {len(ca_sites)}")
    print(f"M-Br cutoff:        {args.cutoff:.3f} A")
    print()
    print("Reference axes:")
    print("  x = (1, 0, 0)")
    print("  y = (0, 1, 0)")
    print("  z = (0, 0, 1)")
    print()

    for metal, records in [("Pb", pb_sites), ("Ca", ca_sites)]:
        if not records:
            continue

        print(f"{metal}:")
        print(
            f"  mean RMS tilt = "
            f"{np.mean([r['rms_tilt_deg'] for r in records]):.3f} deg"
        )
        print(
            f"  median RMS tilt = "
            f"{np.median([r['rms_tilt_deg'] for r in records]):.3f} deg"
        )

        for axis in ["x", "y", "z"]:
            values = [
                r[f"tilt_{axis}_deg"]
                for r in records
            ]
            print(
                f"  mean {axis}-tilt = "
                f"{np.mean(values):.3f} deg"
            )

        print()

    print("Outputs:")
    print(f"  {site_csv}")
    print(f"  {axis_csv}")
    print(f"  {stem.name}_absolute_octahedral_tilts.png")
    print(f"  {stem.name}_absolute_axis_tilts.png")
    print(f"  {stem.name}_absolute_octahedral_tilt_by_site.png")


if __name__ == "__main__":
    main()

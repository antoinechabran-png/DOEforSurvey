import io
import math
import random
from collections import Counter
from itertools import combinations, product

import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# Utility functions
# ============================================================

def generate_random_codes(n, seed=None):
    """Generate unique 3-character sensory codes such as B53 or C12."""
    rng = random.Random(seed)
    letters = "ABCDEFGHJKLMNPQRSTUVWXYZ"  # I and O removed to avoid confusion
    codes = []
    while len(codes) < n:
        code = f"{rng.choice(letters)}{rng.randint(10, 99)}"
        if code not in codes:
            codes.append(code)
    return codes


def to_excel_sheets(sheets):
    """Convert a dict of {sheet_name: dataframe} to an Excel workbook in memory."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        for sheet_name, df in sheets.items():
            safe_name = str(sheet_name)[:31]
            df.to_excel(writer, index=False, sheet_name=safe_name)
    return output.getvalue()


def largest_remainder(total, proportions):
    """Allocate an integer total according to proportions using largest remainder."""
    proportions = np.asarray(proportions, dtype=float)
    if proportions.sum() <= 0:
        raise ValueError("Quota proportions must sum to more than 0.")
    proportions = proportions / proportions.sum()
    raw = proportions * int(total)
    base = np.floor(raw).astype(int)
    remainder = int(total) - int(base.sum())
    if remainder > 0:
        order = np.argsort(-(raw - base))
        for idx in order[:remainder]:
            base[idx] += 1
    return base.tolist()


def parse_percentages(text):
    vals = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not vals:
        return []
    total = sum(vals)
    if total <= 0:
        return []
    return [v / total for v in vals]


def product_position_summary(rotation_df, products):
    """Summarize how often each product appears overall and in each serving position."""
    pos_cols = [c for c in rotation_df.columns if c.startswith("Position") or c.startswith("Rank") or c.startswith("Sample")]
    rows = []
    for p in products:
        row = {"Product": p}
        total = 0
        for c in pos_cols:
            n = int((rotation_df[c] == p).sum())
            row[c] = n
            total += n
        row["Total appearances"] = total
        rows.append(row)
    cols = ["Product", "Total appearances"] + pos_cols
    return pd.DataFrame(rows)[cols]


# ============================================================
# Rotation design logic
# ============================================================

def generate_complete_block(num_assessors, products, seed=1234):
    """Each assessor sees every product, with orders balanced heuristically."""
    rng = random.Random(seed)
    v = len(products)
    position_counts = {p: [0] * v for p in products}
    data = []

    for i in range(num_assessors):
        best_order = None
        best_score = float("inf")
        candidates = [rng.sample(products, v) for _ in range(min(300, max(50, v * 30)))]
        for order in candidates:
            score = 0.0
            for pos, p in enumerate(order):
                score += (position_counts[p][pos] + 1) ** 2
            if score < best_score:
                best_score = score
                best_order = order
        for pos, p in enumerate(best_order):
            position_counts[p][pos] += 1
        data.append([f"Assessor {i + 1}"] + best_order)

    cols = ["Assessor"] + [f"Position {j + 1}" for j in range(v)]
    return pd.DataFrame(data, columns=cols)




def _find_cyclic_difference_set(v, k, lam, max_search=250000):
    """
    Search for a cyclic (v, k, lambda) difference set in Z_v.
    Fixing 0 in the starter block removes equivalent translations and makes
    common sensory-sized symmetric BIBDs (e.g. 11 products, 6 per block) fast.
    """
    if v < 2 or k < 2 or not float(lam).is_integer():
        return None
    lam = int(lam)
    if k * (k - 1) != lam * (v - 1):
        return None

    search_space = math.comb(v - 1, k - 1)
    if search_space > max_search:
        return None

    for rest in combinations(range(1, v), k - 1):
        block = (0,) + rest
        diffs = [0] * v
        for a in block:
            for b in block:
                if a != b:
                    diffs[(a - b) % v] += 1
        if all(diffs[d] == lam for d in range(1, v)):
            return block
    return None

def _candidate_blocks(v, k, rng, max_candidates=800):
    """Return candidate subsets for a balanced incomplete block search."""
    try:
        total_combos = math.comb(v, k)
    except ValueError:
        total_combos = max_candidates + 1

    if total_combos <= max_candidates:
        return [tuple(c) for c in combinations(range(v), k)]

    found = set()
    attempts = 0
    while len(found) < max_candidates and attempts < max_candidates * 20:
        found.add(tuple(sorted(rng.sample(range(v), k))))
        attempts += 1
    return list(found)


def _order_blocks_balanced(blocks, products, seed=1234):
    """Choose serving orders that balance products by position and first-order carryover."""
    rng = random.Random(seed)
    v = len(products)
    k = len(blocks[0]) if blocks else 0
    pos_counts = np.zeros((v, k), dtype=int)
    carry = np.zeros((v, v), dtype=int)
    ordered = []

    for block_index, block in enumerate(blocks):
        best = None
        best_score = float("inf")

        n_try = 500 if k <= 7 else 250
        seen = set()
        for _ in range(n_try):
            perm = tuple(rng.sample(list(block), k))
            if perm in seen:
                continue
            seen.add(perm)

            temp_pos = pos_counts.copy()
            temp_carry = carry.copy()
            for pos, p in enumerate(perm):
                temp_pos[p, pos] += 1
            for a, b in zip(perm[:-1], perm[1:]):
                temp_carry[a, b] += 1

            pos_target = (block_index + 1) / max(v, 1)
            carry_target = ((block_index + 1) * max(k - 1, 0)) / max(v * (v - 1), 1)
            score = np.sum((temp_pos - pos_target) ** 2)
            if v > 1 and k > 1:
                mask = ~np.eye(v, dtype=bool)
                score += 0.20 * np.sum((temp_carry[mask] - carry_target) ** 2)

            if score < best_score:
                best_score = score
                best = perm

        if best is None:
            best = tuple(block)

        for pos, p in enumerate(best):
            pos_counts[p, pos] += 1
        for a, b in zip(best[:-1], best[1:]):
            carry[a, b] += 1
        ordered.append(best)

    rows = []
    for i, block in enumerate(ordered, 1):
        rows.append([f"Rotation {i}"] + [products[idx] for idx in block])
    cols = ["Rotation"] + [f"Position {j + 1}" for j in range(k)]
    return pd.DataFrame(rows, columns=cols)


def generate_balanced_ibd(total_products, products_per_assessor, target_assessors_per_product, products, seed=1234):
    """
    Generate a balanced/near-balanced incomplete block design.

    If classical BIBD parameter conditions are feasible, the heuristic attempts to
    achieve equal replications and pair frequencies. Otherwise it produces a
    near-balanced design and reports achieved balance.
    """
    v = total_products
    k = products_per_assessor
    r = target_assessors_per_product
    rng = random.Random(seed)

    total_evals = v * r
    b = int(math.ceil(total_evals / k))
    target_rep = b * k / v
    target_pair = b * k * (k - 1) / (v * (v - 1)) if v > 1 else 0

    rep_counts = np.zeros(v, dtype=int)
    pair_counts = np.zeros((v, v), dtype=int)
    chosen_blocks = []
    chosen_counter = Counter()

    # For symmetric cases (b=v and r=k), first try an exact cyclic difference-set
    # construction. This covers useful designs such as v=11, k=6, r=6, lambda=3.
    lambda_theoretical = (r * (k - 1) / (v - 1)) if v > 1 else 0
    starter = None
    if b == v and r == k and float(lambda_theoretical).is_integer():
        starter = _find_cyclic_difference_set(v, k, int(lambda_theoretical))

    if starter is not None:
        chosen_blocks = [tuple(sorted(((x + shift) % v for x in starter))) for shift in range(v)]
        for block in chosen_blocks:
            for p in block:
                rep_counts[p] += 1
            for a, c in combinations(block, 2):
                pair_counts[a, c] += 1
                pair_counts[c, a] += 1
    else:
        candidates = _candidate_blocks(v, k, rng)

        for block_no in range(b):
            # Add some fresh random candidates at each step for larger spaces.
            step_candidates = list(candidates)
            if len(step_candidates) < 800:
                for _ in range(100):
                    step_candidates.append(tuple(sorted(rng.sample(range(v), k))))

            best_score = float("inf")
            best_block = None
            progress = block_no + 1
            rep_goal = progress * k / v
            pair_goal = progress * k * (k - 1) / (v * (v - 1)) if v > 1 else 0

            rng.shuffle(step_candidates)
            for block in step_candidates:
                temp_rep = rep_counts.copy()
                temp_pair = pair_counts.copy()
                for p in block:
                    temp_rep[p] += 1
                for a, c in combinations(block, 2):
                    temp_pair[a, c] += 1
                    temp_pair[c, a] += 1

                rep_score = np.sum((temp_rep - rep_goal) ** 2)
                if v > 1:
                    upper = temp_pair[np.triu_indices(v, 1)]
                    pair_score = np.sum((upper - pair_goal) ** 2)
                else:
                    pair_score = 0

                repeat_penalty = 2.5 * chosen_counter[tuple(sorted(block))]
                end_rep_penalty = 0.15 * np.sum((temp_rep - target_rep) ** 2)
                score = 4.0 * rep_score + pair_score + repeat_penalty + end_rep_penalty

                if score < best_score:
                    best_score = score
                    best_block = tuple(block)

            chosen_blocks.append(best_block)
            chosen_counter[tuple(sorted(best_block))] += 1
            for p in best_block:
                rep_counts[p] += 1
            for a, c in combinations(best_block, 2):
                pair_counts[a, c] += 1
                pair_counts[c, a] += 1
    rotation_df = _order_blocks_balanced(chosen_blocks, products, seed=seed + 1)

    # Report design diagnostics.
    rep_values = rep_counts.tolist()
    pair_values = pair_counts[np.triu_indices(v, 1)].tolist() if v > 1 else []
    exact_param_feasible = (
        (v * r) % k == 0
        and ((r * (k - 1)) % (v - 1) == 0 if v > 1 else True)
    )
    exact_achieved = (
        all(x == r for x in rep_values)
        and (len(set(pair_values)) <= 1 if pair_values else True)
    )

    diagnostics = {
        "v_products": v,
        "k_per_block": k,
        "requested_r_per_product": r,
        "generated_blocks": b,
        "classical_BIBD_parameters_feasible": exact_param_feasible,
        "theoretical_lambda": lambda_theoretical,
        "exact_balance_achieved": exact_achieved,
        "min_product_replication": min(rep_values),
        "max_product_replication": max(rep_values),
        "min_pair_frequency": min(pair_values) if pair_values else 0,
        "max_pair_frequency": max(pair_values) if pair_values else 0,
    }
    return rotation_df, diagnostics


def generate_cyclic_ibd(total_products, products_per_rotation, num_rotations, products, seed=1234):
    """Generate cyclic incomplete blocks by shifting a base window around the product list."""
    v = total_products
    k = products_per_rotation
    blocks = []
    for r in range(num_rotations):
        start = r % v
        blocks.append(tuple((start + j) % v for j in range(k)))
    return _order_blocks_balanced(blocks, products, seed=seed)


def generate_williams_design(products):
    """
    Generate a Williams / balanced Latin Square order design.
    For odd n, reverse sequences are added to improve first-order carryover balance.
    """
    n = len(products)
    if n < 2:
        return pd.DataFrame()

    # Standard balanced Latin square first row: 0,1,n-1,2,n-2,...
    first = []
    low, high = 0, n - 1
    take_low = True
    while low <= high:
        if take_low:
            first.append(low)
            low += 1
        else:
            first.append(high)
            high -= 1
        take_low = not take_low

    sequences = []
    for shift in range(n):
        sequences.append(tuple((x + shift) % n for x in first))

    if n % 2 == 1:
        sequences += [tuple(reversed(seq)) for seq in sequences]

    rows = []
    for i, seq in enumerate(sequences, 1):
        rows.append([f"Rotation {i}"] + [products[idx] for idx in seq])
    cols = ["Rotation"] + [f"Position {j + 1}" for j in range(n)]
    return pd.DataFrame(rows, columns=cols)


def generate_triangular(num_assessors, products, seed=1234):
    """Generate AB-A / BA-B triangle-test triads."""
    rng = random.Random(seed)
    pairs = list(combinations(products, 2))
    data = []
    for i in range(1, num_assessors + 1):
        p1, p2 = rng.choice(pairs)
        triad = [p1, p1, p2] if rng.random() > 0.5 else [p2, p2, p1]
        rng.shuffle(triad)
        data.append([f"Assessor {i}"] + triad)
    return pd.DataFrame(data, columns=["Assessor", "Sample 1", "Sample 2", "Sample 3"])


# ============================================================
# Quota-locked logic
# ============================================================


def calculate_quota_locked_rotation_requirements(
    total_products,
    products_per_respondent,
    target_respondents_per_product,
    respondents_per_rotation,
):
    """
    Calculate the minimum whole-number of rotations needed when every rotation
    is assigned the same number of respondents.

    A product included in one rotation receives `respondents_per_rotation`
    evaluations. Therefore each product must appear in at least
    ceil(target_respondents_per_product / respondents_per_rotation) rotations.
    The total number of required product-in-rotation appearances is then
    total_products * required_appearances_per_product, and each rotation provides
    `products_per_respondent` such appearances.
    """
    v = int(total_products)
    k = int(products_per_respondent)
    target = int(target_respondents_per_product)
    per_rotation = int(respondents_per_rotation)

    if v < 1 or k < 1 or per_rotation < 1 or target < 1:
        raise ValueError("All rotation-planning inputs must be greater than 0.")

    required_product_rotations = int(math.ceil(target / per_rotation))
    num_rotations = int(math.ceil(v * required_product_rotations / k))
    total_respondents = num_rotations * per_rotation
    total_product_evaluations = total_respondents * k
    average_evaluations_per_product = total_product_evaluations / v
    minimum_batch_exposure = required_product_rotations * per_rotation

    return {
        "required_product_rotations": required_product_rotations,
        "num_rotations": num_rotations,
        "total_respondents": total_respondents,
        "total_product_evaluations": total_product_evaluations,
        "average_evaluations_per_product": average_evaluations_per_product,
        "minimum_batch_exposure": minimum_batch_exposure,
        "exact_target_multiple_of_rotation_size": target % per_rotation == 0,
    }

def allocate_quota_by_rotation(num_rotations, respondents_per_rotation, proportions):
    """
    Return a [rotation x category] integer allocation.
    It locks each rotation as closely as possible to target proportions while
    matching the overall target exactly after integer rounding.
    """
    p = np.asarray(proportions, dtype=float)
    p = p / p.sum()
    m = int(respondents_per_rotation)
    n_rot = int(num_rotations)

    global_counts = np.array(largest_remainder(n_rot * m, p), dtype=int)
    base = np.floor(m * p).astype(int)
    allocations = np.tile(base, (n_rot, 1))
    remaining_global = global_counts - allocations.sum(axis=0)
    extras_per_rotation = m - int(base.sum())

    for r in range(n_rot):
        chosen = set()
        for _ in range(extras_per_rotation):
            eligible = [j for j in range(len(p)) if remaining_global[j] > 0 and j not in chosen]
            if not eligible:
                eligible = [j for j in range(len(p)) if remaining_global[j] > 0]
            if not eligible:
                break
            # Need first, then target fractional remainder, then rotate ties by rotation.
            frac = m * p - base
            j = max(
                eligible,
                key=lambda x: (remaining_global[x], frac[x], -((x - r) % max(len(p), 1))),
            )
            allocations[r, j] += 1
            remaining_global[j] -= 1
            chosen.add(j)

    # Safety correction if any remainder remains after the pass.
    while remaining_global.sum() > 0:
        j = int(np.argmax(remaining_global))
        for r in range(n_rot):
            if allocations[r].sum() < m:
                allocations[r, j] += 1
                remaining_global[j] -= 1
                if remaining_global[j] <= 0:
                    break
        else:
            break

    return allocations, global_counts


def _joint_probability(profile, target_defs):
    prob = 1.0
    for idx, cat in enumerate(profile):
        cats = target_defs[idx]["categories"]
        props = target_defs[idx]["proportions"]
        prob *= props[cats.index(cat)]
    return prob


def build_quota_slots(num_rotations, respondents_per_rotation, target_defs, seed=1234, balance_intersections=True):
    """
    Build respondent slots with marginal quotas locked by rotation.
    When requested, a heuristic also spreads cross-target combinations evenly.
    """
    rng = random.Random(seed)
    rows = []
    for r in range(num_rotations):
        for s in range(respondents_per_rotation):
            rows.append({"Rotation": f"Rotation {r + 1}", "Slot": s + 1})
    slots = pd.DataFrame(rows)

    quota_summary = pd.DataFrame({"Rotation": [f"Rotation {i + 1}" for i in range(num_rotations)]})

    for d, target in enumerate(target_defs):
        name = target["name"]
        cats = target["categories"]
        props = target["proportions"]
        allocations, global_counts = allocate_quota_by_rotation(
            num_rotations, respondents_per_rotation, props
        )

        for j, cat in enumerate(cats):
            quota_summary[f"{name} | {cat}"] = allocations[:, j]

        slots[name] = None
        global_profile_counts = Counter()
        assigned_so_far = 0

        for r in range(num_rotations):
            idxs = slots.index[slots["Rotation"] == f"Rotation {r + 1}"].tolist()
            labels = []
            for j, cat in enumerate(cats):
                labels.extend([cat] * int(allocations[r, j]))

            if d == 0 or not balance_intersections:
                # Rotate before shuffling so ties do not always fall in the same slot pattern.
                if labels:
                    offset = r % len(labels)
                    labels = labels[offset:] + labels[:offset]
                rng.shuffle(labels)
                chosen_labels = labels
            else:
                best_labels = None
                best_score = float("inf")
                attempts = min(250, max(60, len(labels) * 12))
                for _ in range(attempts):
                    cand = labels[:]
                    rng.shuffle(cand)
                    temp = global_profile_counts.copy()
                    for idx, cat in zip(idxs, cand):
                        prev = tuple(slots.loc[idx, t["name"]] for t in target_defs[:d])
                        temp[prev + (cat,)] += 1
                    total_now = assigned_so_far + len(idxs)

                    score = 0.0
                    all_profiles = product(*[t["categories"] for t in target_defs[: d + 1]])
                    for prof in all_profiles:
                        expected = total_now * _joint_probability(prof, target_defs[: d + 1])
                        observed = temp[tuple(prof)]
                        score += (observed - expected) ** 2 / (expected + 0.5)
                    if score < best_score:
                        best_score = score
                        best_labels = cand
                chosen_labels = best_labels if best_labels is not None else labels

            for idx, cat in zip(idxs, chosen_labels):
                slots.loc[idx, name] = cat
                if d > 0:
                    prev = tuple(slots.loc[idx, t["name"]] for t in target_defs[:d])
                    global_profile_counts[prev + (cat,)] += 1
            assigned_so_far += len(idxs)

    # Full intersection summary, useful for recruitment monitoring.
    if target_defs:
        names = [t["name"] for t in target_defs]
        intersection = (
            slots.groupby(["Rotation"] + names, dropna=False)
            .size()
            .reset_index(name="Target")
            .sort_values(["Rotation"] + names)
        )
    else:
        intersection = pd.DataFrame()

    return slots, quota_summary, intersection


def quota_target_table(target_defs, total_n):
    rows = []
    for target in target_defs:
        counts = largest_remainder(total_n, target["proportions"])
        for cat, prop, count in zip(target["categories"], target["proportions"], counts):
            rows.append({
                "Target": target["name"],
                "Category": cat,
                "Requested %": round(prop * 100, 2),
                "Overall target N": count,
            })
    return pd.DataFrame(rows)


# ============================================================
# Base quantity calculator
# ============================================================

BOTTLE_SIZES_ML = [50, 75, 100, 250, 350, 450, 500, 600, 650, 800, 900, 1000, 1200]
ML_PER_US_FL_OZ = 29.5735295625


def quantity_to_ml(quantity, unit, density_kg_per_l=1.0):
    if unit == "mL":
        return quantity
    if unit == "US fl oz":
        return quantity * ML_PER_US_FL_OZ
    if unit == "kg":
        if density_kg_per_l <= 0:
            raise ValueError("Density must be greater than 0.")
        return (quantity / density_kg_per_l) * 1000.0
    raise ValueError("Unsupported unit")


def ml_to_quantity(quantity_ml, unit, density_kg_per_l=1.0):
    """Convert an mL-equivalent quantity back to the selected calculator unit."""
    if unit == "mL":
        return quantity_ml
    if unit == "US fl oz":
        return quantity_ml / ML_PER_US_FL_OZ
    if unit == "kg":
        if density_kg_per_l <= 0:
            raise ValueError("Density must be greater than 0.")
        return (quantity_ml / 1000.0) * density_kg_per_l
    raise ValueError("Unsupported unit")


def calculate_base_quantities(
    num_candidates,
    num_benchmarks,
    samples_per_candidate,
    samples_per_benchmark,
    fill_per_sample,
    unit,
    loss_margin_pct,
    bottle_size_ml,
    density_kg_per_l=1.0,
):
    loss_factor = 1 + loss_margin_pct / 100.0

    per_candidate_base = samples_per_candidate * fill_per_sample
    per_benchmark_base = samples_per_benchmark * fill_per_sample

    per_candidate_with_loss = per_candidate_base * loss_factor
    per_benchmark_with_loss = per_benchmark_base * loss_factor

    total_candidate_base = num_candidates * per_candidate_base
    total_benchmark_base = num_benchmarks * per_benchmark_base
    total_candidate_with_loss = num_candidates * per_candidate_with_loss
    total_benchmark_with_loss = num_benchmarks * per_benchmark_with_loss

    benchmark_ml = quantity_to_ml(per_benchmark_with_loss, unit, density_kg_per_l)
    bottles_per_benchmark = math.ceil(benchmark_ml / bottle_size_ml) if num_benchmarks > 0 else 0
    total_bottles = bottles_per_benchmark * num_benchmarks

    return {
        "per_candidate_base": per_candidate_base,
        "per_candidate_with_loss": per_candidate_with_loss,
        "total_candidate_base": total_candidate_base,
        "total_candidate_with_loss": total_candidate_with_loss,
        "per_benchmark_base": per_benchmark_base,
        "per_benchmark_with_loss": per_benchmark_with_loss,
        "total_benchmark_base": total_benchmark_base,
        "total_benchmark_with_loss": total_benchmark_with_loss,
        "bottles_per_benchmark": bottles_per_benchmark,
        "total_bottles": total_bottles,
        "benchmark_ml_with_loss": benchmark_ml,
    }


# ============================================================
# Streamlit UI
# ============================================================

st.set_page_config(page_title="CMI EU Test Planing Tool Kit", page_icon="🧪", layout="wide")
st.title("🧪 CMI EU Test Planing Tool Kit")

st.sidebar.header("Menu")
app_page = st.sidebar.radio(
    "Choose a tool",
    ["Base Quantity Calculator", "Rotation Planner"],
)


if app_page == "Rotation Planner":
    st.sidebar.subheader("Rotation settings")
    design_type = st.sidebar.selectbox(
        "Select design type",
        [
            "BIBD / Balanced Incomplete Block Design",
            "Quota-Locked Incomplete Block Design",
            "Cyclic Incomplete Block Design",
            "Williams / Balanced Latin Square",
            "Complete Block Design",
            "Triangular Design",
        ],
    )

    seed = int(st.sidebar.number_input("Random seed", min_value=0, value=1234, step=1))

    st.subheader("1. Product Configuration")
    incomplete_design = design_type in {
        "BIBD / Balanced Incomplete Block Design",
        "Quota-Locked Incomplete Block Design",
        "Cyclic Incomplete Block Design",
    }

    default_products = 11 if incomplete_design else 4
    num_prods = int(st.number_input("Number of products", min_value=2, value=default_products, step=1))

    input_method = st.radio(
        "Product code method",
        ["Random Generation", "Manual Entry / Excel Upload"],
        horizontal=True,
    )

    if input_method == "Random Generation":
        code_key = f"generated_codes_{num_prods}"
        if code_key not in st.session_state:
            st.session_state[code_key] = generate_random_codes(num_prods, seed=seed)
        if st.button("Refresh random codes"):
            st.session_state[code_key] = generate_random_codes(num_prods, seed=random.randint(0, 10**9))
        product_list = st.session_state[code_key]
        st.info(f"Generated codes: {', '.join(product_list)}")
    else:
        col1, col2 = st.columns(2)
        with col1:
            default_manual = ", ".join([f"P{i+1}" for i in range(num_prods)])
            manual_input = st.text_area("Enter codes (comma separated)", default_manual)
        with col2:
            uploaded_file = st.file_uploader("Or upload Excel (codes in column A)", type=["xlsx"])

        if uploaded_file:
            df_upload = pd.read_excel(uploaded_file)
            product_list = df_upload.iloc[:, 0].dropna().astype(str).tolist()
        else:
            product_list = [x.strip() for x in manual_input.split(",") if x.strip()]

        if len(product_list) != num_prods:
            st.warning(f"Expected {num_prods} products, but found {len(product_list)} codes.")

    valid_products = len(product_list) == num_prods and len(set(product_list)) == len(product_list)
    if not valid_products:
        st.error("Please provide exactly the requested number of unique product codes before generating a plan.")

    st.subheader("2. Design Parameters")
    generated_sheets = None

    if design_type == "BIBD / Balanced Incomplete Block Design":
        k = int(st.number_input(
            "Number of products tested per assessor",
            min_value=2,
            max_value=max(2, num_prods - 1),
            value=min(6, max(2, num_prods - 1)),
        ))
        r = int(st.number_input("Desired assessors per product (r)", min_value=1, value=6, step=1))

        b_exact = num_prods * r / k
        lambda_value = r * (k - 1) / (num_prods - 1)
        feasible = b_exact.is_integer() and float(lambda_value).is_integer()
        if feasible:
            st.success(
                f"Classical BIBD parameter check passes: b={int(b_exact)} blocks and λ={int(lambda_value)}. "
                "The generator will try to realize exact balance."
            )
        else:
            st.info(
                f"An exact classical BIBD is not available for these requested parameters "
                f"(b={b_exact:.2f}, λ={lambda_value:.2f}). A near-balanced IBD will be generated."
            )

        if st.button("Generate BIBD / Balanced IBD", type="primary", disabled=not valid_products):
            rotation_df, diagnostics = generate_balanced_ibd(num_prods, k, r, product_list, seed)
            diag_df = pd.DataFrame([diagnostics])
            exposure_df = product_position_summary(rotation_df, product_list)
            generated_sheets = {
                "Rotation Plan": rotation_df,
                "Product Balance": exposure_df,
                "Diagnostics": diag_df,
            }

    elif design_type == "Quota-Locked Incomplete Block Design":
        left, right = st.columns(2)
        with left:
            k = int(st.number_input(
                "Products tested per respondent",
                min_value=2,
                max_value=max(2, num_prods - 1),
                value=min(6, max(2, num_prods - 1)),
            ))
            target_respondents_per_product = int(st.number_input(
                "Desired respondents per product / code",
                min_value=1,
                value=108,
                step=1,
                help=(
                    "Target number of respondent evaluations wanted for each product/code. "
                    "The app can use this to calculate the required number of rotations."
                ),
            ))
        with right:
            respondents_per_rotation = int(st.number_input(
                "Respondents per rotation", min_value=1, value=18, step=1
            ))
            rotation_engine = st.selectbox(
                "Incomplete-block engine",
                ["Optimized balanced", "Cyclic"],
            )

        rotation_count_mode = st.radio(
            "How should the number of rotations be determined?",
            [
                "Calculate automatically from respondents per product",
                "Set number of rotations manually",
            ],
            horizontal=True,
        )

        auto_requirements = calculate_quota_locked_rotation_requirements(
            num_prods,
            k,
            target_respondents_per_product,
            respondents_per_rotation,
        )

        if rotation_count_mode == "Calculate automatically from respondents per product":
            num_rotations = auto_requirements["num_rotations"]
            required_product_rotations = auto_requirements["required_product_rotations"]

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Calculated rotations", num_rotations)
            m2.metric("Product appearances needed", required_product_rotations)
            m3.metric("Planned respondents", auto_requirements["total_respondents"])
            m4.metric(
                "Avg. evaluations / product",
                f'{auto_requirements["average_evaluations_per_product"]:.1f}',
            )

            st.caption(
                f"Calculation: each product must be present in at least "
                f"ceil({target_respondents_per_product} / {respondents_per_rotation}) = "
                f"{required_product_rotations} rotations. With {num_prods} products and "
                f"{k} products per rotation, the minimum is "
                f"ceil({num_prods} × {required_product_rotations} / {k}) = {num_rotations} rotations."
            )

            if not auto_requirements["exact_target_multiple_of_rotation_size"]:
                st.warning(
                    f"{target_respondents_per_product} is not a multiple of "
                    f"{respondents_per_rotation}. Because respondents are assigned in whole "
                    f"rotation groups, a product appearing {required_product_rotations} times "
                    f"would receive {auto_requirements['minimum_batch_exposure']} evaluations. "
                    "The generated design therefore targets at least the requested base rather than "
                    "an exact count for every product."
                )
        else:
            num_rotations = int(st.number_input(
                "Number of rotations",
                min_value=2,
                value=max(2, auto_requirements["num_rotations"]),
                step=1,
            ))
            required_product_rotations = int(math.ceil(
                target_respondents_per_product / respondents_per_rotation
            ))
            planned_avg = num_rotations * k * respondents_per_rotation / num_prods
            st.info(
                f"With {num_rotations} rotations × {respondents_per_rotation} respondents, "
                f"the design contains {num_rotations * respondents_per_rotation} respondents "
                f"and averages {planned_avg:.1f} evaluations per product."
            )
            if planned_avg < target_respondents_per_product:
                st.warning(
                    "The manually selected number of rotations is too low on average to reach "
                    "the desired respondents per product. Increase the number of rotations or "
                    "respondents per rotation."
                )

        st.markdown("#### Recruitment targets / quotas")
        st.caption(
            "Enter each target as categories plus percentages. Marginal quotas are locked by rotation as closely as integer counts allow."
        )
        num_targets = int(st.number_input("Number of quota variables", min_value=1, max_value=5, value=2, step=1))

        target_defs = []
        quota_errors = []
        defaults = [
            ("Age", "18–34, 35–55", "60, 40"),
            ("SEL", "A+B, C", "50, 50"),
            ("Gender", "Female, Male", "50, 50"),
            ("Usage", "Light, Medium, Heavy", "30, 40, 30"),
            ("Region", "North, South", "50, 50"),
        ]

        for i in range(num_targets):
            d_name, d_cats, d_pct = defaults[i]
            c1, c2, c3 = st.columns([1.1, 2.0, 1.5])
            with c1:
                name = st.text_input(f"Target {i + 1} name", d_name, key=f"target_name_{i}")
            with c2:
                cats_text = st.text_input(
                    f"{name} categories (comma separated)", d_cats, key=f"target_cats_{i}"
                )
            with c3:
                pct_text = st.text_input(
                    f"{name} target % (comma separated)", d_pct, key=f"target_pct_{i}"
                )

            cats = [x.strip() for x in cats_text.split(",") if x.strip()]
            try:
                props = parse_percentages(pct_text)
            except ValueError:
                props = []

            if not name.strip():
                quota_errors.append(f"Target {i + 1}: name cannot be blank.")
            if len(cats) < 2:
                quota_errors.append(f"{name}: enter at least two categories.")
            if len(cats) != len(props):
                quota_errors.append(f"{name}: number of categories and percentages must match.")
            if props and any(p < 0 for p in props):
                quota_errors.append(f"{name}: percentages cannot be negative.")

            if name.strip() and len(cats) >= 2 and len(cats) == len(props):
                target_defs.append({"name": name.strip(), "categories": cats, "proportions": props})

        balance_intersections = st.checkbox(
            "Also balance intersections between targets (heuristic)",
            value=True,
            help="For example Age × Gender × SEL. Marginal quotas remain the primary lock.",
        )

        total_n = num_rotations * respondents_per_rotation
        total_evaluations = total_n * k
        st.info(
            f"Planned sample size: {num_rotations} rotations × {respondents_per_rotation} respondents "
            f"= {total_n} respondents, producing {total_evaluations} product evaluations in total."
        )

        if quota_errors:
            for err in quota_errors:
                st.error(err)

        if st.button(
            "Generate Quota-Locked Plan",
            type="primary",
            disabled=(not valid_products or bool(quota_errors) or len(target_defs) != num_targets),
        ):
            if rotation_engine == "Cyclic":
                rotation_df = generate_cyclic_ibd(num_prods, k, num_rotations, product_list, seed)
            else:
                if rotation_count_mode == "Calculate automatically from respondents per product":
                    # In automatic mode the target replication in rotation blocks is known:
                    # one product appearance supplies respondents_per_rotation evaluations.
                    rotation_df, _ = generate_balanced_ibd(
                        num_prods,
                        k,
                        required_product_rotations,
                        product_list,
                        seed,
                    )
                else:
                    # In manual mode, choose a balanced replication close to the requested
                    # number of rotations, then trim/extend to the manual rotation count.
                    approx_r = max(1, round(num_rotations * k / num_prods))
                    rotation_df, _ = generate_balanced_ibd(num_prods, k, approx_r, product_list, seed)
                    if len(rotation_df) > num_rotations:
                        rotation_df = rotation_df.iloc[:num_rotations].copy()
                    elif len(rotation_df) < num_rotations:
                        extra = generate_cyclic_ibd(
                            num_prods, k, num_rotations - len(rotation_df), product_list, seed + 99
                        )
                        extra["Rotation"] = [f"Rotation {len(rotation_df) + i + 1}" for i in range(len(extra))]
                        rotation_df = pd.concat([rotation_df, extra], ignore_index=True)
                    rotation_df["Rotation"] = [f"Rotation {i + 1}" for i in range(len(rotation_df))]

            slots_df, quota_summary_df, intersection_df = build_quota_slots(
                num_rotations,
                respondents_per_rotation,
                target_defs,
                seed=seed,
                balance_intersections=balance_intersections,
            )

            # Attach product sequence to every recruitment slot.
            sequence_map = rotation_df.set_index("Rotation").to_dict(orient="index")
            assignment_df = slots_df.copy()
            for pos in [c for c in rotation_df.columns if c.startswith("Position")]:
                assignment_df[pos] = assignment_df["Rotation"].map(
                    {r: vals[pos] for r, vals in sequence_map.items()}
                )

            exposure = product_position_summary(rotation_df, product_list)
            # Respondent-weighted product exposure: each rotation is used respondents_per_rotation times.
            exposure["Expected respondent evaluations"] = (
                exposure["Total appearances"] * respondents_per_rotation
            )
            exposure["Requested respondents per product"] = target_respondents_per_product
            exposure["Difference vs requested"] = (
                exposure["Expected respondent evaluations"] - target_respondents_per_product
            )
            exposure["Target reached"] = exposure["Expected respondent evaluations"] >= target_respondents_per_product

            achieved_min = int(exposure["Expected respondent evaluations"].min())
            achieved_max = int(exposure["Expected respondent evaluations"].max())
            target_met_all = bool(exposure["Target reached"].all())

            design_summary = pd.DataFrame([{
                "Number of products": num_prods,
                "Products per respondent": k,
                "Requested respondents per product": target_respondents_per_product,
                "Respondents per rotation": respondents_per_rotation,
                "Number of rotations": num_rotations,
                "Total respondents": total_n,
                "Total product evaluations": total_n * k,
                "Min expected evaluations per product": achieved_min,
                "Max expected evaluations per product": achieved_max,
                "Target reached for every product": target_met_all,
                "Rotation count mode": rotation_count_mode,
                "Rotation engine": rotation_engine,
            }])

            if target_met_all:
                st.success(
                    f"Product base check: every code receives at least "
                    f"{target_respondents_per_product} evaluations "
                    f"(achieved range: {achieved_min}–{achieved_max})."
                )
            else:
                st.warning(
                    f"Product base check: the generated plan does not reach the requested "
                    f"{target_respondents_per_product} evaluations for every code "
                    f"(achieved range: {achieved_min}–{achieved_max}). Try the Optimized balanced "
                    "engine, increase rotations, or increase respondents per rotation."
                )

            overall_targets = quota_target_table(target_defs, total_n)
            generated_sheets = {
                "Design Summary": design_summary,
                "Rotations": rotation_df,
                "Quota by Rotation": quota_summary_df,
                "Recruitment Slots": assignment_df,
                "Quota Intersections": intersection_df,
                "Overall Targets": overall_targets,
                "Product Exposure": exposure,
            }

    elif design_type == "Cyclic Incomplete Block Design":
        k = int(st.number_input(
            "Products tested per rotation",
            min_value=2,
            max_value=max(2, num_prods - 1),
            value=min(6, max(2, num_prods - 1)),
        ))
        num_rotations = int(st.number_input("Number of rotations", min_value=2, value=num_prods, step=1))
        if st.button("Generate Cyclic IBD", type="primary", disabled=not valid_products):
            rotation_df = generate_cyclic_ibd(num_prods, k, num_rotations, product_list, seed)
            exposure_df = product_position_summary(rotation_df, product_list)
            generated_sheets = {"Rotation Plan": rotation_df, "Product Balance": exposure_df}

    elif design_type == "Williams / Balanced Latin Square":
        st.caption(
            "Best when each respondent evaluates every product and you want serving-position and first-order carryover balance."
        )
        if st.button("Generate Williams Design", type="primary", disabled=not valid_products):
            rotation_df = generate_williams_design(product_list)
            exposure_df = product_position_summary(rotation_df, product_list)
            generated_sheets = {"Rotation Plan": rotation_df, "Product Balance": exposure_df}

    elif design_type == "Complete Block Design":
        n_assessors = int(st.number_input("Number of assessors", min_value=1, value=10, step=1))
        if st.button("Generate Complete Block Plan", type="primary", disabled=not valid_products):
            rotation_df = generate_complete_block(n_assessors, product_list, seed)
            exposure_df = product_position_summary(rotation_df, product_list)
            generated_sheets = {"Rotation Plan": rotation_df, "Product Balance": exposure_df}

    elif design_type == "Triangular Design":
        n_assessors = int(st.number_input("Number of assessors", min_value=1, value=10, step=1))
        if num_prods < 2:
            st.error("Triangle testing requires at least two products.")
        if st.button("Generate Triangle Plan", type="primary", disabled=not valid_products):
            rotation_df = generate_triangular(n_assessors, product_list, seed)
            generated_sheets = {"Triangle Plan": rotation_df}

    # Results section
    if generated_sheets:
        st.subheader("3. Results")
        st.success("Plan generated successfully.")
        tabs = st.tabs(list(generated_sheets.keys()))
        for tab, (name, df) in zip(tabs, generated_sheets.items()):
            with tab:
                st.dataframe(df, use_container_width=True)

        excel_data = to_excel_sheets(generated_sheets)
        st.download_button(
            label="📥 Download Excel workbook",
            data=excel_data,
            file_name=f"sensory_plan_{design_type.lower().replace(' ', '_').replace('/', '-')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )


else:
    st.subheader("Base Quantity Calculator")
    st.caption(
        "Calculate product quantity for candidates and benchmarks, including a configurable loss / overage margin."
    )

    # Optional base naming. When several bases are entered, the output table is
    # duplicated for each base so the required quantity can be tracked separately.
    use_base_names = st.checkbox(
        "Add name of the base to be used",
        value=False,
        help="Enable this if you want the required quantity table to identify one or several bases.",
    )
    base_names = [""]
    if use_base_names:
        base_text = st.text_area(
            "Base name(s)",
            value="",
            placeholder="Example:\nBase A\nBase B",
            help="Enter one base per line, or separate several base names with commas. The calculation rows will be duplicated for each base.",
        )
        normalized = base_text.replace(",", "\n")
        parsed_bases = [x.strip() for x in normalized.splitlines() if x.strip()]
        base_names = parsed_bases if parsed_bases else [""]

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Candidates")
        num_candidates = int(st.number_input("Total number of candidates", min_value=0, value=6, step=1))
        samples_per_candidate = int(st.number_input(
            "Samples required per candidate", min_value=0, value=30, step=1
        ))

    with c2:
        st.markdown("#### Benchmarks")
        num_benchmarks = int(st.number_input("Total number of benchmarks", min_value=0, value=2, step=1))
        samples_per_benchmark = int(st.number_input(
            "Samples required per benchmark", min_value=0, value=30, step=1
        ))

    st.markdown("#### Fill quantity")
    q1, q2, q3 = st.columns(3)
    with q1:
        unit = st.selectbox("Quantity unit", ["mL", "kg", "US fl oz"], index=0)
    with q2:
        default_fill = 10.0 if unit == "mL" else (0.01 if unit == "kg" else 0.34)
        fill_per_sample = float(st.number_input(
            f"Amount needed to fill one sample ({unit})",
            min_value=0.0,
            value=default_fill,
            step=0.1 if unit != "kg" else 0.001,
            format="%.3f" if unit == "kg" else "%.2f",
        ))
    with q3:
        loss_margin_pct = float(st.number_input(
            "Loss / overage margin (%)", min_value=0.0, max_value=100.0, value=10.0, step=1.0
        ))

    density = 1.0
    if unit == "kg":
        density = float(st.number_input(
            "Product density (kg/L) — used only for bottle conversion",
            min_value=0.001,
            value=1.0,
            step=0.01,
            help="1.00 kg/L is approximately water. Change this for denser or lighter products.",
        ))

    st.markdown("#### Benchmark bottle purchase")
    bottle_size_ml = st.selectbox(
        "Benchmark bottle / pack size (mL)",
        BOTTLE_SIZES_ML,
        index=BOTTLE_SIZES_ML.index(250),
    )

    st.markdown("#### Extra samples for other stages")
    use_extra_stages = st.checkbox(
        "Add extra samples for other stages",
        value=False,
        help=(
            "Use this for additional preparation or testing stages such as dilution. "
            "The stage quantity is packaging size × number of bottles and is added to the final total."
        ),
    )

    extra_stages = []
    if use_extra_stages:
        num_extra_stages = int(st.number_input(
            "Number of extra stages",
            min_value=1,
            max_value=10,
            value=1,
            step=1,
        ))
        st.caption(
            "The loss / overage margin is already applied to candidate and benchmark quantities. "
            "Extra-stage bottle quantities are added as entered and are not given an additional margin."
        )
        for i in range(num_extra_stages):
            s1, s2, s3 = st.columns([1.6, 1.2, 1.2])
            with s1:
                stage_name = st.text_input(
                    f"Stage {i + 1} name",
                    value="Dilution" if i == 0 else f"Stage {i + 1}",
                    key=f"extra_stage_name_{i}",
                )
            with s2:
                package_size_ml = float(st.number_input(
                    f"Packaging size for stage {i + 1} (mL)",
                    min_value=0.0,
                    value=200.0,
                    step=10.0,
                    key=f"extra_stage_pack_{i}",
                ))
            with s3:
                bottles_required = int(st.number_input(
                    f"Bottles required for stage {i + 1}",
                    min_value=0,
                    value=1,
                    step=1,
                    key=f"extra_stage_bottles_{i}",
                ))

            extra_stages.append({
                "Stage": stage_name.strip() or f"Stage {i + 1}",
                "Packaging size (mL)": package_size_ml,
                "Bottles required": bottles_required,
                "Quantity (mL)": package_size_ml * bottles_required,
            })

    result = calculate_base_quantities(
        num_candidates=num_candidates,
        num_benchmarks=num_benchmarks,
        samples_per_candidate=samples_per_candidate,
        samples_per_benchmark=samples_per_benchmark,
        fill_per_sample=fill_per_sample,
        unit=unit,
        loss_margin_pct=loss_margin_pct,
        bottle_size_ml=bottle_size_ml,
        density_kg_per_l=density,
    )

    extra_stage_ml = sum(stage["Quantity (mL)"] for stage in extra_stages)
    extra_stage_quantity = ml_to_quantity(extra_stage_ml, unit, density)
    grand_total_per_base = (
        result["total_candidate_with_loss"]
        + result["total_benchmark_with_loss"]
        + extra_stage_quantity
    )
    number_of_bases = len(base_names) if use_base_names else 1
    grand_total_all_bases = grand_total_per_base * number_of_bases

    st.markdown("### Results")
    required_tab, bottles_tab = st.tabs(["Required Quantity", "Benchmark Bottles"])

    # Only quantities including the additional margin are shown for candidates/benchmarks.
    # Extra-stage quantities are exact package quantities and are added to the grand total.
    rows = []
    extra_stage_rows = []
    for base_name in base_names:
        candidate_row = {
            "Type": "Candidate",
            "Item / stage": "Candidate products",
            "Number of products": num_candidates,
            "Samples per product": samples_per_candidate,
            "Packaging size (mL)": np.nan,
            "Bottles required": np.nan,
            f"Quantity per product incl. {loss_margin_pct:.1f}% margin ({unit})": result["per_candidate_with_loss"],
            f"Total required ({unit})": result["total_candidate_with_loss"],
        }
        benchmark_row = {
            "Type": "Benchmark",
            "Item / stage": "Benchmark products",
            "Number of products": num_benchmarks,
            "Samples per product": samples_per_benchmark,
            "Packaging size (mL)": np.nan,
            "Bottles required": np.nan,
            f"Quantity per product incl. {loss_margin_pct:.1f}% margin ({unit})": result["per_benchmark_with_loss"],
            f"Total required ({unit})": result["total_benchmark_with_loss"],
        }
        if use_base_names:
            candidate_row = {"Base name": base_name or "(not specified)", **candidate_row}
            benchmark_row = {"Base name": base_name or "(not specified)", **benchmark_row}
        rows.extend([candidate_row, benchmark_row])

        for stage in extra_stages:
            stage_quantity_unit = ml_to_quantity(stage["Quantity (mL)"], unit, density)
            stage_row = {
                "Type": "Extra stage",
                "Item / stage": stage["Stage"],
                "Number of products": np.nan,
                "Samples per product": np.nan,
                "Packaging size (mL)": stage["Packaging size (mL)"],
                "Bottles required": stage["Bottles required"],
                f"Quantity per product incl. {loss_margin_pct:.1f}% margin ({unit})": np.nan,
                f"Total required ({unit})": stage_quantity_unit,
            }
            if use_base_names:
                stage_row = {"Base name": base_name or "(not specified)", **stage_row}
            rows.append(stage_row)
            extra_stage_rows.append(stage_row.copy())

        total_row = {
            "Type": "TOTAL",
            "Item / stage": "Grand total incl. extra stages" if use_extra_stages else "Grand total",
            "Number of products": np.nan,
            "Samples per product": np.nan,
            "Packaging size (mL)": np.nan,
            "Bottles required": np.nan,
            f"Quantity per product incl. {loss_margin_pct:.1f}% margin ({unit})": np.nan,
            f"Total required ({unit})": grand_total_per_base,
        }
        if use_base_names:
            total_row = {"Base name": base_name or "(not specified)", **total_row}
        rows.append(total_row)

    summary_df = pd.DataFrame(rows)

    with required_tab:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("All candidates incl. margin / base", f"{result['total_candidate_with_loss']:.2f} {unit}")
        m2.metric("All benchmarks incl. margin / base", f"{result['total_benchmark_with_loss']:.2f} {unit}")
        m3.metric("Extra stages / base", f"{extra_stage_quantity:.2f} {unit}")
        m4.metric("Grand total / base", f"{grand_total_per_base:.2f} {unit}")
        st.dataframe(summary_df, use_container_width=True)

        if use_extra_stages:
            st.caption(
                f"Extra stages add {extra_stage_ml:.1f} mL equivalent ({extra_stage_quantity:.2f} {unit}) "
                "to the candidate + benchmark total for each base."
            )
        if use_base_names and len(base_names) > 1:
            st.info(
                f"The calculation is duplicated for {len(base_names)} bases. "
                f"Grand total across all named bases: {grand_total_all_bases:.2f} {unit}."
            )

    bottle_rows = []
    if num_benchmarks > 0:
        for base_name in base_names:
            row = {
                "Bottle / pack size (mL)": bottle_size_ml,
                "Benchmark quantity incl. margin (mL equivalent)": round(result["benchmark_ml_with_loss"], 2),
                "Bottles per benchmark": result["bottles_per_benchmark"],
                "Total benchmark bottles": result["total_bottles"],
            }
            if use_base_names:
                row = {"Base name": base_name or "(not specified)", **row}
            bottle_rows.append(row)
    bottle_df = pd.DataFrame(bottle_rows)

    with bottles_tab:
        if num_benchmarks > 0:
            b1, b2 = st.columns(2)
            b1.metric(
                f"Bottles per benchmark ({bottle_size_ml} mL)",
                result["bottles_per_benchmark"],
            )
            b2.metric("Total benchmark bottles", result["total_bottles"])
            st.caption(
                f"Bottle calculation uses the benchmark quantity including the {loss_margin_pct:.1f}% margin. "
                f"Equivalent volume per benchmark: {result['benchmark_ml_with_loss']:.1f} mL."
            )
            st.dataframe(bottle_df, use_container_width=True)
        else:
            st.info("Set at least one benchmark to calculate bottles / packs to buy.")

    st.info(
        "Formula used: required quantity per candidate / benchmark = samples required × amount per sample × "
        "(1 + loss / overage margin %). Extra-stage quantity = packaging size × number of bottles; "
        "this is added directly to the final total when activated."
    )

    export_sheets = {"Required Quantity": summary_df}
    if not bottle_df.empty:
        export_sheets["Benchmark Bottles"] = bottle_df
    if extra_stage_rows:
        export_sheets["Extra Stages"] = pd.DataFrame(extra_stage_rows)
    excel_data = to_excel_sheets(export_sheets)
    st.download_button(
        label="📥 Download quantity calculation",
        data=excel_data,
        file_name="base_quantity_calculation.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


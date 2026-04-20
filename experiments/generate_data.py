import random
import torch
import sympy
import numpy as np


def generate_random_list_clauses(num_clauses, num_literals, clauses_length_range, negated=0.5):
    """
    Generates a random list of clauses for a CNF/DNF formula
    Parameters:
        num_clauses (int): Number of clauses in the formula
        num_literals (int): Number of distinct literals
        clauses_length_range (Tuple[int, int]): Min and max range for the number of literals per clause
        negated (float): Probability of negating a literal (between 0 and 1)
    Returns:
        List[List[int]]: A list of clauses, where each clause is a list of integers representing literals
    """
    formula = []
    for _ in range(num_clauses):
        clause_length = random.randint(clauses_length_range[0], clauses_length_range[1]) # Random length of the clause
        clause = random.sample(list(range(1, num_literals + 1)), clause_length)  # Select unique literals
        clause = [lit if random.random() < negated else -lit for lit in clause]  # Randomly negate literals
        formula.append(clause)
    return formula

def assign_proposition_names(list_clauses, proposition_names=None, cnf=True):
    """
    Assigns names to the propositions in the set of clauses
    Parameters:
        list_clauses (List[List[int]]): The CNF formula
        proposition_names (List[str]): List of proposition names
        cnf (bool): If True, the formula is CNF; if False, the formula is DNF
    Returns:
        List[List[str]]: The CNF formula as a list of lists of named propositions
        List[str]:  The CNF formula as a list of clauses
        str: The CNF formula as a string
    """
    if proposition_names is None:
        proposition_names = [f"P_{i+1}" for i in range(len(list_clauses[0]))]
    named_lists = []
    named_strs = []
    for clause in list_clauses:
        named_clause = []
        for literal in clause:
            if literal > 0:
                named_clause.append(proposition_names[literal - 1])
            else:
                named_clause.append(f"~{proposition_names[-literal - 1]}")
        named_lists.append(named_clause)
        if cnf:
            named_strs.append("(" + " | ".join(named_clause) + ")")
        else:
            named_strs.append("(" + " & ".join(named_clause) + ")")
    if cnf:
        named_formula = " & ".join(named_strs)
    else:
        named_formula = " | ".join(named_strs)
    return named_lists, named_strs, named_formula


def generate_all_assignments(num_literals):
    """
    Generates all possible assignments of truth values to a set of literals
    Parameters:
        num_literals (int): Number of distinct literals
    Returns:
        pytorch.Tensor: A tensor of shape (2^num_literals, num_literals) representing all assignments
    """
    num_assignments = 2 ** num_literals
    assignments = torch.zeros((num_assignments, num_literals), dtype=torch.bool)
    for i in range(num_assignments):
        binary_repr = format(i, f'0{num_literals}b')
        assignments[i] = torch.tensor([int(bit) for bit in binary_repr], dtype=torch.bool)
    return assignments


def generate_random_assignments(num_literals, num_samples=1000):
    """
    Generates  possible assignments of truth values to a set of literals
    Parameters:
        num_literals (int): Number of distinct literals
        num_samples (int): Number of samples to generate
    Returns:
        pytorch.Tensor: A tensor of shape (num_samples, num_literals)
    """
    assignments = torch.randint(0, 2, (num_samples, num_literals), dtype=torch.bool)
    return assignments

def evaluate_formula(tensor, clauses, formula_type='cnf', proposition_names=None):
    """
    Evaluates a list of clause over all rows of a boolean tensor in PyTorch
    Parameters:
        tensor (torch.Tensor): Boolean PyTorch tensor of shape (N, M), where each row is a boolean assignment
        clauses (List[List[int]] or str): formula given as clauses for CNF/DNF, otherwise sympy string
        formula_type (str): 'cnf', 'dnf', or 'any' to evaluate the formula type
        proposition_names (List[str]): List of proposition names when formula_type is 'any'
    Returns:
        torch.Tensor: Boolean tensor of shape (N,) indicating whether each row satisfies the CNF/DNF formula
    """
    
    def evaluate_cnf(tensor, list_clauses):
        N, M = tensor.shape
        results = torch.ones(N, dtype=torch.bool, device=tensor.device)
        for clause in list_clauses:
            clause_results = torch.zeros(N, dtype=torch.bool, device=tensor.device)
            for literal in clause:
                var_idx = abs(literal) - 1
                if literal > 0:
                    clause_results |= tensor[:, var_idx]
                else:
                    clause_results |= ~tensor[:, var_idx]
            results &= clause_results
        return results
    
    def evaluate_dnf(tensor, list_clauses):
        N, M = tensor.shape
        results = torch.zeros(N, dtype=torch.bool, device=tensor.device)
        for clause in list_clauses:
            clause_results = torch.ones(N, dtype=torch.bool, device=tensor.device)
            for literal in clause:
                var_idx = abs(literal) - 1
                if literal > 0:
                    clause_results &= tensor[:, var_idx]
                else:
                    clause_results &= ~tensor[:, var_idx]
            results |= clause_results
        return results
    
    def evaluate_any(tensor, formula, proposition_names):
        formula = sympy.parse_expr(formula)
        predictions = []
        for sample in tensor:
            substitutions = {}
            for i, value in enumerate(sample):
                substitutions[proposition_names[i]] = value.item()
            predictions.append(bool(formula.subs(substitutions)))
        return torch.tensor(predictions, dtype=torch.bool, device=tensor.device)
    
    if formula_type == 'cnf':
        results = evaluate_cnf(tensor, clauses)
    elif formula_type == 'dnf':
        results = evaluate_dnf(tensor, clauses)
    else:
        results = evaluate_any(tensor, clauses, proposition_names)
    return results


def create_dataset(num_literals, clauses, formula_type='cnf', proposition_names=None):
    """
    Creates a dataset of all possible assignments and their evaluations for a given formula
    Parameters:
        num_literals (int): Number of distinct literals
        clauses (List[List[int]] or str): formula given as clauses for CNF/DNF, otherwise sympy string
        formula_type (str): 'cnf', 'dnf', or 'any' to evaluate the formula type
        proposition_names (List[str]): List of proposition names when formula_type is 'any'
    Returns:
        torch.Tensor: tensor of shape (2^num_literals, num_literals) representing all assignments
        torch.Tensor: tensor of shape (2^num_literals,) indicating whether each assignment satisfies the formula
    """
    tensor = generate_all_assignments(num_literals)
    results = evaluate_formula(tensor, clauses, formula_type, proposition_names)
    return tensor, results



WILDFIRE_VISUAL_VARIABLES = [
    'dense_forest',
    'dry_vegetation',
]

WILDFIRE_NON_VISUAL_VARIABLES = [
    'low_humidity',
    'strong_wind',
    'rained_recently',
    'high_temperature',
    'minimal_human_activity',
    'lightnings_frequent',
    'power_lines_nearby',
]

WILDFIRE_VARIABLE_NAMES = WILDFIRE_VISUAL_VARIABLES + WILDFIRE_NON_VISUAL_VARIABLES

WILDFIRE_FUEL_RULE = 'dense_forest | (dry_vegetation & strong_wind)'
WILDFIRE_DRY_RULE = 'low_humidity | (high_temperature & ~rained_recently)'
WILDFIRE_TRIGGER_RULE = 'lightnings_frequent | ~minimal_human_activity | power_lines_nearby'
WILDFIRE_GROUND_TRUTH_FORMULA = f'({WILDFIRE_FUEL_RULE}) & ({WILDFIRE_DRY_RULE}) & ({WILDFIRE_TRIGGER_RULE})'


def _wildfire_generate_non_visual_features(n_samples, seed=42):
    rng = np.random.RandomState(seed)
    return (rng.rand(n_samples, len(WILDFIRE_NON_VISUAL_VARIABLES)) > 0.5).astype(np.int32)


def _wildfire_compute_labels(visual_features, non_visual_features):
    features = np.concatenate([visual_features, non_visual_features], axis=1).astype(np.bool_)
    tensor = torch.from_numpy(features)
    labels = evaluate_formula(
        tensor,
        WILDFIRE_GROUND_TRUTH_FORMULA,
        formula_type='any',
        proposition_names=WILDFIRE_VARIABLE_NAMES,
    )
    return labels.numpy().astype(np.int32).reshape(-1)


def _wildfire_train_val_split(n_samples, train_ratio=0.75, seed=42):
    rng = np.random.RandomState(seed)
    idx = np.arange(n_samples)
    rng.shuffle(idx)
    train_size = int(train_ratio * n_samples)
    split = np.array(['val'] * n_samples, dtype=object)
    split[idx[:train_size]] = 'train'
    return split


def _enrich_wildfire_labels_dataframe(df, non_visual_seed=42, split_seed=42, train_ratio=0.75):
    # Keep deterministic schema shared by LoH and dILP runs.
    if not all(col in df.columns for col in WILDFIRE_VISUAL_VARIABLES):
        raise ValueError('labels.csv is missing required visual columns: %s' % WILDFIRE_VISUAL_VARIABLES)

    n_samples = len(df)

    if all(col in df.columns for col in WILDFIRE_NON_VISUAL_VARIABLES):
        non_visual = df[WILDFIRE_NON_VISUAL_VARIABLES].to_numpy(dtype=np.int32)
    else:
        print('No non-visual columns found in CSV, generating random features...')
        non_visual = _wildfire_generate_non_visual_features(n_samples, seed=non_visual_seed)
        for i, col in enumerate(WILDFIRE_NON_VISUAL_VARIABLES):
            df[col] = non_visual[:, i]

    if 'wildfire_risk' not in df.columns:
        print('No wildfire_risk column found in CSV, computing ground truth labels from features...')
        visual = df[WILDFIRE_VISUAL_VARIABLES].to_numpy(dtype=np.int32)
        df['wildfire_risk'] = _wildfire_compute_labels(visual, non_visual)

    if 'split' in df.columns:
        split = df['split'].astype(str).str.lower().to_numpy()
        split = np.where(split == 'train', 'train', 'val')
    else:
        print('No split column found in CSV, generating train/val split...')
        split = _wildfire_train_val_split(n_samples, train_ratio=train_ratio, seed=split_seed)
    df['split'] = split

    ordered_cols = ['filename'] + WILDFIRE_VISUAL_VARIABLES + WILDFIRE_NON_VISUAL_VARIABLES + ['wildfire_risk', 'split']
    existing = [c for c in ordered_cols if c in df.columns]
    remaining = [c for c in df.columns if c not in existing]
    return df[existing + remaining]


def prepare_wildfire_dataset(
    dataset_dir='LoH/experiments/data/wildfire_risk',
    image_size=256,
    n_images=2048,
    forest_probability=0.5,
    dryness_probability=0.4,
    image_seed=42,
    non_visual_seed=42,
    split_seed=42,
    train_ratio=0.75,
    force_regenerate=False,
):
    import os
    import pandas as pd

    img_dir = os.path.join(dataset_dir, 'images')
    csv_path = os.path.join(dataset_dir, 'labels.csv')

    needs_generation = (
        force_regenerate
        or not os.path.exists(csv_path)
        or not os.path.exists(img_dir)
        or len(os.listdir(img_dir)) == 0
    )

    if needs_generation:
        generate_wildfire_dataset(
            dataset_dir=dataset_dir,
            image_size=image_size,
            n_images=n_images,
            forest_probability=forest_probability,
            dryness_probability=dryness_probability,
            image_seed=image_seed,
            non_visual_seed=non_visual_seed,
            split_seed=split_seed,
            train_ratio=train_ratio,
        )
    else:
        df = pd.read_csv(csv_path)
        df = _enrich_wildfire_labels_dataframe(
            df,
            non_visual_seed=non_visual_seed,
            split_seed=split_seed,
            train_ratio=train_ratio,
        )
        df.to_csv(csv_path, index=False)
        print('Dataset already existed; labels.csv normalized to shared schema.')
        print(f'Images: {img_dir}')
        print(f'Labels: {csv_path}')

    return {
        'dataset_dir': dataset_dir,
        'img_dir': img_dir,
        'csv_path': csv_path,
    }


def generate_wildfire_dataset(
    dataset_dir='LoH/experiments/data/wildfire_risk',
    image_size=256,
    n_images=2048,
    forest_probability=0.5,
    dryness_probability=0.4,
    image_seed=42,
    non_visual_seed=42,
    split_seed=42,
    train_ratio=0.75,
):
    """
    Generate a synthetic wildfire image dataset and save a unified labels.csv schema:
    filename, visual features, non-visual features, wildfire_risk, split(train/val).
    """

    from PIL import Image, ImageDraw
    import numpy as np
    import os
    import pandas as pd

    random.seed(image_seed)
    np.random.seed(image_seed)

    img_dir = os.path.join(dataset_dir, 'images')
    csv_path = os.path.join(dataset_dir, 'labels.csv')
    os.makedirs(img_dir, exist_ok=True)

    def apply_dryness(color, dryness):
        """
        Shift a greenish color to a dry yellowish/brownish tone.
        dryness ∈ [0, 1]
        """
        if dryness <= 0:
            return color

        r, g, b = color
        # Reduce green, increase red/yellow tones
        g = int(g * (1 - 0.6 * dryness))
        r = int(r + 50 * dryness)
        b = int(b * (1 - 0.3 * dryness))

        # clamp
        r = min(max(r, 0), 255)
        g = min(max(g, 0), 255)
        b = min(max(b, 0), 255)

        return (r, g, b)


    def draw_background(draw, dry=False):
        """Draw a terrain background with dryness-aware coloring."""
        if dry:
            base_color = (160, 150, 60)  # dry yellow-ish
        else:
            base_color = (80, 180, 80)   # lush green

        draw.rectangle([0, 0, image_size, image_size], fill=base_color)

        # Additional large patches for texture variation
        for _ in range(random.randint(3, 6)):
            patch_color = (
                base_color[0] + random.randint(-20, 20),
                base_color[1] + random.randint(-20, 20),
                base_color[2] + random.randint(-20, 20),
            )
            x1 = random.randint(0, image_size - 60)
            y1 = random.randint(0, image_size - 60)
            x2 = x1 + random.randint(60, 140)
            y2 = y1 + random.randint(60, 140)
            draw.ellipse([x1, y1, x2, y2], fill=patch_color)


    def draw_river(draw):
        """Add a simple river in ~40% of images."""
        if random.random() < 0.6:
            return

        width = random.randint(10, 25)
        points = []
        for x in range(0, image_size, 30):
            y = random.randint(0, image_size)
            points.append((x, y))
        draw.line(points, fill=(40, 90, 200), width=width)


    def draw_house(draw):
        """Draw small houses."""
        for _ in range(random.randint(0, 3)):
            x = random.randint(10, image_size - 40)
            y = random.randint(10, image_size - 40)
            draw.rectangle([x, y, x + 30, y + 20], fill=(180, 150, 80))
            draw.polygon(
                [(x, y), (x + 15, y - 15), (x + 30, y)],
                fill=(160, 60, 60)
            )


    def draw_dense_forest(draw, dry=False):
        """Draw a dense forest using many overlapping leaf circles."""
        cx = random.randint(70, image_size - 70)
        cy = random.randint(70, image_size - 70)
        r = random.randint(40, 70)

        dryness_level = 1.0 if dry else 0.0

        for _ in range(90):  # high density
            x = int(np.random.normal(cx, r / 2))
            y = int(np.random.normal(cy, r / 2))
            size = random.randint(8, 20)

            # base green
            raw_color = (random.randint(20, 60), random.randint(100, 150), random.randint(20, 60))
            leaf_color = apply_dryness(raw_color, dryness_level)

            draw.ellipse([x, y, x + size, y + size], fill=leaf_color)


    def draw_sparse_vegetation(draw, dry=False):
        """Add scattered bushes and shrubs even if no dense forest."""
        dryness_level = 1.0 if dry else 0.0

        for _ in range(random.randint(20, 35)):
            x = random.randint(0, image_size)
            y = random.randint(0, image_size)
            size = random.randint(6, 12)
            raw_color = (random.randint(30, 70), random.randint(120, 180), random.randint(30, 70))
            veg_color = apply_dryness(raw_color, dryness_level)
            draw.ellipse([x, y, x + size, y + size], fill=veg_color)


    # ------------------------------------------------------
    # Dataset generation
    # ------------------------------------------------------
    rows = []
    print(f'Generating {n_images} images...')
    for i in range(n_images):
        img = Image.new('RGB', (image_size, image_size))
        draw = ImageDraw.Draw(img)

        has_forest = 1 if random.random() < forest_probability else 0
        is_dry = 1 if random.random() < dryness_probability else 0

        draw_background(draw, dry=bool(is_dry))
        draw_river(draw)
        draw_house(draw)
        if has_forest:
            draw_dense_forest(draw, dry=bool(is_dry))
        draw_sparse_vegetation(draw, dry=bool(is_dry))

        filename = f'image_{i:04d}.png'
        img.save(os.path.join(img_dir, filename))
        rows.append({
            'filename': filename,
            'dense_forest': has_forest,
            'dry_vegetation': is_dry,
        })

    df = pd.DataFrame(rows)
    df = _enrich_wildfire_labels_dataframe(
        df,
        non_visual_seed=non_visual_seed,
        split_seed=split_seed,
        train_ratio=train_ratio,
    )
    os.makedirs(dataset_dir, exist_ok=True)
    df.to_csv(csv_path, index=False)

    print('Generation complete.')
    print(f'Images: {img_dir}')
    print(f'Labels: {csv_path}')
    return {
        'dataset_dir': dataset_dir,
        'img_dir': img_dir,
        'csv_path': csv_path,
    }

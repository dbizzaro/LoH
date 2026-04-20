"""Wildfire experiment constants aligned with experiments/notebooks/wildfire_risk.ipynb."""

VISUAL_VARIABLES = [
    'dense_forest',
    'dry_vegetation',
]

NON_VISUAL_VARIABLES = [
    'low_humidity',
    'strong_wind',
    'rained_recently',
    'high_temperature',
    'minimal_human_activity',
    'lightnings_frequent',
    'power_lines_nearby',
]

ALL_VARIABLES = VISUAL_VARIABLES + NON_VISUAL_VARIABLES

FUEL_RULE = 'dense_forest | (dry_vegetation & strong_wind)'
DRY_RULE = 'low_humidity | (high_temperature & ~rained_recently)'
TRIGGER_RULE = 'lightnings_frequent | ~minimal_human_activity | power_lines_nearby'

GROUND_TRUTH_FORMULA = f'({FUEL_RULE}) & ({DRY_RULE}) & ({TRIGGER_RULE})'

CANDIDATE_RULE_GROUPS = {
    'fuel': [
        FUEL_RULE,
        'dry_vegetation & (strong_wind | low_humidity)',
        'dense_forest',
        'dense_forest & dry_vegetation',
    ],
    'dry': [
        DRY_RULE,
        'low_humidity & high_temperature & ~rained_recently',
        'high_temperature | (low_humidity & ~rained_recently)',
        '~rained_recently | (low_humidity & strong_wind)',
        '~rained_recently | (low_humidity & high_temperature)',
        'dry_vegetation & ~rained_recently',
    ],
    'trigger': [
        TRIGGER_RULE,
        'lightnings_frequent & minimal_human_activity',
        'power_lines_nearby & strong_wind',
        'rained_recently & lightnings_frequent',
        '~minimal_human_activity',
    ],
}

REGIME_FULL_KNOWLEDGE = 'full_knowledge'
REGIME_SELECT_RELIABLE = 'select_reliable_rules'
REGIME_SELECT_ONE_PER_SET = 'select_one_rule_per_set'
REGIME_PARTIAL_FUEL_KNOWN = 'partial_kb_fuel_known'

ALL_REGIMES = [
    REGIME_FULL_KNOWLEDGE,
    REGIME_SELECT_RELIABLE,
    REGIME_SELECT_ONE_PER_SET,
    REGIME_PARTIAL_FUEL_KNOWN,
]

FUEL_KNOWN_INDEX = 0

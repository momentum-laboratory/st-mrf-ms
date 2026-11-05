"""Scans in the study: (mouse, timepoint, scan name).

A scan's inputs, including its brain mask, live in
`data/in_vivo/<mouse>/week<N>/`; the scan name prefixes its output maps.

Cuprizone runs weeks 0-7; week 12 follows five weeks of recovery. Group sizes
shrink over time (one death before week 7; five animals taken for histology
at the end of week 7), so not every mouse has all four timepoints.
"""

SCANS = [
    ('mouse1',   '0',   'mouse1_week0'),
    ('mouse1',   '4',   'mouse1_week4'),
    ('mouse1',   '7',   'mouse1_week7'),
    ('mouse1',   '12',  'mouse1_week12'),

    ('mouse2',   '0',   'mouse2_week0'),
    ('mouse2',   '4',   'mouse2_week4'),
    ('mouse2',   '7',   'mouse2_week7'),
    ('mouse2',   '12',  'mouse2_week12'),

    ('mouse3',   '0',   'mouse3_week0'),
    ('mouse3',   '4',   'mouse3_week4'),
    ('mouse3',   '7',   'mouse3_week7'),
    ('mouse3',   '12',  'mouse3_week12'),

    ('mouse4',   '0',   'mouse4_week0'),
    ('mouse4',   '4',   'mouse4_week4'),

    ('mouse5',   '0',   'mouse5_week0'),
    ('mouse5',   '4',   'mouse5_week4'),
    ('mouse5',   '7',   'mouse5_week7'),

    ('mouse6',   '0',   'mouse6_week0'),
    ('mouse6',   '4',   'mouse6_week4'),
    ('mouse6',   '7',   'mouse6_week7'),

    ('mouse7',   '0',   'mouse7_week0'),
    ('mouse7',   '4',   'mouse7_week4'),
    ('mouse7',   '7',   'mouse7_week7'),

    ('mouse8',   '0',   'mouse8_week0'),
    ('mouse8',   '4',   'mouse8_week4'),
    ('mouse8',   '7',   'mouse8_week7'),
    ('mouse8',   '12',  'mouse8_week12'),

    ('mouse9',   '0',   'mouse9_week0'),
    ('mouse9',   '4',   'mouse9_week4'),
    ('mouse9',   '7',   'mouse9_week7'),

    ('mouse10',  '0',   'mouse10_week0'),
    ('mouse10',  '4',   'mouse10_week4'),
    ('mouse10',  '7',   'mouse10_week7'),
    ('mouse10',  '12',  'mouse10_week12'),

    ('mouse11',  '0',   'mouse11_week0'),
    ('mouse11',  '4',   'mouse11_week4'),
    ('mouse11',  '7',   'mouse11_week7'),

    ('mouse12',  '0',   'mouse12_week0'),
    ('mouse12',  '4',   'mouse12_week4'),
    ('mouse12',  '7',   'mouse12_week7'),
    ('mouse12',  '12',  'mouse12_week12'),
]

MICE = sorted({m for m, _, _ in SCANS}, key=lambda m: int(m[5:]))


# Cage and ear-tag of each animal, and the identifier used during acquisition.
# The raw scan deposit is organised under those original identifiers, so this
# is what maps a scan here back to its raw data.
ANIMALS = {
    'mouse1'  : {'cage': 1, 'ear_tag': '1R1L'  , 'acquisition_id': '1R1L_0032'},
    'mouse2'  : {'cage': 1, 'ear_tag': '1R'    , 'acquisition_id': '1R_0033'},
    'mouse3'  : {'cage': 1, 'ear_tag': '1L'    , 'acquisition_id': '1L_0034'},
    'mouse4'  : {'cage': 1, 'ear_tag': 'None'  , 'acquisition_id': 'None_0035'},
    'mouse5'  : {'cage': 2, 'ear_tag': 'None'  , 'acquisition_id': 'None_0036'},
    'mouse6'  : {'cage': 2, 'ear_tag': '1R1L'  , 'acquisition_id': '1R1L_0037'},
    'mouse7'  : {'cage': 2, 'ear_tag': '1R'    , 'acquisition_id': '1R_0038'},
    'mouse8'  : {'cage': 2, 'ear_tag': '1L'    , 'acquisition_id': '1L_0039'},
    'mouse9'  : {'cage': 3, 'ear_tag': '1R'    , 'acquisition_id': '1R_0040'},
    'mouse10' : {'cage': 3, 'ear_tag': 'None'  , 'acquisition_id': 'None_0041'},
    'mouse11' : {'cage': 3, 'ear_tag': '1L'    , 'acquisition_id': '1L_0042'},
    'mouse12' : {'cage': 3, 'ear_tag': '1R1L'  , 'acquisition_id': '1R1L_0043'},
}

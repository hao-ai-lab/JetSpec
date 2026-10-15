"""Training-free decoder calibration: schemas, survival fit, thresholds and replay."""


from __future__ import annotations


import argparse
import collections
import copy
import hashlib
import json
from jetspec.engines.jetspec_plus_plus.draft.method.legacy import _environment_name as environment_name
import math
import shutil
import sys
from pathlib import Path

from ..ops import reference_op
from ..passes import normalized_label, resolve_tau
N_WIRE = 15
JOINT_SLOT_NAMES = ("tau_J0d0", "tau_J1", "tau_J2", "temperature")


PASS_LABELS_ALL = ("J0d0", "J1", "J2")


DATASETS = ("gsm8k", "math500")


SPEC_SCHEMA = "e04-policy-v1"


POLICY_MS = (1, 2, 3, 4, 6, 8)


SLOTS_BY_M = {m: ({"k": "0.95", "anchor": "0.94"} if m == 1 else {"k": "0.99", "anchor": "0.99"}) for m in POLICY_MS}


ROW_BUDGETS = (16, 32)


ROW_BUDGET_POOL = 16


ROW_BUDGET_SLOTS = {"k": "0.99", "anchor": "0.99"}


S_SOURCES = ("fr", "posterior", "blend")


KSCORE_BLEND = "blend"


BLEND_COEFFICIENTS = ("a", "b", "c")


BLEND_THRESHOLDS_KEY = "e04_kscore_blend"


KSCORE_BLEND_DEC = "blend_dec"


BLEND_DEC_COEFFICIENTS = ("a", "b", "c", "argmax", "top", "pos")


KSCORE_COEFFICIENTS = {KSCORE_BLEND: BLEND_COEFFICIENTS, KSCORE_BLEND_DEC: BLEND_DEC_COEFFICIENTS}


ROWRULE_JOINT = "joint"


JOINT_POOL = 32


JOINT_ROWS_CAP = 32


JOINT_THRESHOLDS_KEY = "e04_rowrule_joint"


JOINT_SLOT = "rowrule_joint"


JOINT_T_GRID = tuple(round(0.30 + 0.05 * index, 2) for index in range(45))  # .30 .. 2.50


POLICY_FAMILIES = ("i", "p", "b", "u", "j")


FAMILY_MS = {"i": POLICY_MS, "p": (1, 2), "b": (), "u": (2, 3, 4), "j": ()}


PERMISSIVE_LEVEL = 1.0


K_CAP_PINNED = 10


BANK_CAPS = (16, 16, 128)


class GateError(SystemExit):
    """A failed gate: nonzero exit with a single-line reason."""

    def __init__(self, message, code=1):
        self.message = message
        print(message, file=sys.stderr, flush=True)
        super().__init__(code)






















KSCORE_SLOT = "kscore_blend"




def blend_fit(x, y, weights=None, start=None, tolerance=1e-12):
    """Newton's method on the logistic survival NLL (serving_f02 fit_blend; slot weights for a bootstrap draw).

    Returns (coefficients a, b, c as a float64 array, NLL at them). Unweighted from zero this is f02's fit step for
    step, which is how u01 reproduces the f02 coefficients. A weighted fit (a bootstrap draw) sums the three columns
    with plain reductions instead of BLAS: the same quantity, several times faster on a [n, 3] design, and its last
    bits never reach a served number (the draws only give half-widths).
    """
    import numpy as np
    width = x.shape[1]
    w = np.zeros(width) if start is None else np.array(start, dtype=np.float64)
    columns = None if weights is None else np.ascontiguousarray(x.T)
    for _ in range(100):
        if weights is None:
            z = x @ w
            p = 1.0 / (1.0 + np.exp(-z))
            gradient = x.T @ (p - y)
            hessian = (x * (p * (1 - p))[:, None]).T @ x + 1e-9 * np.eye(width)
        else:
            z = columns[0] * w[0]
            for index in range(1, width):
                z = z + columns[index] * w[index]
            p = 1.0 / (1.0 + np.exp(-z))
            residual, curvature = weights * (p - y), weights * p * (1 - p)
            gradient = np.array([np.sum(column * residual) for column in columns])
            hessian = np.array([[np.sum(columns[j] * curvature * columns[k]) for k in range(width)]
                                for j in range(width)]) + 1e-9 * np.eye(width)
        step = np.linalg.solve(hessian, gradient)
        w -= step
        if np.abs(step).max() < tolerance:
            break
    z = x @ w
    return w, float(np.sum(np.logaddexp(0.0, z) - y * z))


def joint_pool(row, offset):
    """(ids, probabilities) of the pool at one wire offset: the recorded top-JOINT_POOL offers, which are the wire's own
    pool (anchor q-hat 1.0, the post-exclusion softmax ranked by (-p, index), d_rej removed; the recorder's
    top_probabilities are the same post-exclusion values, so d_rej, at probability 0, is never among them)."""
    offers = row["offers"]
    ids, probabilities = offers["top_ids"][offset], offers["top_probabilities"][offset]
    if len(ids) < JOINT_POOL or len(probabilities) != len(ids):
        raise GateError(f"E04_JOINT_INPUT round {row.get('prompt_index')}/{row.get('round')} offset {offset}: "
                        f"{len(ids)} recorded offers < the pool {JOINT_POOL}")
    rejected = int(offers["d_rej"][offset])
    pool = [int(token) for token in ids[:JOINT_POOL]]
    if rejected >= 0 and rejected in pool:
        raise GateError(f"E04_JOINT_INPUT round {row.get('prompt_index')}/{row.get('round')} offset {offset}: d_rej "
                        "is among the recorded top offers")
    return pool, [float(value) for value in probabilities[:JOINT_POOL]]


def joint_anchor_probabilities(probabilities, temperature):
    """P_T(a | k) over the pool rows [r, JOINT_POOL]: p^(1/T) renormalized inside the pool (serving_x01
    anchor_probabilities with renormalize True; p clipped below at 1e-300 as there)."""
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    scaled = np.power(np.clip(probabilities, 1e-300, None), 1.0 / float(temperature))
    return scaled / np.clip(scaled.sum(axis=-1, keepdims=True), 1e-300, None)


def joint_temperature(rows, temperatures=JOINT_T_GRID):
    """(T, nll at T, nll at T = 1 renormalized, nll as served, rounds) on the normal proposal rounds whose true anchor is
    in the pool at k_true: the mean NLL of the true anchor under P_T over the grid (serving_x01 fit_temperature)."""
    import numpy as np
    pools = []
    for row in rows:
        if bool(row["after_miss"]):
            continue
        ids, probabilities = joint_pool(row, int(row["k_true"]))
        if int(row["true_anchor"]) in ids:
            pools.append((probabilities, ids.index(int(row["true_anchor"]))))
    if not pools:
        raise GateError("E04_JOINT_INPUT no normal round with the true anchor in the pool")
    probabilities = np.array([pool[0] for pool in pools], dtype=np.float64)
    truth = np.array([pool[1] for pool in pools])
    select = np.arange(len(pools))

    def nll(temperature, renormalize=True):
        values = joint_anchor_probabilities(probabilities, temperature) if renormalize else probabilities
        return float(-np.log(np.clip(values[select, truth], 1e-300, None)).mean())
    values = [nll(value) for value in temperatures]
    best = int(np.argmin(values))
    return float(temperatures[best]), values[best], nll(1.0), nll(1.0, renormalize=False), len(pools)




@reference_op("conformal_qhat")
def conformal_qhat(scores, coverage, *, validate_coverage=True,
                   empty_message="split-conformal calibration requires at least one score"):
    """The ceil((n+1)*coverage)-th score, or 1 for the unseen tail.

    Historical reader adapters disable coverage validation; their valid-input
    calibration results use exactly the same quantile computation.
    """
    values = sorted(float(value) for value in scores)
    coverage = float(coverage)
    if not values:
        raise ValueError(empty_message)
    if validate_coverage and not 0.0 < coverage <= 1.0:
        raise ValueError("split-conformal coverage must be in (0,1]")
    rank = math.ceil((len(values) + 1) * coverage)
    return 1.0 if rank > len(values) else values[rank - 1]


def dataset_entry(entries, dataset):
    """Resolve a per-dataset threshold, falling back to its pooled entry."""
    if dataset in entries and 'pooled' in entries:
        raise ValueError(f'ambiguous pooled and per-dataset thresholds for {dataset}')
    return entries[dataset] if dataset in entries else entries['pooled']


COEFFICIENTS = ("a", "b", "c", "argmax", "top", "pos")
PASS_LABELS = ("J0d0", "J1", "J2", "J3")
GRID = 2.0 ** 32
POSITION_SCALE = 14.0
DRAFT_WIDTHS = (3, 7, 15, 31)
JOINT_MIN_GROUP = 100


def _served_qhat(scores, coverage):
    return conformal_qhat(scores, coverage, validate_coverage=False, empty_message="no scores")


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _logit(values):
    import numpy as np

    values = np.clip(np.asarray(values, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(values) - np.log1p(-values)


def _features(rows, k_max):
    import numpy as np

    width = k_max + 1
    p_top1 = np.asarray([row["p_top1"] for row in rows], dtype=np.float64)
    q_shadow = np.asarray(
        [[np.nan if value is None else value for value in row["q_shadow"]] for row in rows], dtype=np.float64
    )
    if p_top1.shape != (len(rows), width) or q_shadow.shape != (len(rows), width):
        raise ValueError(f"SERVED_REFIT_INPUT p_top1 and q_shadow must each have {width} entries")
    argmax = np.asarray(
        [[float(best == rejected) for best, rejected in zip(row["shadow_argmax"], row["offers"]["d_rej"])]
         for row in rows],
        dtype=np.float64,
    )
    top = np.asarray([[max(values) for values in row["offers"]["top_probabilities"]] for row in rows])
    columns = [
        _logit(p_top1[:, 1:]),
        _logit(q_shadow[:, :k_max]),
        np.ones((len(rows), k_max)),
        argmax[:, :k_max],
        _logit(top[:, :k_max]),
        # Serving divides slot i by the constant live_rows E04_KSCORE_POSITION_SCALE (14) at every width.
        np.broadcast_to(np.arange(1, k_max + 1) / POSITION_SCALE, (len(rows), k_max)),
    ]
    result = np.stack(columns, axis=2)
    if not np.isfinite(result).all():
        raise ValueError("SERVED_REFIT_INPUT non-finite decoder features")
    return result




def _masses(features, weights):
    import numpy as np

    z = weights[0] * features[:, :, 0] + weights[1] * features[:, :, 1] + weights[2]
    z = z + weights[3] * features[:, :, 3] + weights[4] * features[:, :, 4] + weights[5] * features[:, :, 5]
    continuation = 1.0 / (1.0 + np.exp(-z))
    survival = np.concatenate([np.ones((len(continuation), 1)), np.cumprod(continuation, axis=1)], axis=1)
    mass = np.concatenate([survival[:, :-1] * (1.0 - continuation), survival[:, -1:]], axis=1)
    return np.round(mass * GRID) / GRID


def _verified_pin(pin, checkpoint):
    required = {"checkpoint_sha256", "decoder_sha256", "tensor_count"}
    if not isinstance(pin, dict) or not required.issubset(pin):
        raise ValueError(f"DECODER_PIN_INVALID missing {sorted(required - set(pin or {}))}")
    observed = _sha256(checkpoint)
    if observed != pin["checkpoint_sha256"]:
        raise ValueError(f"DECODER_PIN_MISMATCH {observed} != {pin['checkpoint_sha256']}")
    return dict(pin)


def fit_served_population(populations, *, arm, decoder_pin, decoder_checkpoint, level=0.93,
                          pooled=False, dataset_names=("gsm8k", "math500"), mode='generic-v1',
                          _reader_compat=False):
    """Fit one decoder arm over named calibration populations, including J3."""
    import numpy as np

    if arm not in ("U", "T"):
        raise ValueError("SERVED_REFIT_INPUT arm must be U or T")
    if mode not in ('generic-v1', 'historical-9732388a'):
        raise ValueError(f'unknown served refit mode {mode!r}')
    historical = mode == 'historical-9732388a'
    pin = _verified_pin(decoder_pin, decoder_checkpoint)
    datasets = tuple(dataset_names)
    if not datasets or len(populations) != len(datasets) or len(set(datasets)) != len(datasets):
        if _reader_compat:
            raise ValueError("SERVED_REFIT_INPUT expected gsm8k and math500 populations")
        raise ValueError("SERVED_REFIT_INPUT populations need distinct dataset names")
    prepared, designs, labels = [], [], []
    k_max = None
    for dataset, raw in zip(datasets, populations):
        rows = [row for row in raw if row.get("flat_proposal") and row.get("offers") is not None
                and not bool(row.get("after_miss"))]
        if not rows or any(row.get("dataset") != dataset for row in rows):
            raise ValueError(f"SERVED_REFIT_INPUT no valid {dataset} rows")
        if any((row.get("pass_label") not in PASS_LABELS if historical else
                not normalized_label(row.get("pass_label"))) for row in rows):
            raise ValueError("SERVED_REFIT_INPUT pass label outside J0d0/J1/J2/J3")
        if _reader_compat:
            width = 15
        else:
            widths = {len(row["p_top1"]) - 1 for row in rows}
            if len(widths) != 1 or next(iter(widths)) not in DRAFT_WIDTHS:
                raise ValueError("SERVED_REFIT_INPUT expected draft width 3, 7, 15 or 31")
            width = next(iter(widths))
        if k_max is not None and width != k_max:
            raise ValueError("SERVED_REFIT_INPUT mixed draft widths")
        k_max = width
        features = _features(rows, k_max)
        k_true = np.asarray([int(row["k_true"]) for row in rows])
        if ((k_true < 0) | (k_true > k_max)).any():
            raise ValueError(f"SERVED_REFIT_INPUT k_true outside 0..{k_max}")
        slots = np.arange(1, k_max + 1)[None, :]
        risk = slots <= np.minimum(k_true[:, None] + 1, k_max)
        designs.append(features[risk])
        labels.append((slots <= k_true[:, None])[risk].astype(np.float64))
        prepared.append((dataset, rows, features, len(raw)))
    weights, nll = blend_fit(np.concatenate(designs), np.concatenate(labels))
    result = {
        "arm": arm,
        "decoder_pin": pin,
        "k_max": k_max,
        "coefficients": dict(zip(COEFFICIENTS, map(float, weights))),
        "fit_nll": nll,
        "fit_slots": sum(len(values) for values in labels),
        "datasets": {},
    }
    groups = [("pooled", [row for _, rows, _, _ in prepared for row in rows],
               np.concatenate([features for _, _, features, _ in prepared]),
               sum(total for _, _, _, total in prepared))] if pooled else prepared
    for dataset, rows, features, total in groups:
        temperature, nll_temperature, _, _, in_pool = joint_temperature(rows)
        masses = _masses(features, weights)
        values = []
        for index, row in enumerate(rows):
            k_true = int(row["k_true"])
            ids = row["offers"]["top_ids"][k_true][:32]
            probabilities = np.asarray(row["offers"]["top_probabilities"][k_true][:32], dtype=np.float64)
            if len(ids) != 32 or len(probabilities) != 32:
                raise ValueError("SERVED_REFIT_INPUT joint offer pool must contain 32 entries")
            scaled = np.clip(probabilities, 1e-300, None) ** (1.0 / temperature)
            anchor = scaled / max(scaled.sum(), 1e-300)
            truth = int(row["true_anchor"])
            value = masses[index, k_true] * anchor[ids.index(truth)] if truth in ids else 0.0
            values.append(float(np.round(value * GRID) / GRID))
        scores = 1.0 - np.asarray(values)
        pooled_qhat = _served_qhat(scores.tolist(), level)
        pass_labels = PASS_LABELS if historical else tuple(sorted(
            set(PASS_LABELS) | {row['pass_label'] for row in rows},
            key=lambda label: 0 if label == 'J0d0' else int(label[1:])))
        tau = {"J0d0": 1.0 - pooled_qhat}
        if not historical:
            tau['pooled'] = 1.0 - pooled_qhat
        counts = {}
        row_labels = np.asarray([row["pass_label"] for row in rows])
        for label in pass_labels:
            selected = row_labels == label
            counts[label] = int(selected.sum())
            if label == "J0d0":
                continue
            own = counts[label] >= JOINT_MIN_GROUP or (historical and label == "J3" and counts[label] > 0)
            tau[label] = 1.0 - (_served_qhat(scores[selected].tolist(), level) if own else pooled_qhat)
        destination = result if pooled else result["datasets"]
        destination[dataset] = {
            "T": float(temperature),
            "T_nll": float(nll_temperature),
            "tau": {label: float(value) for label, value in tau.items()},
            "total_rounds": total,
            "normal_rounds": len(rows),
            "in_pool": in_pool,
            "pass_counts": counts,
        }
    return result




PASSES = PASS_LABELS_ALL
K_SLOTS, A_SLOTS = ("0.95", "0.99"), ("0.94", "0.99")
SERVED_G = {"gsm8k": {"0.94": 0.9618559069931507, "0.99": 0.999704080139054},
            "math500": {"0.94": 0.9029182195663452, "0.99": 0.9933349778875709}}




def _require(condition, message):
    if not condition:
        raise ComposeError(message)


require = _require




CAL28_SHA = 'f9f77088ac74b91634f451adc1aff1a31d3946fd54915255d9a9008d15de7b0f'
LEGACY_SHA = 'd42a9c855c4b802bbc42c557675423031e26869c9990897059a2669b9519a5f5'
PROMPTS_SHA = {'8B': '5393ea5ee6ed26f795e508c6d2487d4ae07da28fb844402cd246c2b56edda87a',
               '27B': '43ad6a9de6647589232c4546473c8f1ee936be3fcc06e56ffa223446b917aace'}


def stage_cal28(argv=None):
    """Build the pinned 27B cal28 bundle from copied, verified source files."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--fit', type=Path, required=True)
    parser.add_argument('--legacy', type=Path, required=True)
    parser.add_argument('--host', type=Path, required=True)
    parser.add_argument('--cache-source', type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.root
    sources = root / 'sources'
    sources.mkdir(parents=True, exist_ok=True)
    fit = sources / 'cal28-thresholds.json'
    legacy = sources / 'e6v3-thresholds.json'
    shutil.copyfile(args.fit / 'thresholds.json', fit)
    shutil.copyfile(args.legacy, legacy)
    shutil.copyfile(args.fit / 'pin.json', sources / 'cal28-pin.json')
    expected_fit = 'f9f77088ac74b91634f451adc1aff1a31d3946fd54915255d9a9008d15de7b0f'
    expected_legacy = 'd42a9c855c4b802bbc42c557675423031e26869c9990897059a2669b9519a5f5'
    assert sha(fit) == expected_fit, sha(fit)
    assert sha(legacy) == expected_legacy, sha(legacy)
    host = json.loads(args.host.read_text())
    placeholder = Path(host['artifacts']['jetspecpp_27b_placeholder_heads'])
    manifest = placeholder / 'decoder/manifest.json'
    assert manifest.read_bytes() == (args.fit / 'decoder/manifest.json').read_bytes()
    assert sha(manifest) == '43eb014293a5bace28116f6289ca3b893507ccbe2104a102bb4d370b4a0c308b'
    name = 'jetspecpp_27b_base800k_heads_cal28'
    bundle = root / 'artifacts' / name
    shutil.copytree(placeholder, bundle, dirs_exist_ok=True)
    blended = json.loads(legacy.read_text())
    fitted = json.loads(fit.read_text())
    sections = ('e04_kscore_blend', 'e04_rowrule_joint')
    for section in sections:
        blended[section] = fitted[section]
    (bundle / 'thresholds.json').write_text(json.dumps(blended, indent=2) + '\n')
    receipt = dict(schema='jetspec.cal28_fit_receipt/1', bundle=name,
                   thresholds_sha256=sha(bundle / 'thresholds.json'),
                   decoder_manifest_sha256=sha(manifest),
                   sources=[dict(label='cal28-27b-fit', sha256=expected_fit,
                                 sections=list(sections)),
                            dict(label='served_heads_reused_e6v3_b16', sha256=expected_legacy,
                                 repository='Moonlight556/jetspec-served-heads-reused',
                                 revision='a9b066f1', sections='all other sections')])
    (root / 'fit-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    override = root / 'cal28-as-is'
    (override / 'decoder').mkdir(parents=True, exist_ok=True)
    shutil.copyfile(fit, override / 'thresholds.json')
    shutil.copyfile(manifest, override / 'decoder/manifest.json')
    (override / 'pin.json').write_text(json.dumps(dict(
        schema='jetspec.hybrid_thresholds_pin/1', thresholds_sha256=expected_fit)) + '\n')
    host['artifacts'][name] = str(bundle)
    # Copy the cache into this dispatch before any launcher writes to it.
    cache = root / 'hf-cache'
    if not cache.exists():
        shutil.copytree(args.cache_source, cache)
    host['environment'][environment_name('hardware', 'hardware_settings.hf_cache')] = str(cache)
    (root / 'host-values.json').write_text(json.dumps(host, indent=2) + '\n')
    print('CAL28_STAGE=PASS ' + json.dumps(receipt), flush=True)


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True) + '\n').encode()


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


sha = _sha


def first_difference(left, right, field='$'):
    if type(left) is not type(right):
        return f'{field}: {left!r} != {right!r}'
    if isinstance(left, dict):
        for key in sorted(left.keys() | right.keys()):
            if key not in left or key not in right:
                return f'{field}.{key}: {left.get(key)!r} != {right.get(key)!r}'
            diff = first_difference(left[key], right[key], f'{field}.{key}')
            if diff:
                return diff
    elif isinstance(left, list):
        if len(left) != len(right):
            return f'{field}.length: {len(left)} != {len(right)}'
        for index, (a, b) in enumerate(zip(left, right)):
            diff = first_difference(a, b, f'{field}[{index}]')
            if diff:
                return diff
    elif left != right:
        return f'{field}: {left!r} != {right!r}'
    return None


def double_fit(fit, output):
    output.mkdir(parents=True, exist_ok=True)
    left = fit()
    (output / 'fit-1.json').write_bytes(encoded(left))
    right = fit()
    (output / 'fit-2.json').write_bytes(encoded(right))
    if encoded(left) != encoded(right):
        raise ValueError('FIT_BYTE_GATE_FAIL ' + first_difference(left, right))
    print('FIT_BYTE_GATE=PASS sha256=' + sha(output / 'fit-1.json'), flush=True)
    return left




def load_capture(capture):
    """Keep scored proposal truth, excluding recovery records and unverified tails."""
    out = json.loads((capture / 'output.json').read_text())
    files = sorted((capture / 'output.json.rows').glob('*.jsonl'))
    if not files:
        raise ValueError('CAPTURE_NO_RECORDS')
    records = [json.loads(line) for path in files for line in path.read_text().splitlines() if line.strip()]
    native = out['metrics']['jetspec_round_records']
    joined = {(r['seq_id'], r['candidate_round'] + (1 if 'next_correction' in r else 0)): r for r in native
              if r['candidate_round'] is not None}
    props = [r for r in records if r.get('record_kind', 'proposal') == 'proposal']
    keys = collections.Counter((r['seq_id'], r['candidate_round']) for r in props)
    if any(value > 1 for value in keys.values()):
        raise ValueError('CAPTURE_DUPLICATE_PROPOSAL')
    truth_joined = 0
    for row in props:
        # Eager records snapshot scoring before verification; chain records
        # already carry their joined truth. Only join the identical scored block.
        if 'record_kind' not in row and row.get('k_true') is None:
            truth = joined.get((row['seq_id'], row['candidate_round']))
            if truth is not None:
                if (row['offers']['d_rej'][:-1] != truth['candidate_draft_ids']
                        or row['fallback_anchor'] != truth['correction']):
                    raise ValueError(f"CAPTURE_SCORED_BLOCK_JOIN {(row['seq_id'], row['candidate_round'])}")
                row.update(k_true=truth['accepted_len'] - 1, true_anchor=truth['next_correction'])
                truth_joined += 1
    missing = [r for r in props if r.get('k_true') is None or r.get('true_anchor') is None]
    kept = [r for r in props if r not in missing]
    if not kept:
        raise ValueError('CAPTURE_NO_VERIFIED_PROPOSALS')
    for row in kept:
        key = row['seq_id'], row['candidate_round']
        truth = joined.get(key)
        if (truth is None or row['k_true'] != (truth['accepted_len'] - 1 if 'next_correction' in truth else truth['k_true'])
                or row['true_anchor'] != truth.get('next_correction', truth.get('correction'))):
            raise ValueError(f'CAPTURE_TRUTH_JOIN {key}')
    rows = [dict(row, dataset='gsm8k', flat_proposal=True) for row in kept]
    audit = {'capture': str(capture), 'output_sha256': sha(capture / 'output.json'),
             'records': len(records), 'proposals': len(props),
             'redrafts_excluded': sum(r.get('record_kind') == 'redraft' for r in records),
             'no_truth_excluded': len(missing), 'native_rounds': len(native), 'verified_joined': len(kept),
             'after_miss_rows_filtered_by_tool': sum(bool(r.get('after_miss')) for r in kept),
             'rounds_per_pass_label': dict(collections.Counter(r['pass_label'] for r in kept)),
             'raw_pass_labels': dict(collections.Counter(r.get('raw_pass_label') for r in kept)),
             'prompts': len(out['outputs']), 'tokens': sorted({len(o['token_ids']) for o in out['outputs']})}
    if truth_joined:
        audit['truth_joined_from_native'] = truth_joined
    return rows, audit


def fit_capture(rows, manifest, checkpoint=None, *, mode='generic-v1'):
    digest = sha(manifest)
    pin = {'checkpoint_sha256': sha(checkpoint or manifest), 'decoder_sha256': digest, 'tensor_count': None,
           'pin_mode': 'checkpoint' if checkpoint else
                       'target_slice decoder/manifest.json (no checkpoint file in the 27B bundle)'}
    return fit_served_population([rows], arm='U', decoder_pin=pin,
                                 decoder_checkpoint=checkpoint or manifest, pooled=True,
                                 dataset_names=('gsm8k',), mode=mode)


def cal28_gate(capture, base, manifest, reference, output, *, served_source):
    """Reproduce historical bytes, including their original fit provenance."""
    rows, audit = load_capture(capture)
    def fit():
        result = fit_capture(rows, manifest, mode='historical-9732388a')
        result['eng04_decoder'] = '27b-target-slice'
        result['calibration_cells'] = [dict(audit, cell='cal28-p2-capture-56x512',
                                           prompts_sha256=PROMPTS_SHA['27B'])]
        result['fit_tool'] = {
            'source': '9732388a predictor/reader/refit_served_e04.py',
            'refit_served_e04_blob': '52d18cf05393cc336f665f6aec01d47cb4f01c85',
            'e04lib_blob': 'd1a6083c3eb95559da5bcdc538368b49e59c7109',
            'served_py_sha256': sha(served_source)}
        return result
    result = double_fit(fit, output)
    historical = dict(json.loads(base.read_text()))
    historical['eng04_ablation_refit'] = {
        'decoder': '27b-target-slice (decoder/manifest.json ' + sha(manifest) + ')',
        'calibration': 'eng04_cal28 pooled_calibration_27b_56 56x512 deadline off',
        'refit_sha256': sha(output / 'fit-1.json'), 'level': 0.93,
        'base_thresholds_sha256': sha(base), 'calibration_cells': ['cal28-p2-capture-56x512']}
    for index in (1, 2):
        make_bundle(historical, result, manifest, output / f'bundle-{index}')
        actual = output / f'bundle-{index}/thresholds.json'
        if sha(actual) != CAL28_SHA:
            expected = json.loads(reference.read_text())
            difference = first_difference(expected, json.loads(actual.read_text()))
            raise ValueError(f'CAL28_BYTE_GATE_FAIL expected={CAL28_SHA} actual={sha(actual)} '
                             + (difference or 'JSON equal but serialization differs'))
        print(f'CAL28_BYTE_GATE=PASS fit={index} thresholds_sha256={sha(actual)}', flush=True)
    report = replay(rows, result)
    (output / 'rows-min.json').write_bytes(encoded(report))
    (output / 'gate-receipt.json').write_bytes(encoded({
        'schema': 'jetspec.cal28_reproduction_gate/1',
        'sources': sources_for(capture, [base, manifest, reference, Path(__file__),
                                        Path(__file__),
                                        Path(__file__),
                                        served_source]),
        'fit_sha256': sha(output / 'fit-1.json'), 'thresholds_sha256': CAL28_SHA,
        'fit_byte_identical': True, 'replay': report}))
    print('CAL28_REPLAY ' + json.dumps(report, sort_keys=True), flush=True)
    return result


def select_replay_rows(scores, keys, *, tau, floor, cap, fallback):
    import torch
    from jetspec.engines.jetspec_plus_plus.draft.method.cache_rows import DeviceOffers, floor_passing, select_rows as method_select
    score = torch.tensor(scores, dtype=torch.float64)
    pairs = torch.tensor(keys, dtype=torch.long).reshape(-1, 2)
    valid = torch.ones_like(score, dtype=torch.bool)
    passing = floor_passing(score, (score >= tau) & (score > 0), valid, floor)
    offers = DeviceOffers(pairs[:, 1:2], score[:, None], valid[:, None], pairs[:, 0])
    requested, selected = method_select(score, passing, offers, cap, cap)
    return selected[:min(int(requested), cap)].tolist() or [[0, fallback]]


select_rows = select_replay_rows




def replay_capture(rows, fit, *, served_floor=0):
    normal = [row for row in rows if not row['after_miss']]
    hit_accept = [row['k_true'] + 1 for row in normal
                  if [row['k_true'], row['true_anchor']] in row['published_keys']]
    miss_accept = [row['k_true'] + 1 for row in normal
                   if [row['k_true'], row['true_anchor']] not in row['published_keys']]
    means = [sum(values) / len(values) if values else None for values in (hit_accept, miss_accept)]
    hits = collections.Counter()
    mismatches = []
    tolerated = 0
    order_only = 0
    old_rows = new_rows = 0
    for row in rows:
        scores, keys = pair_scores(row, fit)
        options = dict(tau=resolve_tau(fit['pooled']['tau'], row['pass_label']),
                       cap=row.get('row_cap', 16), fallback=row['fallback_anchor'])
        own = select_rows(scores, keys, floor=served_floor, **options)
        old_rows += len(row['published_keys'])
        new_rows += len(own)
        selected = {tuple(key) for key in own}
        recorded = {tuple(key) for key in row['published_keys']}
        if selected != recorded:
            score_by_key = dict(zip(map(tuple, keys), scores))
            removed, added = recorded - selected, selected - recorded
            pairs = []
            if len(removed) == len(added) and all(key in score_by_key for key in removed | added):
                pairs = list(zip(sorted(score_by_key[key] for key in removed),
                                 sorted(score_by_key[key] for key in added)))
            # Relative distance uses the larger score. Zero/unknown scores and
            # changed cardinality cannot be excused as a near-tied substitution.
            near_tie = bool(pairs) and all(max(a, b) > 0 and min(a, b) >= 0.98 * max(a, b)
                                          for a, b in pairs)
            tolerated += int(near_tie)
            mismatches.append({'seq_id': row['seq_id'], 'candidate_round': row['candidate_round'],
                               'recorded': row['published_keys'], 'replayed': own,
                               'swapped_score_pairs': pairs, 'within_score_tolerance': near_tie})
        elif own != row['published_keys']:
            order_only += 1
        if not row['after_miss']:
            for n in (0, 4, 8, 12, 16):
                expanded = dict(options)
                if n:
                    expanded['cap'] = 1 << (max(4, options['cap'], n) - 1).bit_length()
                chosen = select_rows(scores, keys, floor=n, **expanded)
                hits[n] += [row['k_true'], row['true_anchor']] in chosen
    table = []
    for n in (0, 4, 8, 12, 16):
        h = hits[n] / len(normal) if normal else None
        # an06 estimate holds drafts fixed; it cannot predict changed future paths.
        estimate = h * means[0] + (1 - h) * means[1] if h is not None and None not in means else None
        table.append({'rows_min': n, 'rounds': len(normal), 'joint_hit': h,
                      'estimated_tau': estimate, 'tau_hit': means[0], 'tau_miss': means[1]})
    return {'table': table, 'published_keys_total': len(rows),
            'published_keys_matched': len(rows) - len(mismatches),
            'mismatch_count': len(mismatches), 'first_mismatch': mismatches[:1],
            'order_only_count': order_only, 'tolerated_mismatch_count': tolerated,
            'tolerated_mismatch_share': tolerated / len(rows),
            'untolerated_mismatch_count': len(mismatches) - tolerated,
            'first_untolerated_mismatch': [m for m in mismatches if not m['within_score_tolerance']][:1],
            'published_keys_within_tolerance': len(rows) - len(mismatches) + tolerated,
            'relative_score_tolerance': 0.02, 'tolerated_proposal_limit': 0.005,
            'key_sets_changed_share': len(mismatches) / len(rows),
            'rows_per_round': {'old': old_rows / len(rows), 'new': new_rows / len(rows)},
            'estimate_method': 'an06: hit * tau_given_hit + (1-hit) * tau_given_miss; normal scored rounds; '
                               'row buckets expand as serve_policy.rows_min_capacity'}


replay = replay_capture




def published_keys_gate(report, name):
    tolerated = report.get('tolerated_mismatch_count', 0)
    outside = report.get('untolerated_mismatch_count', report['mismatch_count'])
    if outside or tolerated * 200 > report['published_keys_total']:
        first = report.get('first_untolerated_mismatch') or report['first_mismatch']
        raise ValueError(name + ' ' + json.dumps(dict(first_mismatch=first,
            tolerated_mismatch_count=tolerated, published_keys_total=report['published_keys_total'])))






def main(argv=None):
    """Fit or replay captured CPU records; model weights are never loaded."""
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == 'stage-cal28':
        stage_cal28(argv[1:])
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    fit_parser = commands.add_parser('fit')
    fit_parser.add_argument('--capture', type=Path, required=True)
    fit_parser.add_argument('--decoder-manifest', type=Path, required=True)
    fit_parser.add_argument('--decoder-checkpoint', type=Path)
    fit_parser.add_argument('--mode', choices=('generic-v1', 'historical-9732388a'), default='generic-v1')
    fit_parser.add_argument('--output', type=Path, required=True)
    replay_parser = commands.add_parser('replay')
    replay_parser.add_argument('--capture', type=Path, required=True)
    replay_parser.add_argument('--fit', type=Path, required=True)
    replay_parser.add_argument('--served-floor', type=int, default=0)
    replay_parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError(f'OUTPUT_EXISTS {args.output}')
    rows, audit = load_capture(args.capture)
    if args.command == 'fit':
        result = double_fit(lambda: fit_capture(rows, args.decoder_manifest, args.decoder_checkpoint,
                                               mode=args.mode), args.output)
        (args.output / 'capture-audit.json').write_bytes(encoded(audit))
    else:
        report = replay(rows, json.loads(args.fit.read_text()), served_floor=args.served_floor)
        published_keys_gate(report, 'PUBLISHED_KEYS_REPLAY_FAIL')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(encoded(report))
        print('PUBLISHED_KEYS_REPLAY=PASS ' + json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

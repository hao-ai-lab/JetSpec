"""CPU equivalence for BLEND mass, keyed rows, acceptance and local artifacts."""
from golden_values import golden
import json
from types import SimpleNamespace

import pytest
import torch

@pytest.mark.parametrize('seed', [0, 1, 2])
def test_blend_mass_matches_golden(seed):
    from jetspec.methods.jetspec_plus_plus.method.kmass import blend_mass

    torch.manual_seed(seed)
    logits = torch.randn(2, 3, dtype=torch.float32)
    assert torch.equal(blend_mass(logits), golden('value_1'))
    assert torch.equal(blend_mass(logits.double()), golden('value_2'))


@pytest.mark.parametrize('seed', [0, 1, 2])
@pytest.mark.parametrize('drafts', [3, 15])
def test_blend_features_match_served_golden(monkeypatch, seed, drafts):
    from jetspec.methods.jetspec_plus_plus.method.kmass import blend

    for name, value in {'JETSPEC_DRAFTS': str(drafts), 'JETSPEC_ROW_LAYOUT': 'anchor0',
                        'JETSPEC_DECODER_FRAME': 'wire', 'E04_KSCORE': 'blend_dec'}.items():
        monkeypatch.setenv(name, value)
    torch.manual_seed(seed)
    logits = torch.randn(drafts + 1, 11, dtype=torch.float32)
    draft_ids = torch.randint(0, 11, (drafts,))
    draft_ids[0] = logits[0].argmax()
    p_top1 = torch.rand(drafts + 1, dtype=torch.float32)
    p_top1[1:3] = torch.tensor([0., 1.])
    probabilities = torch.rand(drafts + 1, 4, dtype=torch.float64)
    sizes = torch.randint(1, 5, (drafts + 1,))
    coefficients = dict(a=0.8, b=0.4, c=-0.1, argmax=0.3, top=-0.2, pos=0.7)
    thresholds = {'e04_kscore_blend': {'pooled': {
        name: {'value': value} for name, value in coefficients.items()}}}
    block = SimpleNamespace(draft_ids=draft_ids, e04_p_top1=p_top1)
    expected, checks = golden('value_3')
    actual = blend(p_top1, logits, draft_ids, probabilities, sizes,
                   coefficients=coefficients)
    assert checks.tolist() == [0, 0]
    assert actual.shape == (drafts + 2,)
    assert actual.dtype == torch.float64
    assert torch.equal(actual, expected)


@pytest.mark.parametrize('seed', [0, 1, 2])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_keyed_row_functions_match_golden(seed, dtype):
    from jetspec.methods.jetspec_plus_plus.method import rows

    torch.manual_seed(seed)
    ids = torch.randint(0, 31, (3, 4))
    probabilities = torch.rand(3, 4, dtype=dtype)
    valid = torch.rand(3, 4) > 0.2
    ks = torch.tensor([0, 3, 7])
    offers = rows.DeviceOffers(ids, probabilities, valid, ks)
    for name in ('ids', 'probabilities', 'valid', 'ks'):
        assert torch.equal(getattr(offers, name), golden('value_4'))
    mass = torch.softmax(torch.randn(3, dtype=dtype), 0)
    score = rows.joint_scores(mass, probabilities, 0.7)
    assert torch.equal(score, golden('value_5'))
    score, passing = rows.joint_rule(0.9, 0.7)(mass, offers)
    expected_score, expected_passing = golden('value_6')
    assert torch.equal(score, expected_score)
    assert torch.equal(passing, expected_passing)
    for fill_zero in (False, True):
        tied = torch.tensor([0., 0.5, 0.5, 0., 0.])
        initial = torch.tensor([False, False, True, False, False])
        allowed = torch.tensor([True, True, True, False, True])
        for floor in (0, 2, 8):
            actual = rows.floor_passing(tied, initial, allowed, floor, fill_zero=fill_zero)
            expected = golden('value_7')
            assert torch.equal(actual, expected)
    for cap in (1, 4, 12):
        actual = rows.select_rows(score, passing, offers, cap, 12)
        expected = golden('value_8')
        assert all(torch.equal(a, b) for a, b in zip(actual, expected))


def test_policy_floor_ties_zero_mass_and_key_padding():
    from jetspec.methods.jetspec_plus_plus.method.rows import DeviceOffers, joint_rule, select_rows

    offers = DeviceOffers(torch.arange(12).reshape(3, 4), torch.ones(3, 4),
                          torch.ones(3, 4, dtype=torch.bool), torch.tensor([0, 3, 7]))
    score, passing = joint_rule(0.9, 1.0)(torch.ones(3) / 3, offers)
    requested, keys = select_rows(score, passing, offers, 5, 8)
    assert requested.item() == 8
    assert keys.tolist() == [[0, 0], [0, 1], [0, 2], [0, 3], [3, 4],
                             [-1, -1], [-1, -1], [-1, -1]]
    score, passing = joint_rule(0., 1.0)(torch.zeros(3), offers)
    requested, keys = select_rows(score, passing, offers, 8, 8)
    assert requested.item() == 0
    assert torch.equal(keys, torch.full((8, 2), -1))


@pytest.mark.parametrize('tau, temperature', [(-0.1, 1), (1.1, 1), (0, 0), (0, float('inf'))])
def test_joint_rule_rejects_invalid_parameters(tau, temperature):
    from jetspec.methods.jetspec_plus_plus.method.rows import joint_rule

    with pytest.raises(ValueError):
        joint_rule(tau, temperature)


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_greedy_accept_functions_match_golden(seed):
    from jetspec.methods.jetspec_plus_plus.method.accept import greedy_accept, greedy_accept_device

    torch.manual_seed(seed)
    logits = torch.randn(4, 5, 13, dtype=torch.float32)
    predictions = logits.argmax(-1)
    proposals = torch.cat((torch.randint(0, 13, (4, 1)), predictions[:, :-1]), 1)
    for row, mismatch in enumerate((0, 1, 3)):
        proposals[row, mismatch + 1] = (predictions[row, mismatch] + 1) % 13
    actual = greedy_accept_device(logits, proposals)
    expected = golden('value_9')
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))
    assert actual[0].tolist() == [0, 1, 3, 4]
    for return_predictions in (False, True):
        for proposal_rows in (None, proposals.tolist()):
            options = dict(return_predictions=return_predictions, proposal_rows=proposal_rows)
            actual = greedy_accept(logits, proposals, **options)
            expected = golden('value_10')
            assert actual == expected
            for a, b in zip(actual, expected):
                for actual_row, expected_row in zip(a, b):
                    assert torch.equal(torch.tensor(actual_row), torch.tensor(expected_row))
    # A device-only anchor proposal has no drafts and recovers its next argmax.
    actual = greedy_accept_device(logits[:, :1], proposals[:, :1])
    expected = golden('value_11')
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))


def _bundle_documents(seed=0):
    generator = torch.Generator().manual_seed(seed)
    coefficients = {name: {'value': float(torch.rand((), generator=generator))}
                    for name in ('a', 'b', 'c', 'argmax', 'top', 'pos')}
    thresholds = {
        'e04_kscore_blend': {'pooled': coefficients},
        'e04_rowrule_joint': {'pooled': {
            'tau': {'J1': {'value': 0.03}, 'pooled': {'value': 0.04}},
            'temperature': {'value': 0.7}}},
        'anchor_predictor': {'pooled': {'pooled': {'0.99': {'value': 1.0}}}},
    }
    manifest = {'source': 'target_slice', 'target_files': {
        'model.safetensors.index.json': '0' * 64, 'model-00001.safetensors': '1' * 64}}
    return thresholds, manifest


def _write_bundle(root, thresholds, manifest):
    (root / 'decoder').mkdir(exist_ok=True)
    (root / 'thresholds.json').write_text(json.dumps(thresholds))
    (root / 'decoder/manifest.json').write_text(json.dumps(manifest))


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_artifact_json_and_dataset_lookup_match_golden(monkeypatch, tmp_path, seed):
    """The new bundle wrapper loads metadata only; decoder weights belong to its caller."""
    from jetspec.methods.jetspec_plus_plus.artifacts import dataset_entry, load_heads_bundle, read_thresholds

    monkeypatch.delenv('JETSPEC_HEADS_FAST', raising=False)
    thresholds, manifest = _bundle_documents(seed)
    _write_bundle(tmp_path, thresholds, manifest)
    actual = read_thresholds(tmp_path)
    expected = golden('value_12')
    assert actual == expected
    for dataset in ('tiny', 'other'):
        coefficients = dataset_entry(actual['e04_kscore_blend'], dataset)
        expected_coefficients = golden('value_13')
        assert torch.equal(torch.tensor([entry['value'] for entry in coefficients.values()]),
                           torch.tensor([entry['value'] for entry in expected_coefficients.values()]))
    entries = {'tiny': {'value': 0.3}}
    assert dataset_entry(entries, 'tiny') == golden('value_14')
    with pytest.raises(ValueError, match='ambiguous'):
        dataset_entry({'tiny': {}, 'pooled': {}}, 'tiny')
    bundle = load_heads_bundle(tmp_path)
    assert bundle.thresholds == expected
    assert bundle.decoder_manifest == manifest
    # No cache: a changed or invalid local document must be read again.
    thresholds['e04_kscore_blend']['pooled']['a']['value'] += 1
    _write_bundle(tmp_path, thresholds, manifest)
    assert read_thresholds(tmp_path) == golden('value_12')


@pytest.mark.parametrize('missing', ['thresholds.json', 'decoder/manifest.json'])
def test_bundle_requires_both_documents(tmp_path, missing):
    from jetspec.methods.jetspec_plus_plus.artifacts import load_heads_bundle

    thresholds, manifest = _bundle_documents()
    (tmp_path / 'decoder').mkdir()
    if missing != 'thresholds.json':
        (tmp_path / 'thresholds.json').write_text(json.dumps(thresholds))
    if missing != 'decoder/manifest.json':
        (tmp_path / 'decoder/manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(FileNotFoundError):
        load_heads_bundle(tmp_path)


@pytest.mark.parametrize('fault', ['missing_blend', 'missing_joint', 'extra_coefficient',
                                  'missing_coefficient', 'nan', 'boolean', 'tau_range',
                                  'temperature_zero', 'tau_label', 'extra_joint_key',
                                  'scalar_coefficient', 'scalar_temperature'])
def test_thresholds_fail_closed(tmp_path, fault):
    from jetspec.methods.jetspec_plus_plus.artifacts import read_thresholds

    thresholds, manifest = _bundle_documents()
    coefficients = thresholds['e04_kscore_blend']['pooled']
    rule = thresholds['e04_rowrule_joint']['pooled']
    if fault == 'missing_blend':
        thresholds.pop('e04_kscore_blend')
    elif fault == 'missing_joint':
        thresholds.pop('e04_rowrule_joint')
    elif fault == 'extra_coefficient':
        coefficients['unknown'] = {'value': 1}
    elif fault == 'missing_coefficient':
        coefficients.pop('top')
    elif fault == 'nan':
        coefficients['top']['value'] = float('nan')
    elif fault == 'boolean':
        coefficients['top']['value'] = True
    elif fault == 'tau_range':
        rule['tau']['J1']['value'] = 1.1
    elif fault == 'temperature_zero':
        rule['temperature']['value'] = 0
    elif fault == 'tau_label':
        rule['tau']['unknown'] = {'value': 0.1}
    elif fault == 'scalar_coefficient':
        coefficients['a'] = 1.0
    elif fault == 'scalar_temperature':
        rule['temperature'] = 1.0
    else:
        rule['unknown'] = 0
    _write_bundle(tmp_path, thresholds, manifest)
    with pytest.raises(ValueError):
        read_thresholds(tmp_path)


@pytest.mark.parametrize('fault', ['file_source', 'unknown_source', 'unexpected_key',
                                  'missing_pins', 'missing_index', 'bad_sha', 'traversal',
                                  'absolute', 'pickle', 'no_shard', 'bad_fallback_sha'])
def test_decoder_manifest_fail_closed(tmp_path, fault):
    from jetspec.methods.jetspec_plus_plus.artifacts import load_heads_bundle

    thresholds, manifest = _bundle_documents()
    pins = manifest['target_files']
    if fault == 'file_source':
        manifest = {'source': 'file', 'path': 'decoder/checkpoint-0.pt', 'sha256': '0' * 64}
    elif fault == 'unknown_source':
        manifest['source'] = 'unknown'
    elif fault == 'unexpected_key':
        manifest['unknown'] = 1
    elif fault == 'missing_pins':
        manifest.pop('target_files')
    elif fault == 'missing_index':
        pins.pop('model.safetensors.index.json')
    elif fault == 'bad_sha':
        pins['model-00001.safetensors'] = 'invalid'
    elif fault == 'traversal':
        pins['../model.safetensors'] = '0' * 64
    elif fault == 'absolute':
        pins['/model.safetensors'] = '0' * 64
    elif fault == 'pickle':
        pins['model.pt'] = '0' * 64
    elif fault == 'no_shard':
        pins.pop('model-00001.safetensors')
    else:
        manifest['fallback_sha256'] = 'invalid'
    _write_bundle(tmp_path, thresholds, manifest)
    with pytest.raises(ValueError):
        load_heads_bundle(tmp_path)


@pytest.mark.parametrize('text', ['[]', '{}', '{"x": 1, "x": 2}', '{"x": NaN}',
                                 '{"x": Infinity}', 'not json'])
def test_invalid_json_is_refused(tmp_path, text):
    from jetspec.methods.jetspec_plus_plus.artifacts import read_thresholds

    (tmp_path / 'thresholds.json').write_text(text)
    with pytest.raises(ValueError):
        read_thresholds(tmp_path)


def test_bundle_metadata_symlink_cannot_escape_root(tmp_path):
    from jetspec.methods.jetspec_plus_plus.artifacts import read_thresholds

    bundle = tmp_path / 'bundle'
    bundle.mkdir()
    thresholds, _ = _bundle_documents()
    external = tmp_path / 'external.json'
    external.write_text(json.dumps(thresholds))
    (bundle / 'thresholds.json').symlink_to(external)
    with pytest.raises(ValueError, match='escapes'):
        read_thresholds(bundle)


@pytest.mark.parametrize('fault', ['draft', 'offer', 'nonfinite', 'coefficients', 'shape'])
def test_blend_rejects_invalid_scored_inputs(fault):
    from jetspec.methods.jetspec_plus_plus.method.kmass import blend

    drafts = torch.zeros(3, dtype=torch.long)
    logits = torch.zeros(4, 5)
    p_top1 = torch.full((4,), 0.5)
    probabilities = torch.full((4, 2), 0.5)
    sizes = torch.full((4,), 2)
    coefficients = dict(a=1., b=1., c=0., argmax=1., top=1., pos=1.)
    if fault == 'draft':
        drafts[0] = -1
    elif fault == 'offer':
        sizes[0] = 0
    elif fault == 'nonfinite':
        logits[0, 0] = float('nan')
    elif fault == 'coefficients':
        coefficients.pop('top')
    else:
        p_top1 = p_top1[:3]
    with pytest.raises(ValueError):
        blend(p_top1, logits, drafts, probabilities, sizes, coefficients=coefficients)

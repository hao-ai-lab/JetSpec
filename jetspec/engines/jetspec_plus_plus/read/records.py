"""Structured copies of the markers still printed by legacy serving reads."""
import json

MARKERS = ('CHAIN_ROUND', 'CHAIN_FALLBACK', 'EVAL_ROUND', 'RUN_MANIFEST',
           'SERVE_READ_PHASE_START', 'SERVE_READ_PHASE_END')


def read_fields(condition, stdout, config_read=None):
    fields = {marker: [] for marker in MARKERS}
    fields['RESOLVED_CONDITION'] = condition
    for line in stdout.splitlines():
        for marker in MARKERS:
            position = line.find(marker + ' ')
            if position < 0:
                continue
            payload = line[position + len(marker) + 1:]
            try:
                value, _ = json.JSONDecoder().raw_decode(payload)
            except ValueError:
                value = payload
            fields[marker].append(value)
            break
    if config_read is not None:
        fields.update(config=config_read['config'], canonical_json=config_read['canonical_json'],
                      config_sha256=config_read['config_sha256'])
    return fields

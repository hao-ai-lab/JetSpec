"""Eval records built from the serving launcher evidence."""
import json
from pathlib import Path


def write_eval_records(config, *, recipe_method, output, eval_values, process_wall_seconds):
    """Use the existing wrapper's reducers and the launcher's recorded evidence."""
    from jetspec.engines.jetspec_plus_plus.read.analysis import anatomy
    from jetspec.engines.jetspec_plus_plus.read.analysis import reduce_paper
    from jetspec.engines.jetspec_plus_plus.jetspec.profile import read_counters
    method, model = config['method'], config['model']
    output = Path(output)
    campaign = json.loads(Path(str(output) + '.campaign.json').read_text())
    condition = campaign['resolved_condition']
    for read in campaign['reads']:
        path = Path(read['output'])
        raw = json.loads(path.read_text())
        schedule = method.get('schedule', dict(j=0, r=0))
        raw['eval'] = dict(model='27B' if model['target']['family'] == 'hybrid' else '8B',
            j=schedule['j'], r=schedule['r'], budget=method['block_size'],
            rows=int(eval_values['rows']),
            drafting_mode=eval_values['drafting_mode'],
            j3_label=eval_values['j3_label'],
            dtype=model['target']['dtype'], placement=config['hardware']['placement'],
            process_wall_seconds=process_wall_seconds,
            served_wall_seconds=raw['metrics']['serve_read']['measured']['wall_seconds'],
            draft_delay_ms=float(eval_values['draft_delay_ms']),
            host_fast=eval_values['host_fast'], eos_token_ids=condition['eos_token_ids'])
        if recipe_method == 'jetspecpp':
            log = '\n'.join(marker + ' ' + json.dumps(value)
                            for marker in ('EVAL_ROUND', 'CHAIN_FALLBACK')
                            for value in read.get(marker, []))
            raw['eval']['counters'] = read_counters(log, raw['metrics'])
            raw['eval']['anatomy'] = anatomy(raw, log=log)
            raw['eval']['paper_metrics'] = reduce_paper(raw)
            refresh = eval_values['refresh_mode']
            if refresh != 'full':
                raw['eval']['refresh_mode'] = refresh
            capture = eval_values['capture_round_candidates']
            if capture != '0':
                from jetspec.engines.jetspec_plus_plus.read.eval_read import export_candidates
                from jetspec.engines.jetspec_plus_plus.read.eval_read import export_decoder_inputs
                raw['eval']['round_candidates'] = export_candidates(path, raw['metrics'])
                if capture == 'decoder':
                    raw['eval']['decoder_inputs'] = export_decoder_inputs(path, raw['metrics'])
        else:
            from jetspec.engines.jetspec_plus_plus.read.analysis import paper_metrics
            paper_metrics.EOS = condition['eos_token_ids']
            raw['eval']['paper_metrics'] = paper_metrics.reduce_read(raw, recipe_method)
        path.write_text(json.dumps(raw, indent=2) + '\n')

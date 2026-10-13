"""Ported from the JetSpec++ reference implementation."""
from collections import Counter
from dataclasses import replace
from types import SimpleNamespace
from pathlib import Path
import hashlib
import math

import torch

from jetspec.methods.jetspec.draft import Drafter
from .context import OneRoundAdaptedContext, build_protocol_hypothesis_contexts
from .method.kmass import blend
from .method.rows import joint_rule, select_rows, DeviceOffers
from .method.decoder.lac import lac_candidates
from .models.decoder import refresh_decoder_logits
from .round import JetSpecPPConfig, RoundDriver, resolve_tau


def anchor_offers(logits, drafts, *, q_hat, cap, after_miss=False):
    """Stable LAC offers from the served post-exclusion fp32 softmax."""
    logits = logits.float()
    rejected = torch.cat((drafts.reshape(-1), drafts.new_full((1,), -1)))
    if after_miss:
        rejected[0] = -1
    filtered = logits.clone()
    valid = rejected >= 0
    filtered[valid, rejected[valid]] = float('-inf')
    probabilities = filtered.softmax(-1).double()
    width = min(cap, logits.shape[-1])
    ids = torch.zeros((16, width), dtype=torch.long, device=logits.device)
    p64 = probabilities.new_zeros(16, width)
    kept = torch.zeros_like(ids, dtype=torch.bool)
    for row in range(16):
        excluded = [int(rejected[row])] if valid[row] else []
        selected, values = lac_candidates(probabilities[row], q_hat=q_hat, cap=cap, excluded=excluded)
        count = selected.numel()
        ids[row, :count], p64[row, :count], kept[row, :count] = selected, values, True
    return DeviceOffers(ids, p64, kept, torch.arange(16, device=logits.device))


class JetSpecPPDrafter(Drafter):
    """Sequential keyed rows for the lossless B16 torch verification seam."""
    def __init__(self, *, target, head, adapter, decoder, thresholds, config=None):
        self.config = config or JetSpecPPConfig()
        if len(head.target_layer_ids) != 5 or list(head.target_layer_ids) != adapter.target_layer_ids:
            raise ValueError('drafter and adapter must declare the same five target taps')
        self.target_layer_ids = list(head.target_layer_ids)
        self.driver = RoundDriver(target, head, adapter, decoder, self.config)
        self.thresholds = thresholds
        self.capture_rounds = False
        self.reset_cache()

    def reset_cache(self):
        self.row_cache, self.selected = {}, None
        self.staged = OneRoundAdaptedContext()
        self.decoder_tail = None
        self.records, self.round_results, self.counts = [], [], Counter()
        self.candidate_records, self.capture_prediction = [], None
        self.sequence = SimpleNamespace(taps=None)
        self.driver.counts.clear()
        self.driver.decoder._prefixes.clear()

    @torch.inference_mode()
    def _prepare(self, context_ids, exact, *, prompt_index=0):
        if exact is None or exact.shape[1] != context_ids.shape[1] - 1:
            raise ValueError('target taps must end immediately before the correction')
        self.sequence.taps = exact[0].reshape(-1, 5, self.driver.decoder.hidden_size)
        hybrid = getattr(self.driver.decoder, 'hybrid', False)
        # The reference casts before marker/tail assembly.
        exact = exact.to(next(self.driver.head.parameters()))
        if self.staged.start_position is not None:
            _, _, accepted = self.staged.consume(exact, context_ids)
            tail = self.staged.predicted_hidden[0, :accepted, -self.driver.decoder.hidden_size:]
            if not hybrid:
                self.decoder_tail = torch.cat((self.decoder_tail.to(tail), tail))[-4:].clone()
        elif hybrid:
            self.decoder_tail = None
        elif exact.shape[1] >= 5:
            boundary = exact.shape[1] - 4
            tokens = torch.cat((context_ids[:, boundary:exact.shape[1]],
                                context_ids[:, -1:].expand(1, 11)), 1)
            # The public D1 seed uses the trainer mask, as required by the scope.
            self.decoder_tail = self.driver.predict(exact[:, :boundary], tokens)[0, :4, -self.driver.decoder.hidden_size:]
        else:
            raise ValueError('admission requires at least five prompt tokens')
        after_miss = bool(self.records) and self.selected is None
        if self.selected is None:
            result = self.driver.run(context_ids, exact, prompt_index=prompt_index, admission=True,
                                     sequence=self.sequence)
            self.counts['redrafts' if after_miss else 'admission'] += 1
        else:
            result = self.selected
            if int(result.ids[0, 0]) != int(context_ids[0, -1]):
                raise RuntimeError('selected row anchor differs from the verified correction')
        self.selected = None
        if self.capture_rounds and (not self.records or after_miss):
            from .capture import candidate_record
            r = len(self.records)
            if after_miss:
                previous = self.records[-1]
                keys = [[previous['k_true'], previous['correction']]]
            else:
                keys = [[k, int(result.ids[0, 0])] for k in range(16)]
            record = candidate_record(result, [result] * len(keys), keys,
                                      seq_id=prompt_index, round_id=r,
                                      kind='redraft' if after_miss else 'admission',
                                      requested=len(keys), after_miss=after_miss)
            if after_miss:
                record.update(k_true=keys[0][0], anchor_true=keys[0][1],
                              true_anchor=keys[0][1], correction=keys[0][1])
            self.candidate_records.append(record)
        length = exact.shape[1] if hybrid else exact.shape[1] - 4
        self.driver.decoder._prefix(self.sequence, length)
        logits, _, _ = refresh_decoder_logits(
            self.driver.decoder, self.sequence, length=length, tail_hidden=self.decoder_tail,
            tail_counts=[0 if hybrid else 4], block_hidden=result.predicted, slot_counts=[16])
        # The reference joint rule scores the whole top-32 pool, independently of LAC coverage.
        offers = anchor_offers(logits, result.ids[0, 1:],
                               q_hat=1.0 if self.config.score_all_anchors else self.thresholds['anchor_q_hat'],
                               cap=self.config.anchor_cap,
                               after_miss=after_miss and not self.config.reject_after_miss)
        sizes = offers.valid.sum(-1)
        prediction = blend(result.p_top1[0], logits, result.ids[0, 1:],
                           offers.probabilities, sizes, coefficients=self.thresholds['blend'])
        mass = prediction[1:]
        if after_miss:
            mass = torch.zeros_like(mass)
            mass[0] = 1
            offers = DeviceOffers(offers.ids, offers.probabilities,
                                  offers.valid & (offers.ks[:, None] == 0), offers.ks)
        tau = resolve_tau(self.thresholds['row_tau'], result.label)
        score, passing = joint_rule(tau, self.thresholds['row_temperature'],
                                   floor=self.config.rows_min,
                                   fill_zero=self.config.rows_fill_zero)(mass, offers)
        requested, keys = select_rows(score, passing, offers, self.config.rows_cap, self.config.rows_cap)
        keys = keys[:min(int(requested), self.config.rows_cap)]
        if not keys.numel():
            raise ValueError('row policy published no keyed rows')
        self.row_cache = {}
        for k, anchor in keys.tolist():
            tail_count = k + 1
            context, _, _ = build_protocol_hypothesis_contexts(
                exact, predicted_hidden=result.predicted, k_hypotheses=[tail_count],
                adapted_position_embedding=self.driver.head.adapted_position_embedding, block_size=16)
            row = self.driver.run(context_ids, exact, prompt_index=prompt_index, anchor=anchor,
                                  target_context=context, tail_tokens=result.ids[:, :tail_count],
                                  tail_hidden=result.predicted[:, :tail_count], sequence=self.sequence)
            self.row_cache[k, anchor] = row
        if self.capture_rounds:
            from .capture import candidate_record
            if not after_miss:
                self.capture_prediction = prediction.clone()
            self.candidate_records.append(candidate_record(
                result, list(self.row_cache.values()), keys.tolist(), seq_id=prompt_index,
                round_id=len(self.records) + 1, kind='proposal', requested=int(requested),
                after_miss=after_miss, decoder_logits=logits, prediction=self.capture_prediction))
        self.staged.stage(start_position=exact.shape[1], token_ids=result.ids, predicted_hidden=result.predicted)
        self.records.append(dict(label=result.label, rows_requested=int(requested),
                                 keys=keys.tolist(), hit=None))
        self.current = result
        self.round_results.append(result)
        return result

    @torch.inference_mode()
    def propose_logits(self, context_ids, depth, target_hidden=None, **kwargs):
        if depth != 15:
            raise ValueError('JetSpec++ requires B16 and depth 15')
        if not self.records and kwargs.get('seed') is not None:
            self.config = replace(self.config, seed=int(kwargs['seed']))
            self.driver.config = self.config
        result = self._prepare(context_ids, target_hidden, prompt_index=kwargs.get('prompt_index', 0))
        logits = result.logits.new_full((1, 15, result.logits.shape[-1]), float('-inf'))
        return logits.scatter_(2, result.ids[:, 1:, None], 0.0)

    @torch.inference_mode()
    def propose(self, context_ids, k, target_hidden=None, **kwargs):
        self.propose_logits(context_ids, k, target_hidden=target_hidden, **kwargs)
        return self.current.ids[0, 1:].clone()

    def observe(self, acc, correction):
        key = int(acc), int(correction)
        self.selected = self.row_cache.get(key)
        hit = self.selected is not None
        self.counts['hits' if hit else 'misses'] += 1
        self.records[-1].update(hit=hit, k_true=key[0], correction=key[1])
        if self.capture_rounds:
            # The next publication is selected by this verification's outgoing truth.
            self.candidate_records[-1].update(k_true=key[0], anchor_true=key[1],
                                             true_anchor=key[1], correction=key[1],
                                             k_hit=any(k == key[0] for k, _ in self.row_cache),
                                             anchor_hit=hit, outcome_miss=not hit)


def build_drafter(*, llm, config, drafter_checkpoint, adapter_artifact, heads_bundle, dataset=None,
                  draft_dtype=None):
    """Construct local artifacts and verify target-slice pins.

    Mixed precision loads a separate decoder at its checkpoint dtype; the
    checkpoint slice must already match draft_dtype, as required by the reference.
    """
    from .artifacts import load_heads_bundle, dataset_entry
    from .models.drafter import load_drafter
    from .models.adapter import load_adapter
    from .models.decoder import AnchorDecoder, load_target_tensors

    bundle = load_heads_bundle(heads_bundle)
    target_root = Path(llm.model.config._name_or_path).resolve(strict=True)
    for name, expected in bundle.decoder_manifest['target_files'].items():
        path = (target_root / name).resolve(strict=True)
        if not path.is_relative_to(target_root):
            raise ValueError(f'target-file pin escapes the target directory: {name}')
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError(f'target-file SHA256 mismatch: {name}')
    dtype = llm.dtype if draft_dtype is None else draft_dtype
    hybrid = llm.model.config.model_type == 'qwen3_5_text'
    head = load_drafter(drafter_checkpoint, device=llm.device, dtype=dtype,
                        model_family='27b' if hybrid else '8b')
    adapter = load_adapter(adapter_artifact, device=llm.device, dtype=dtype)
    # C1 requires projection_weight=0: its validated converter is provenance only, as in the reference.
    if (not hybrid and not torch.equal(head.rotary_emb.inv_freq, llm.model.model.rotary_emb.inv_freq)):
        raise ValueError('drafter and target rotary frequencies differ')
    if (head.config.num_target_layers != llm.model.config.num_hidden_layers
            or head.config.hidden_size != llm.model.config.hidden_size
            or head.config.vocab_size != llm.model.config.vocab_size):
        raise ValueError('drafter geometry differs from target config')
    if hybrid and head.config.rope_theta != llm.model.config.rope_parameters['rope_theta']:
        raise ValueError('drafter and target rotary theta differ')
    raw = bundle.thresholds
    if dataset is None:
        names = set(raw['e04_kscore_blend']) & set(raw['e04_rowrule_joint'])
        if len(names) != 1:
            raise ValueError('a dataset is required for a bundle with multiple calibration entries')
        dataset = next(iter(names))
        if dataset == 'pooled':
            dataset = '_default'
    coefficients = dataset_entry(raw['e04_kscore_blend'], dataset)
    rule = dataset_entry(raw['e04_rowrule_joint'], dataset)
    anchor = dataset_entry(raw['anchor_predictor']['pooled'], dataset)[str(config.anchor_coverage)]['value']
    if type(anchor) not in (int, float) or not math.isfinite(anchor) or not 0 <= anchor <= 1:
        raise ValueError('anchor LAC threshold must be finite and in [0,1]')
    thresholds = dict(blend={key: entry['value'] for key, entry in coefficients.items()},
                      anchor_q_hat=float(anchor), row_tau=rule['tau'],
                      row_temperature=rule['temperature']['value'])
    mixed = dtype != llm.dtype
    if hybrid:
        if mixed:
            raise ValueError('hybrid target and draft artifacts must share the checkpoint dtype')
        from .models.hybrid import HybridAnchorDecoder
        decoder = HybridAnchorDecoder.from_checkpoint(llm.model, tap_ids=head.target_layer_ids,
            file_pins=bundle.decoder_manifest['target_files'])
    else:
        decoder = (AnchorDecoder.from_checkpoint(llm.model, tap_ids=head.target_layer_ids,
                   device=llm.device, dtype=dtype, file_pins=bundle.decoder_manifest['target_files'])
               if mixed else AnchorDecoder.from_target(llm.model, tap_ids=head.target_layer_ids))
    drafter = JetSpecPPDrafter(target=llm.model, head=head, adapter=adapter, decoder=decoder,
                             thresholds=thresholds, config=config)
    if mixed:
        # The reference owns bf16 release embeddings and LM head;
        # this reader's target checkpoint supplies those same target tensors.
        state = load_target_tensors(target_root, ('model.embed_tokens.weight', 'lm_head.weight'),
                                    prefixes=('model.embed_tokens.', 'lm_head.'))
        vocab, width = state['lm_head.weight'].shape
        for name, module, key in (
                ('embed_tokens', torch.nn.Embedding(vocab, width, dtype=torch.bfloat16),
                 'model.embed_tokens.weight'),
                ('lm_head', torch.nn.Linear(width, vocab, bias=False, dtype=torch.bfloat16), 'lm_head.weight')):
            module.load_state_dict({'weight': state[key]}, strict=True)
            # The reference applies an optional fp32 cast after release load.
            setattr(drafter.driver, name, module.to(device=llm.device, dtype=dtype).eval().requires_grad_(False))
    drafter.artifact_pins = dict(bundle.decoder_manifest['target_files'])
    return drafter

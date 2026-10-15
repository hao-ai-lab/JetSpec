"""Serving cutoffs and the learned draft clock. No method imports serving policy."""
import json
import math
import time
from jetspec.engines.jetspec_plus_plus.draft.method.schedule import CHAIN_SEGMENTS, Schedule, chain_variants
from jetspec.engines.jetspec_plus_plus.jetspec.config_values import config_values
from jetspec.engines.jetspec_plus_plus.legacy_env import environment_name

class DeadlineBudget:
    """Host-clock deadline for the round-ahead draft (JETSPEC_DRAFT_DEADLINE=1).

    The budget is the running median of the target's round length, read on the draft rank as
    the interval between consecutive request arrivals over rounds where the draft rank was
    idle before the next arrival (the draft was hidden, so the interval is the target's own
    round). Per-pass costs are the running medians of the host time from each refinement
    pass's top to the publish, over rounds that ran every pass untruncated."""


    uncut = False

    @staticmethod
    def wait_ms(value):
        return value

    @staticmethod
    def redraft_deadline(start, milliseconds):
        if not milliseconds > 0:
            raise ValueError("miss redraft valve must be positive (inf disables the cutoff)")
        if math.isinf(milliseconds):
            return float('inf')
        return start + milliseconds / 1000

    @staticmethod
    def switch_length(value):
        return None if value is None else int(value)


    @staticmethod
    def ar_step(kind, miss_mode):
        return kind == 'late' or (kind == 'miss' and miss_mode == 'ar')

    @staticmethod
    def late(valid):
        return not valid

    @staticmethod
    def deadline_enabled(environ):
        value = environ.get(environment_name('engine', 'engine_settings.draft_deadline'), '0')
        if value not in ('0', '1'):
            raise ValueError('JETSPEC_DRAFT_DEADLINE must be 0 or 1')
        return value == '1'



    @staticmethod
    def row_cap(cap, capacity):
        if not 1 <= cap <= capacity:
            raise ValueError('JETSPEC_ROWS_RULE_CAP must be in 1..JETSPEC_ROWS_CAP')
        return cap

    @staticmethod
    def required_redraft(environ):
        return (environment_name('engine', 'engine_settings.chain_miss_redraft')
                if environ.get(environment_name('engine', 'engine_settings.chain_graph')) == 'seg'
                else environment_name('method', 'method_settings.miss_redraft'))


    @staticmethod
    def fallback_choice():
        return 1, False

    WINDOW = 64
    MIN_SAMPLES = 8
    OUTLIER_FACTOR = 3.0

    def __init__(self, *, run_config=None):
        values = config_values(run_config)
        self.margin_ms = float(values.get('engine.engine_settings.draft_deadline_margin_ms', '1.0'))
        raw_tau = values.get('engine.engine_settings.chain_tau_table')
        self._adapter_refresh = values.get('method.method_settings.adapter_refresh', 'off')
        self.chain_tau = None
        self.chain_tau_cap = False
        if raw_tau is not None:
            try:
                table = json.loads(raw_tau)
            except json.JSONDecodeError as error:
                raise ValueError('JETSPEC_CHAIN_TAU_TABLE must be JSON') from error
            labels = {str(Schedule(j, int(refresh))) for j, refresh, _ in chain_variants()}
            cap_labels = {f'{label}-C{cap}' for label in labels for cap in (4, 8)}
            if (not isinstance(table, dict) or set(table) not in (labels, cap_labels)
                    or any(not isinstance(value, (int, float)) or isinstance(value, bool)
                           or not math.isfinite(value) or value <= 0
                           for value in table.values())):
                raise ValueError('JETSPEC_CHAIN_TAU_TABLE needs five or ten positive finite tau values')
            self.chain_tau_cap = set(table) == cap_labels
            self.chain_tau = {name: float(value) for name, value in table.items()}
        self.intervals = []
        self.remaining = {}
        self.stage_remaining = {}
        self.stop_tails = []
        self.arrival = None
        self.start = None
        self.idle_since = None
        self.clean = False
        self.tops = {}
        self.stage_tops = {}
        self.limit = None
        self.decision = None
        self.chain_times = {}
        self.chain_tails = {}
        self.chain_published_passes = None
        self.eager_published_passes = None

    @staticmethod
    def _push(values, value, window):
        values.append(float(value))
        if len(values) > window:
            del values[0]

    def arrive(self, now, *, clean):
        """A request arrived at ``now``; ``clean`` = the finished round was an ordinary one."""
        if (self.arrival is not None and self.idle_since is not None and self.clean
                and now - self.idle_since > 2e-4):
            self._push(self.intervals, (now - self.arrival) * 1000.0, self.WINDOW)
        self.arrival = self.start = now
        self.idle_since = None
        self.clean = clean
        self.tops = {}
        self.stage_tops = {}
        self.limit = None
        self.decision = None
        self.chain_published_passes = None
        self.eager_published_passes = None


    def budget_ms(self):
        if len(self.intervals) < self.MIN_SAMPLES:
            return None
        ordered = sorted(self.intervals)
        # Use the fast cluster as the reference. Cold target loads and compile
        # stalls can be much longer than the steady verify round.
        pivot = ordered[len(ordered) // 4]
        steady = [value for value in ordered if value <= pivot * self.OUTLIER_FACTOR]
        if len(steady) < self.MIN_SAMPLES:
            return None
        return steady[len(steady) // 2] - self.margin_ms

    def _median(self, index):
        values = self.remaining.get(index)
        if not values or len(values) < self.MIN_SAMPLES:
            return None
        ordered = sorted(values)
        return ordered[len(ordered) // 2]

    @staticmethod
    def _p95(values):
        if len(values) < DeadlineBudget.MIN_SAMPLES:
            return None
        ordered = sorted(values)
        return ordered[(95 * len(ordered) + 99) // 100 - 1]



    def launch_chain(self, now, key, *, host_pre_ms=0.0):
        """Choose learned throughput, or the J2-R0 primary and J1-R0 tail."""
        variants = chain_variants()
        primary = variants[2]
        default_ladder = (variants[2], variants[4])
        self.limit = 4
        self.chain_published_passes = 3
        budget = self.budget_ms()
        tail = self._p95(self.chain_tails.get(key, ()))
        costs = {name: self._p95(self.chain_times.get((key, name), ()))
                 for name in CHAIN_SEGMENTS}
        if costs['B1n'] is None:
            costs['B1n'] = costs['B1r']  # conservative until no-refresh is sampled
        if tail is None or any(costs[name] is None for name in primary[2]):
            return (2, False)
        if self.chain_tau is not None:
            lane = self._p95(self.intervals)
            if lane is None:
                return (2, False)
            scored = []
            for j, refresh, names in variants:
                if any(costs[name] is None for name in names):
                    continue
                chain_ms = sum(costs[name] for name in names)
                label = f'J{j}-R{int(refresh)}'
                tau_key = f'{label}-C{key.R}' if self.chain_tau_cap else label
                score = self.chain_tau[tau_key] / (max(lane, chain_ms) + tail)
                scored.append((score, j, refresh, chain_ms, label))
            _, j, refresh, chain_ms, label = max(scored, key=lambda item: item[0])
            self.chain_published_passes = j + 1
            self.decision = dict(J_round=j, refresh=int(refresh), policy='throughput',
                                 config=label, lane_ms=lane, chain_ms=chain_ms,
                                 handoff_ms=tail)
            return (j, refresh)
        if budget is None:
            return (2, False)
        elapsed = 0.0 if self.start is None else (now - self.start) * 1000.0
        elapsed += host_pre_ms
        for j, refresh, names in default_ladder:
            if any(costs[name] is None for name in names):
                continue
            predicted = elapsed + sum(costs[name] for name in names) + tail
            if predicted <= budget:
                self.chain_published_passes = j + 1
                self.decision = dict(J_round=j, refresh=int(refresh), elapsed_ms=elapsed,
                                     predicted_ms=predicted, budget_ms=budget)
                return (j, refresh)
        self.decision = dict(stopped_before=0, elapsed_ms=elapsed,
                             budget_ms=budget, deadline_miss=1, stage='before_j0')
        self.chain_published_passes = 0
        raise DraftDeadlineMiss('before_j0')

    def launch_chain_capacity(self, now, keys, *, host_pre_ms=0.0):
        """Choose the J/refresh/row-cap triple from trained p95 clocks and tau."""
        by_cap = {key.R: key for key in keys}
        if set(by_cap) != {4, 8}:
            raise ValueError('chain capacity policy requires prewarmed r4 and r8 keys')
        primary = by_cap[8]
        if not self.chain_tau_cap:
            choice = self.launch_chain(now, primary, host_pre_ms=host_pre_ms)
            if self.decision is not None:
                self.decision['row_cap'] = 8
            return primary, choice
        lane = self._p95(self.intervals)
        variants = chain_variants(capacity=True)
        scored = []
        if lane is not None:
            for cap, key in by_cap.items():
                tail = self._p95(self.chain_tails.get(key, ()))
                costs = {name: self._p95(self.chain_times.get((key, name), ()))
                         for name in CHAIN_SEGMENTS}
                if costs['B1n'] is None:
                    costs['B1n'] = costs['B1r']
                if tail is None:
                    continue
                for j, refresh, names in variants:
                    if any(costs[name] is None for name in names):
                        continue
                    chain_ms = sum(costs[name] for name in names)
                    label = f'J{j}-R{int(refresh)}'
                    score = self.chain_tau[f'{label}-C{cap}'] / (max(lane, chain_ms) + tail)
                    scored.append((score, cap == 8, key, j, refresh, chain_ms, tail, label))
        if not scored:
            self.chain_published_passes = 3
            self.decision = dict(J_round=2, refresh=0, row_cap=8,
                                 policy='untrained-cap8')
            return primary, (2, False)
        score, _, key, j, refresh, chain_ms, tail, label = max(
            scored, key=lambda item: (item[0], item[1]))
        self.chain_published_passes = j + 1
        self.decision = dict(J_round=j, refresh=int(refresh), row_cap=key.R,
                             policy='throughput', config=label, score=score,
                             lane_ms=lane, chain_ms=chain_ms, handoff_ms=tail)
        return key, (j, refresh)

    def launch_chain_fixed(self, config, key):
        """Pin a paper-cell chain choice independently of learned clocks."""
        from jetspec.engines.jetspec_plus_plus.draft.method.schedule import Schedule
        schedule = Schedule.parse(config)
        j, refreshes = schedule.j, schedule.r
        refresh = refreshes if refreshes > 1 else bool(refreshes)
        self.chain_published_passes = j + 1
        self.decision = dict(J_round=j, refresh=int(refresh), row_cap=key.R,
                             policy='fixed', config=config)
        return j, refresh

    def checkpoint(self, stage):
        """Reserve time for J0 and publication before starting another draft stage."""
        now = time.perf_counter()
        self.stage_tops.setdefault(stage, now)
        budget = self.budget_ms()
        samples = self.stage_remaining.get(stage, ())
        if budget is None or len(samples) < self.MIN_SAMPLES or self.start is None:
            return
        ordered = sorted(samples)
        minimum_remaining = ordered[len(ordered) // 2]
        # Full-round samples include later refinements. Reserve only J0 plus the
        # publish tail: a late optional J pass must not force an empty candidate.
        if self.limit is not None and self.limit > 1:
            later = self._median(1)
            tail = self._median(self.limit - 1)
            if later is None or tail is None:
                return
            minimum_remaining = max(0.0, minimum_remaining - later) + tail
        elapsed = (now - self.start) * 1000.0
        predicted = elapsed + minimum_remaining
        if predicted > budget:
            self.tops[0] = now
            self.decision = dict(stopped_before=0, elapsed_ms=elapsed,
                                 predicted_ms=predicted, budget_ms=budget,
                                 deadline_miss=1, stage=stage)
            raise DraftDeadlineMiss(stage)

    def truncate_if(self, limit):
        """The packed_refinement hook: True before pass ``index`` stops the refinement there."""
        self.limit = limit

        def check(index):
            now = time.perf_counter()
            self.tops[index] = now
            if index == 0:
                self.checkpoint('before_j0')
                return False
            if self.start is None:
                return False
            budget = self.budget_ms()
            if budget is None:
                return False
            # Run pass ``index`` only if stopping at the next top still publishes in time:
            # the full-round remaining cost from this top, minus that from the next top,
            # plus the remaining cost of a stop (learned from truncated rounds; until then
            # the full remaining cost from the last top, an upper bound).
            here, last = self._median(index), self._median(limit - 1)
            if len(self.stop_tails) >= self.MIN_SAMPLES:
                last = sorted(self.stop_tails)[len(self.stop_tails) // 2]
            after = 0.0 if index + 1 >= limit else self._median(index + 1)
            if here is None or last is None or after is None:
                return False
            elapsed = (now - self.start) * 1000.0
            predicted = elapsed + (here - after) + (0.0 if index + 1 >= limit else last)
            if predicted <= budget:
                return False
            self.decision = dict(stopped_before=index, elapsed_ms=elapsed,
                                 predicted_ms=predicted, budget_ms=budget)
            return True
        return check

    def published(self, now):
        """The round's rows are on the wire; learn the pass costs from an untruncated round."""
        if self.chain_published_passes is not None:
            passes_run = self.chain_published_passes
        elif self.decision is not None:
            passes_run = self.decision['stopped_before']
        else:
            passes_run = max(self.tops) + 1 if self.tops else (1 if self.limit else None)
        budget = self.budget_ms()
        draft_ms = None if self.start is None else (now - self.start) * 1000.0
        # Do not teach the pass/stage predictors from the cold start or a
        # compiler stall. Wait for a stable verify budget before any learning.
        steady = (budget is not None and draft_ms is not None
                  and draft_ms <= (budget + self.margin_ms) * self.OUTLIER_FACTOR)
        if (self.chain_published_passes is None and steady and self.decision is None
                and self.limit is not None and passes_run == self.limit):
            for index, top in self.tops.items():
                self._push(self.remaining.setdefault(index, []), (now - top) * 1000.0, self.WINDOW)
        if (self.chain_published_passes is None and steady and self.decision is None
                and self.limit is not None and passes_run == self.limit):
            for stage, top in self.stage_tops.items():
                self._push(self.stage_remaining.setdefault(stage, []),
                           (now - top) * 1000.0, self.WINDOW)
        if (self.chain_published_passes is None and steady and self.decision is not None
                and not self.decision.get('deadline_miss')):
            stop = self.tops[self.decision['stopped_before']]
            self._push(self.stop_tails, (now - stop) * 1000.0, self.WINDOW)
        self.idle_since = now
        cutoff = self.decision is not None and "stopped_before" in self.decision
        record = dict(published_pass=(passes_run if self.eager_published_passes is None else
                                      self.eager_published_passes),
                      truncated=int(cutoff if self.chain_published_passes is not None
                                    else self.decision is not None),
                      deadline_miss=int(bool(self.decision and self.decision.get('deadline_miss'))),
                      budget_ms=budget,
                      published_ns=int(now * 1e9),
                      draft_ms=draft_ms)
        if self.decision is not None:
            record.update(self.decision)
            if cutoff:
                index = self.decision['stopped_before']
                refresh = self._adapter_refresh
                record['cutoff_stage'] = record.get('cutoff_stage') or (
                    'refresh-1' if index == 1 and refresh in ('after_j0', 'after_j0_j1') else
                    'refresh-2' if index == 2 and refresh == 'after_j0_j1' else f'J{index}')
        return record


class DraftDeadlineMiss(Exception):
    """No complete first pass can be published within the learned verify budget."""


class NullBudget:
    """Run the method without serving cutoffs; capacity remains a caller contract."""
    uncut = True

    @staticmethod
    def wait_ms(value):
        return float('inf')

    @staticmethod
    def redraft_deadline(start, milliseconds):
        return None

    @staticmethod
    def switch_length(value):
        return None


    @staticmethod
    def ar_step(kind, miss_mode):
        return False

    @staticmethod
    def late(valid):
        if not valid:
            raise RuntimeError('uncut verify requires the exact sequence/round candidate')
        return False




    row_cap = staticmethod(DeadlineBudget.row_cap)
    required_redraft = staticmethod(DeadlineBudget.required_redraft)


    @staticmethod
    def fallback_choice():
        raise RuntimeError('uncut draft cannot shorten a configured schedule')



def budget_type(environ=None):
    """Compatibility selector; live ranks read the installed Engine's cutoffs."""
    from jetspec.engines.jetspec_plus_plus.engine import current_engine
    return current_engine(environ).budget


# Historical import is an alias to the single implementation.
DeadlineClock = DeadlineBudget

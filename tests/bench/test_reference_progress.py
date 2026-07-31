from bench.reference import benchmark


def test_timed_samples_reports_rank_local_and_global_progress(monkeypatch, capsys):
    monkeypatch.setattr(benchmark.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(benchmark, "_barrier", lambda world_size: None)

    outputs, latency = benchmark._timed_samples(
        lambda prompt: {"prompt": prompt},
        ["first", "second"],
        8,
        rank=2,
        total_samples=20,
        label="spec:accum_logp",
    )

    assert outputs == [{"prompt": "first"}, {"prompt": "second"}]
    assert latency >= 0
    stdout = capsys.readouterr().out
    assert "[progress][spec:accum_logp][rank 2/8] starting 2 local samples" in stdout
    assert "local 1/2 global_sample 3/20" in stdout
    assert "local 2/2 global_sample 11/20" in stdout

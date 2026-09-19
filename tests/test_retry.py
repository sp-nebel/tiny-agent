"""call_ollama's backoff: which failures are retried, how often, and that a
context overflow or a 4xx never is. No server: urlopen is scripted."""
import io
import urllib.error

import pytest

import config
import ollama


def http_error(code, body=""):
    return urllib.error.HTTPError("http://x/api/chat", code, "err", {}, io.BytesIO(body.encode()))


@pytest.fixture
def script(monkeypatch):
    sleeps = []
    monkeypatch.setattr(ollama.time, "sleep", sleeps.append)
    monkeypatch.setattr(config, "THINK", False)

    def install(errors):
        errors = list(errors)
        attempts = []

        def fake_urlopen(req, timeout=None):
            attempts.append(req)
            raise errors.pop(0) if len(errors) > 1 else errors[0]
        monkeypatch.setattr(ollama.urllib.request, "urlopen", fake_urlopen)
        return attempts
    return install, sleeps


def test_refused_then_bad_request_retries_then_surfaces(script):
    install, sleeps = script
    refused = urllib.error.URLError(ConnectionRefusedError())
    attempts = install([refused, refused, http_error(400, "bad")])
    with pytest.raises(urllib.error.HTTPError):
        ollama.call_ollama([{"role": "user", "content": "x"}])
    assert len(attempts) == 3 and len(sleeps) == 2
    # Same bytes every time: a retry must not cost the prefix cache.
    assert len({a.data for a in attempts}) == 1


def test_gives_up_after_retry_max_with_capped_backoff(script, monkeypatch):
    install, sleeps = script
    monkeypatch.setattr(config, "RETRY_MAX", 5)
    install([http_error(503)])
    with pytest.raises(urllib.error.HTTPError):
        ollama.call_ollama([{"role": "user", "content": "x"}])
    assert len(sleeps) == 5
    for n, s in enumerate(sleeps):
        base = min(config.RETRY_MAX_DELAY, config.RETRY_BASE_DELAY * 2 ** n)
        assert base * 0.8 <= s <= base * 1.2


def test_unwrapped_reset_is_retried(script):
    install, sleeps = script
    import http.client
    install([http.client.RemoteDisconnected("gone"), http_error(400)])
    with pytest.raises(urllib.error.HTTPError):
        ollama.call_ollama([{"role": "user", "content": "x"}])
    assert len(sleeps) == 1


@pytest.mark.parametrize("err", [
    http_error(500, '{"error":"input exceeds the context length"}'),
    http_error(404, "model not found"),
])
def test_never_retried(script, err):
    install, sleeps = script
    install([err])
    with pytest.raises(urllib.error.HTTPError):
        ollama.call_ollama([{"role": "user", "content": "x"}])
    assert sleeps == []

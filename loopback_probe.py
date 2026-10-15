"""Loopback probe for PR 671 echo cancellation on real PortAudio callbacks.

Needs a digital loopback (output fed straight back to an input): VB-CABLE on Windows
("CABLE Input" plays, "CABLE Output" records) or Pro Tools Audio Bridge/BlackHole on macOS.
Nothing reaches a speaker.

For each host API it reports, as JSON lines:
  timestamps  - are inputBufferAdcTime/outputBufferDacTime reported, and how regular are they
  lead        - when the looped-back signal actually arrives, relative to the PR's predicted
                playback time (positive = reference leads its echo, which AEC3 needs)
  canceller   - echo reduction and microphone sample loss through the PR's own code paths:
                the in-process streams (flag on vs off) and the spawned audio worker

Run from a checkout of the PR:  python loopback_probe.py --src src --out "CABLE Input" --in "CABLE Output"
"""

from __future__ import annotations

import argparse
import json
import platform
import queue
import sys
import time
from pathlib import Path

import numpy as np

RATE = 16000
BLOCK = 1024


def emit(**record):
    print(json.dumps(record, default=float), flush=True)


def load_speech(src: Path, seconds: float) -> np.ndarray:
    from scipy.io import wavfile

    # The package's TTS/ref_audio.wav (MP3 in WAV), resampled to 16 kHz PCM once.
    _sr, data = wavfile.read(Path(__file__).with_name("speech16k.wav"))
    speech = data.astype(np.float64) / 32767
    reps = int(np.ceil(seconds * RATE / len(speech)))
    speech = np.tile(speech, reps)[: int(seconds * RATE)]
    return (speech * 12000).astype(np.int16)


def find_devices(sd, hostapi_index: int, out_pat: str, in_pat: str):
    out_idx = in_idx = None
    for i, d in enumerate(sd.query_devices()):
        if d["hostapi"] != hostapi_index:
            continue
        if out_idx is None and out_pat.lower() in d["name"].lower() and d["max_output_channels"] > 0:
            out_idx = i
        if in_idx is None and in_pat.lower() in d["name"].lower() and d["max_input_channels"] > 0:
            in_idx = i
    return in_idx, out_idx


def extra_for(sd, hostapi_name: str):
    if "WASAPI" in hostapi_name:
        return sd.WasapiSettings(auto_convert=True)
    return None


def run_streams(sd, ac, in_idx, out_idx, hostapi_name, signal, seconds):
    """Play `signal` and record, logging every callback's time_info like the PR does."""
    out_log, in_log, chunks = [], [], []
    state = {"pos": 0}
    raw = signal.tobytes()

    def cb_out(outdata, frames, time_info, status):
        pos = state["pos"]
        piece = raw[pos * 2 : (pos + frames) * 2]
        outdata[: len(piece)] = piece
        outdata[len(piece) :] = b"\x00" * (len(outdata) - len(piece))
        out_log.append((time_info.outputBufferDacTime, time_info.currentTime, time.monotonic(), pos, frames, bool(status)))
        state["pos"] = pos + frames

    def cb_in(indata, frames, time_info, status):
        in_log.append((time_info.inputBufferAdcTime, time_info.currentTime, time.monotonic(), frames, bool(status)))
        chunks.append(bytes(indata))

    extra = extra_for(sd, hostapi_name)
    kw = dict(samplerate=RATE, channels=1, dtype="int16", blocksize=BLOCK)
    istream = sd.RawInputStream(device=in_idx, callback=cb_in, extra_settings=extra, **kw)
    ostream = sd.RawOutputStream(device=out_idx, callback=cb_out, extra_settings=extra, **kw)
    in_off = ac._stream_clock_offset(istream)
    out_off = ac._stream_clock_offset(ostream)
    latencies = (istream.latency, ostream.latency)
    istream.start()
    ostream.start()
    time.sleep(seconds)
    ostream.stop()
    istream.stop()
    ostream.close()
    istream.close()
    return out_log, in_log, chunks, in_off, out_off, latencies


def regularity(times, frames):
    t = np.array(times, dtype=float)
    if len(t) < 3:
        return {}
    nominal = np.array(frames[:-1], dtype=float) / RATE
    dev = np.diff(t) - nominal
    return {
        "zero_fraction": float(np.mean(t == 0)),
        "step_dev_median_ms": float(np.median(dev) * 1000),
        "step_dev_max_abs_ms": float(np.max(np.abs(dev)) * 1000),
        "steps_off_by_10ms_or_more": int(np.sum(np.abs(dev) >= 0.010)),
        "steps": int(len(dev)),
    }


def probe_hostapi(sd, ac, src, hostapi_index, hostapi_name, out_pat, in_pat, seconds):
    in_idx, out_idx = find_devices(sd, hostapi_index, out_pat, in_pat)
    base = {"hostapi": hostapi_name, "input": in_idx, "output": out_idx}
    if in_idx is None or out_idx is None:
        emit(kind="skip", reason="no loopback devices on this host API", **base)
        return
    base["input_name"] = sd.query_devices(in_idx)["name"]
    base["output_name"] = sd.query_devices(out_idx)["name"]
    signal = load_speech(src, seconds)

    # 1+2: timestamps and actual arrival time of the looped-back signal.
    try:
        out_log, in_log, chunks, in_off, out_off, lat = run_streams(
            sd, ac, in_idx, out_idx, hostapi_name, signal, seconds + 1.5
        )
    except Exception as exc:  # noqa: BLE001
        emit(kind="timestamps", error=f"{type(exc).__name__}: {exc}", **base)
        return
    emit(
        kind="timestamps",
        adc=regularity([r[0] for r in in_log], [r[3] for r in in_log]),
        dac=regularity([r[0] for r in out_log], [r[4] for r in out_log]),
        input_currentTime=regularity([r[1] for r in in_log], [r[3] for r in in_log]),
        callbacks=(len(in_log), len(out_log)),
        status_flags=(sum(r[4] for r in in_log), sum(r[5] for r in out_log)),
        stream_latency_ms=[x * 1000 for x in lat],
        clock_offsets_s=(in_off, out_off),
        first_adc=in_log[0][0] if in_log else None,
        first_dac=out_log[0][0] if out_log else None,
        **base,
    )
    from scipy.signal import correlate, correlation_lags

    captured = np.frombuffer(b"".join(chunks), dtype=np.int16).astype(float)
    ref = signal[: 4 * RATE].astype(float)
    scores = correlate(captured, ref, mode="full", method="fft")
    lags = correlation_lags(len(captured), len(ref))
    keep = lags >= 0
    best = int(np.argmax(np.abs(scores[keep])))
    lag = int(lags[keep][best])
    seg = captured[lag : lag + len(ref)]
    peak = float(np.abs(scores[keep][best]) / (np.linalg.norm(seg) * np.linalg.norm(ref) + 1e-9))

    in_starts = np.cumsum([0] + [r[3] for r in in_log[:-1]])
    adc = np.array([r[0] for r in in_log]) + in_off
    dac = np.array([r[0] for r in out_log]) + out_off
    pos = np.array([r[3] for r in out_log])
    adc_cur = np.array([r[1] for r in in_log]) + in_off
    dac_cur = np.array([r[1] for r in out_log]) + out_off
    anchors = {
        # What the PR effectively uses: first callback's time, then sample counting.
        "first_callback": (adc[0], dac[0] - pos[0] / RATE),
        # Same timeline with the median anchor (insensitive to one bad callback).
        "median": (np.median(adc - in_starts / RATE), np.median(dac - pos / RATE)),
        # Fallback without driver timestamps: callback time -/+ reported stream latency.
        "currentTime_latency": (
            np.median(adc_cur - lat[0] - in_starts / RATE),
            np.median(dac_cur + lat[1] - pos / RATE),
        ),
    }
    leads = {name: float(((a + lag / RATE) - d) * 1000) for name, (a, d) in anchors.items()}
    emit(kind="lead", lag_samples=lag, correlation=peak, lead_ms=leads, **base)

    # 3: the PR's own code paths.
    def collect(config, use_worker):
        received = []
        deadline = time.monotonic() + seconds + 25
        need = int((seconds - 0.5) * RATE)
        first = None
        if use_worker:
            worker = ac._AudioWorker(config)
            ctx = worker
            mic = worker.mic_queue
        else:
            playback = ac.PlaybackBuffer(RATE, 0)
            mic = queue.Queue(maxsize=100000)
            ctx = ac._open_audio_streams(config, playback, mic)
        with ctx:
            if use_worker:
                worker.playback.append(signal.tobytes())
                worker.playback.finish()
            else:
                playback.append(signal.tobytes())
                playback.finish()
            total = 0
            while total < need and time.monotonic() < deadline:
                try:
                    chunk = mic.get(timeout=0.5)
                except queue.Empty:
                    continue
                if first is None:
                    first = time.monotonic()
                received.append(chunk)
                total += len(chunk) // 2
            elapsed = time.monotonic() - first if first else 0.0
        audio = np.frombuffer(b"".join(received), dtype=np.int16).astype(float)
        return audio, elapsed

    window = slice(int(4 * RATE), int((seconds - 1) * RATE))

    def energy(x):
        return float(np.mean(x[window] ** 2)) if len(x) > window.stop else float("nan")

    common = dict(input_device=in_idx, output_device=out_idx, chunk_size=BLOCK, playback_buffer_ms=0)
    results = {}
    for label, cfg, use_worker in (
        ("raw_in_process", ac.RealtimeAudioClientConfig(echo_cancellation=False, **common), False),
        ("aec_in_process", ac.RealtimeAudioClientConfig(echo_cancellation=True, **common), False),
        ("aec_worker", ac.RealtimeAudioClientConfig(echo_cancellation=True, **common), True),
        # Suggested fix for B2: reference timestamps 20 ms earlier (in-process only).
        ("aec_in_process_lead20", ac.RealtimeAudioClientConfig(echo_cancellation=True, **common), False),
    ):
        original_render = ac.EchoCanceller.render
        if label.endswith("lead20"):
            ac.EchoCanceller.render = lambda self, audio, t: original_render(self, audio, t - 0.020)
        try:
            audio, elapsed = collect(cfg, use_worker)
            results[label] = {
                "energy": energy(audio),
                "samples": int(len(audio)),
                "received_rate_ratio": float(len(audio) / (RATE * elapsed)) if elapsed else None,
            }
        except Exception as exc:  # noqa: BLE001
            results[label] = {"error": f"{type(exc).__name__}: {exc}"}
        finally:
            ac.EchoCanceller.render = original_render
    raw_e = results.get("raw_in_process", {}).get("energy")
    for label in ("aec_in_process", "aec_worker", "aec_in_process_lead20"):
        e = results.get(label, {}).get("energy")
        if raw_e and e is not None and e == e:
            results[label]["reduction_db"] = float(10 * np.log10(raw_e / (e + 1e-9)))
    emit(kind="canceller", results=results, **base)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="src")
    parser.add_argument("--out", required=True, help="substring of the loopback playback device name")
    parser.add_argument("--in", dest="inp", required=True, help="substring of the loopback capture device name")
    parser.add_argument("--hostapi", action="append", help="host API name substring; repeatable; default all")
    parser.add_argument("--seconds", type=float, default=12.0)
    args = parser.parse_args()
    src = Path(args.src).resolve()
    sys.path.insert(0, str(src))
    import sounddevice as sd

    from speech_to_speech.api.openai_realtime import audio_client as ac

    emit(kind="env", platform=platform.platform(), python=sys.version.split()[0],
         portaudio=sd.get_portaudio_version()[1], sounddevice=sd.__version__,
         default_device=list(sd.default.device), hostapis=[h["name"] for h in sd.query_hostapis()])
    for i, h in enumerate(sd.query_hostapis()):
        if args.hostapi and not any(p.lower() in h["name"].lower() for p in args.hostapi):
            continue
        probe_hostapi(sd, ac, src, i, h["name"], args.out, args.inp, args.seconds)


if __name__ == "__main__":
    main()

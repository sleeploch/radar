#!/usr/bin/env python3
"""Burst-aware RSSI logger for an RTL-SDR. Logs only; no decoding."""
import argparse, csv, json, socket, threading, time
from datetime import datetime, timezone
import numpy as np
from rtlsdr import RtlSdr

ap = argparse.ArgumentParser()
ap.add_argument("--start", type=float, required=True, help="MHz")
ap.add_argument("--stop", type=float, required=True, help="MHz")
ap.add_argument("--out", required=True)
ap.add_argument("--gain", type=float, default=30.0)
ap.add_argument("--rate", type=float, default=2.4e6)
ap.add_argument("--dwell", type=float, default=0.2, help="seconds per tuning")
ap.add_argument("--chan", type=float, default=25e3, help="channel width Hz")
ap.add_argument("--thresh", type=float, default=10.0, help="dB above floor to flag a hit")
ap.add_argument("--full", action="store_true", help="also log every channel's peak power")
a = ap.parse_args()

NFFT, AVG = 1024, 8
usable = 0.7 * a.rate
span = (a.stop - a.start) * 1e6
n_tun = max(1, int(np.ceil(span / usable)))
step = span / n_tun
centres = [a.start * 1e6 + step * (i + 0.5) for i in range(n_tun)]
bins_per_chan = max(1, int(round(a.chan / (a.rate / NFFT))))
win = np.hanning(NFFT)

gps = {"lat": "", "lon": "", "speed": "", "mode": 0}
def gps_thread():
    while True:
        try:
            s = socket.create_connection(("127.0.0.1", 2947), timeout=5)
            s.sendall(b'?WATCH={"enable":true,"json":true}\n')
            f = s.makefile("r")
            for line in f:
                try:
                    m = json.loads(line)
                except ValueError:
                    continue
                if m.get("class") == "TPV":
                    gps.update(lat=m.get("lat", ""), lon=m.get("lon", ""),
                               speed=m.get("speed", ""), mode=m.get("mode", 0))
        except OSError:
            time.sleep(2)
threading.Thread(target=gps_thread, daemon=True).start()

def channel_power(iq, centre):
    n = (len(iq) // NFFT) * NFFT
    fr = iq[:n].reshape(-1, NFFT) * win
    p = np.abs(np.fft.fftshift(np.fft.fft(fr, axis=1), axes=1)) ** 2 / NFFT
    k = (len(p) // AVG) * AVG
    p = p[:k].reshape(-1, AVG, NFFT).mean(axis=1)
    freqs = centre + (np.arange(NFFT) - NFFT // 2) * (a.rate / NFFT)
    keep = np.abs(freqs - centre) <= usable / 2
    p, freqs = p[:, keep], freqs[keep]
    nb = (p.shape[1] // bins_per_chan) * bins_per_chan
    p = p[:, :nb].reshape(p.shape[0], -1, bins_per_chan).sum(axis=2)
    cf = freqs[:nb].reshape(-1, bins_per_chan).mean(axis=1)
    return 10 * np.log10(p + 1e-12), cf / 1e6

sdr = RtlSdr()
sdr.sample_rate = a.rate
sdr.gain = a.gain
nsamp = int(a.dwell * a.rate) // NFFT * NFFT

with open(a.out, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["utc", "lat", "lon", "speed_mps", "centre_mhz", "floor_db", "peak_db",
                "peak_mhz", "peak_minus_floor_db", "hit"])
    print(f"{n_tun} tunings: {[round(c/1e6,3) for c in centres]} MHz; Ctrl-C to stop")
    try:
        while True:
            for c in centres:
                sdr.center_freq = c
                sdr.read_samples(NFFT * 16)
                iq = sdr.read_samples(nsamp)
                db, cf = channel_power(iq, c)
                floor = float(np.median(db))
                chan_peak = db.max(axis=0)
                i = int(chan_peak.argmax())
                diff = float(chan_peak[i]) - floor
                hit = int(diff >= a.thresh)
                row = [datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                       gps["lat"], gps["lon"], gps["speed"], round(c / 1e6, 3),
                       round(floor, 1), round(float(chan_peak[i]), 1), round(float(cf[i]), 4),
                       round(diff, 1), hit]
                if a.full:
                    row += [round(float(x), 1) for x in chan_peak]
                w.writerow(row)
                fh.flush()
                if hit:
                    print(f"HIT {row[0]} {row[7]} MHz +{diff:.1f} dB")
    except KeyboardInterrupt:
        pass
    finally:
        sdr.close()
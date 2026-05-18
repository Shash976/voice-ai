# benchmark_whisper.py
import subprocess, time, psutil, sys, os

def run_whisper(wav_path, model="tiny.en"):
    cmd = [
        "./whisper.cpp/build/bin/whisper-cli",          # adjust path as needed
        "-m", f"whisper.cpp/models/ggml-{model}.bin",
        "-f", wav_path,
        "--no-timestamps",
    ]
    proc = psutil.Process(os.getpid())
    mem_before = proc.memory_info().rss / 1e6

    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0

    mem_after = proc.memory_info().rss / 1e6
    cpu = psutil.cpu_percent(interval=None)

    return {
        "transcript": result.stdout.strip(),
        "latency_s": round(elapsed, 2),
        "mem_delta_mb": round(mem_after - mem_before, 1),
        "file": wav_path,
    }

if __name__ == "__main__":
    r = run_whisper(sys.argv[1])
    print(r)


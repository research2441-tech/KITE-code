"""Parallel job runner with in-progress markers, so that a restarted queue
never duplicates a job that another worker is still running.
usage: python jobqueue2.py jobs.txt [workers]"""
import os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(HERE, "results", ".running")
os.makedirs(RUN, exist_ok=True)


def tag_of(line):
    p = line.split()
    name, seed, extra = p[1], p[2], p[3:]
    return name + "".join("_" + kv.replace("=", "-") for kv in extra) + f"__s{seed}"


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def busy(tag):
    m = os.path.join(RUN, tag)
    if not os.path.exists(m):
        return False
    try:
        return alive(int(open(m).read().strip()))
    except Exception:
        return False


DEADLINE = [float("inf")]


def work(line):
    if time.time() > DEADLINE[0]:
        return
    tag = tag_of(line)
    if os.path.exists(os.path.join(HERE, "results", tag + ".json")) or busy(tag):
        return
    t = time.time()
    p = subprocess.Popen(["python3"] + line.split(), cwd=HERE, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True)
    open(os.path.join(RUN, tag), "w").write(str(p.pid))
    out, err = p.communicate()
    try:
        os.remove(os.path.join(RUN, tag))
    except OSError:
        pass
    with open(os.path.join(HERE, "queue.log"), "a") as f:
        f.write(f"{time.strftime('%H:%M:%S')} {line} rc={p.returncode} {time.time()-t:.0f}s "
                f"{out.strip()[-300:]} {err.strip()[-300:] if p.returncode else ''}\n")


if __name__ == "__main__":
    jobs = [l.strip() for l in open(sys.argv[1]) if l.strip() and not l.startswith("#")]
    w = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    if len(sys.argv) > 3:  # stop starting new jobs after this many minutes
        DEADLINE[0] = time.time() + 60 * float(sys.argv[3])
    with ThreadPoolExecutor(w) as ex:
        list(ex.map(work, jobs))

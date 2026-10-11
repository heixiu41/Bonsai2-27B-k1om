"""During one generation turn, sample `ps` on the card once per second
(via short ssh calls) to catch whatever periodically stalls inference."""
import subprocess, sys, time, os

home = os.environ.get("USERPROFILE")
SSH = ["ssh", "-i", os.path.join(home, ".ssh", "id_rsa_mic"),
       "-o", f"UserKnownHostsFile={os.path.join(home, '.ssh', 'known_hosts')}",
       "-o", "StrictHostKeyChecking=no", "root@192.168.1.100"]

print("sampling card ps for 25s -- send a message NOW in the browser",
      flush=True)
t0 = time.time()
while time.time() - t0 < 25:
    r = subprocess.run(SSH + ["ps"], capture_output=True, text=True,
                       timeout=20)
    lines = [l for l in r.stdout.splitlines()
             if any(k in l for k in ("bonsai", "scif", "cat ", "sh -",
                                     "sleep", "stat", "sshd"))]
    stamp = f"{time.time() - t0:5.1f}s"
    for l in lines:
        print(stamp, l.strip(), flush=True)
    print(stamp, "----", flush=True)
    time.sleep(1)

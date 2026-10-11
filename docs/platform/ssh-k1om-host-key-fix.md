# Fixing the SSH Host Key Problem When Connecting to 192.168.1.100 (Xeon Phi k1om)

> Purpose: lock in a fixed fix procedure so it is not forgotten in later conversations / after switching machines. Applies to Windows host + OpenSSH.

## Symptoms

```
ssh -i %USERPROFILE%\.ssh\id_rsa_mic root@192.168.1.100
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@    WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED!     @
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
Offending ECDSA key in C:\Users\Administrator\.ssh\known_hosts:18
Host key for 192.168.1.100 has changed ...
Host key verification failed.
```

## Root Cause

- After the Phi was reflashed / MPSS was reinstalled, the coprocessor regenerated its SSH host key
- The local `known_hosts` still contains the old key → strict OpenSSH verification rejects the connection
- The local `known_hosts` may also be corrupted (garbage bytes at the start of the first line), which even `ssh-keygen -R` cannot fix

## Fix Steps (PowerShell)

```powershell
# 1. Back up the corrupted known_hosts
$KH = "$env:USERPROFILE\.ssh\known_hosts"
Copy-Item $KH "$KH.bak"

# 2. Re-fetch Phi's current host key with ssh-keyscan and rebuild known_hosts
ssh-keyscan -t ed25519,ecdsa,rsa 192.168.1.100 2>$null | Out-File -Encoding ascii $KH

# 3. Verify the connection
ssh -o BatchMode=yes -o ConnectTimeout=8 -i "$env:USERPROFILE\.ssh\id_rsa_mic" root@192.168.1.100 "uptime"
```

Key points:
- `Out-File -Encoding ascii`: ssh requires known_hosts to be ASCII/UTF-8; the default UTF-16 causes parsing failures
- If you don't want to rebuild the whole file, you can first delete the corrupted first line and then append with `ssh-keyscan`

## Related: uploads must use scp -O

The `scp` in newer Windows OpenSSH uses the SFTP subsystem by default; KNC's BusyBox has no SFTP → `Connection closed (exit 255)`.

```powershell
# Must add -O (legacy scp protocol)
scp -O -i "$env:USERPROFILE\.ssh\id_rsa_mic" <local-file> root@192.168.1.100:/tmp/
# After uploading an ELF, add the execute bit
ssh -i "$env:USERPROFILE\.ssh\id_rsa_mic" root@192.168.1.100 "chmod +x /tmp/<filename>"
```

## Running Inference (with library path)

```powershell
ssh -i "$env:USERPROFILE\.ssh\id_rsa_mic" root@192.168.1.100 "cd /tmp && LD_LIBRARY_PATH=/tmp OMP_NUM_THREADS=61 ./llama2_k1om stories110M.bin -t 0.9 -s 42 -n 128 -z tokenizer.bin -N 2"
```

- `LD_LIBRARY_PATH=/tmp`: libiomp5.so is placed in /tmp
- `OMP_NUM_THREADS`: OpenMP thread count (1-244)
- The Segmentation fault when the program exits is a known issue and does not affect the tok/s results

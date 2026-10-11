# WSL Ubuntu Login Info

| Item | Value |
|------|-----|
| Distribution | Ubuntu 22.04.5 LTS (Jammy Jellyfish) |
| Hostname | WIN-PVGD2RBUIOQ |
| Default user | **root** (uid=0) |
| Password | **Not set** (WSL uses Windows authentication, no password needed) |

## Usage

```powershell
# Log in as root
wsl

# Or log in as a specified user
wsl -u root
```

## Notes

- WSL defaults to `root`; no regular user has been created
- Enter directly via the `wsl` command, no password required
- To set a password: `passwd`

## Quick Session

```bash
# Activate the k1om toolchain
source /mnt/c/Users/Administrator/Desktop/project/wsl-k1om-env.sh

# Test the Xeon Phi connection
mic-ssh
```

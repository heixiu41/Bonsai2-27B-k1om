# mic.ps1 -- upload a built k1om binary to the coprocessor and run it.
# BusyBox on the card has no sftp subsystem, so scp must use the legacy
# protocol (-O) and a modern cipher.  chmod +x is mandatory.
param(
    [Parameter(Mandatory=$true)][string]$Exe,   # local path, e.g. D:\zigtest\k1om\pq2imci
    [string]$Name,                              # remote basename (default: file name)
    [string]$ArgLine = '',
    [string]$Host_ = '192.168.1.100'
)
$key = "$env:USERPROFILE\.ssh\id_rsa_mic"
$kh  = "-o UserKnownHostsFile=`"$env:USERPROFILE\.ssh\known_hosts`" -o StrictHostKeyChecking=no"
if (-not $Name) { $Name = [System.IO.Path]::GetFileName($Exe) }
scp -O -c chacha20-poly1305@openssh.com -i "$key" $kh $Exe "root@${Host_}:/tmp/$Name" 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) { throw "scp failed for $Exe" }
ssh -i "$key" $kh "root@$Host_" "chmod +x /tmp/$Name && export LD_LIBRARY_PATH=/tmp && export OMP_NUM_THREADS=${env:MIC_OMP:-244} && /tmp/$Name $ArgLine"
exit $LASTEXITCODE

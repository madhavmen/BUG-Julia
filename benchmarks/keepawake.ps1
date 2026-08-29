# Hold the system awake while the L=16 runs are in flight.
#
# The first attempt lost NINE HOURS of wall clock to the box sleeping: between 22:37 and 07:29 the
# four Julia jobs accumulated 67 CPU-seconds between them.
#
# ⚠ This is NOT a persistent power-setting change. SetThreadExecutionState sets a flag on THIS
# process only -- the same mechanism a media player uses -- and Windows drops it the moment this
# process exits. Nothing to undo, and nothing left behind if it is killed.
Add-Type -Name Power -Namespace Win32 -MemberDefinition @'
[DllImport("kernel32.dll", SetLastError = true)]
public static extern uint SetThreadExecutionState(uint esFlags);
'@
# ES_CONTINUOUS (0x80000000) | ES_SYSTEM_REQUIRED (0x00000001)
$r = [Win32.Power]::SetThreadExecutionState(0x80000001)
if ($r -eq 0) { Write-Output "SetThreadExecutionState FAILED" } else { Write-Output "keep-awake armed" }
while ($true) { Start-Sleep -Seconds 60 }

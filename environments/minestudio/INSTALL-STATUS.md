# Visual runtime installation checkpoint

The user authorized installing the required system dependencies.

Windows optional features were enabled successfully using `-All -NoRestart`:

- VirtualMachinePlatform
- Microsoft-Windows-Subsystem-Linux

Both operations returned `RestartNeeded: True`. The user subsequently restarted.
Ubuntu 22.04 is now installed as the WSL distribution `MinecraftScientist` at
`D:\minecraft-scientist-linux`. Linux kernel startup has been verified.

Microsoft WSL 2.7.13 was subsequently installed successfully through winget's
official Microsoft.WSL package. Winget verified the installer hash and returned
exit code 0 with a successful installation message. It reiterated that the enabled
VirtualMachinePlatform feature requires a Windows restart.

Verified on 2026-09-17:

- OpenJDK 1.8.0_502 installed via Ubuntu apt.
- WSLg OpenGL direct rendering uses D3D12 / NVIDIA RTX 3070 Laptop GPU.
- Xvfb, Mesa, FFmpeg, compiler tools, Python pip/venv installed.
- uv installed; isolated Python 3.11.16 environment at
  `/opt/minecraft-scientist-visual`.
- MineStudio 1.1.6 installed using `overrides.txt`; actual simulator imports pass.
- CPU PyTorch 2.8.0 and NumPy 1.26.4 are isolated from the Windows environment.
- Persistent simulator engine resides in `/opt/minecraft-scientist-engine`.
  Set `MINESTUDIO_DIR` to this directory; the default `/tmp` engine is not durable.
- Real simulator reset and five no-op steps completed with exit code 0.
  Six RGB observations have shape 224 x 224 x 3. Saved initial image was visually
  inspected and shows terrain, trees, HUD and player hand, not a blank frame.
  Evidence: `runs/minestudio-live-smoke/report.json`, `initial.png`, `final.png`.

Remaining work:

1. Production adapter smoke passed: select/interact/wait, seven simulator steps,
   four saved RGB frames, exit code 0. Evidence:
   `runs/minestudio-adapter-smoke/report.json` and `events.jsonl`.
   The Windows default suite also passed all 82 tests after the wait-duration fix.
2. Keep monitoring disk space before downloading additional models.
3. Replace screen-cell targeting with grounded perception before claiming visual
   scientific discovery. Simulator startup is not evidence of mechanism recovery.

Latest verified controlled gate (2026-09-19):

- Backend readiness reports Linux, MineStudio 1.1.6 and OpenJDK 1.8.0_502 ready.
- `runs/active-visual-persistent-region-seeds22-23` completed two active discovery
  probes and two independently seeded holdouts with 1.0 accuracy and coverage.
- Four successful opaque `SELECT` events and four equipment traces passed;
  four automatic grid-grounding artifacts also passed integrity audit.
- Frozen prediction SHA256:
  `981759aef3c79308305f71db7477ede40719ac4c1896dd9edd6eaa0707d03d08`.
- The preregistered changed-region IoU threshold was 0.5; held-out IoU was 0.857.
- `runs/camera-search-suite-seed30` correctly selected all three preregistered yaw
  placements using 15/15 successful `LOOK` actions; PNG score recomputation and
  five-file source-snapshot audit pass.
- Windows 122-test suite and Linux 48-test visual subset pass; lock check and both
  sdist/wheel builds pass. The offline MVP benchmark remains fully successful.
- This is a controlled, fixture-bound result—not autonomous object grounding,
  navigation, or a full visual Minecraft scientific-discovery benchmark.

Reproduction (PowerShell):

```powershell
wsl -d MinecraftScientist -u root --exec /opt/minecraft-scientist-visual/bin/python /mnt/d/minecraft-scientist/environments/minestudio/smoke.py
wsl -d MinecraftScientist -u root --exec /opt/minecraft-scientist-visual/bin/python /mnt/d/minecraft-scientist/environments/minestudio/adapter_smoke.py
```

The dependency overrides replace obsolete upstream OpenCV and NetworkX pins.
This is a tested compatibility environment, not a claim that upstream dependency
metadata is conflict-free. Do not install the visual stack into the default venv.

The built-in `wsl --install` command previously returned an uninstalled-runtime
message. An official GitHub release lookup also failed with a transport EOF;
retry after reboot before choosing another official download route.

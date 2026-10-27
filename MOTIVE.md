# Live Motive connection

## Live skeleton viewer

Run `./view-motive.ps1` to open a direct NatNet feed in AI4AnimationPy. It receives full source-rate frames through an in-process callback and displays the latest pose at 60 FPS. It does not read the twice-per-second status snapshot for playback. Source coordinates and Motive parent IDs are preserved, without camera-style flips or vertical shifts. Untracked bones are hidden; stale motion is hidden after one second. Closing the viewer stops its own connection.

Space freezes/resumes display, A/D orbit, W/S zoom, T toggles bone rotation axes, and Escape exits. Right-side bones are orange. Verify right/left identification, elbow bending and wrist orientation against the actual person before training. A visible skeleton confirms rendering and hierarchy, not anatomical calibration accuracy. Validation rendered Skeleton 002 upright with 19 tracked bones. All 30 tests pass.

The launcher currently uses the prepared environment from the earlier workspace. Set `DARTISTIC_PYTHON` to another working AI4AnimationPy interpreter when relocating the project.

Run `./connect-motive.ps1` from this project to connect to the current `Skeleton 002` asset. Ctrl+C stops the connection. Use `-Skeleton 'Skeleton'` to select the other asset. This uses the official installed NatNet SDK 4.4 Python decoder without modifying its files. It connected successfully to Motive 3.0.3, negotiating NatNet 4.0.

Current Motive settings verified in its UI: NatNet enabled, Loopback interface, Unicast transmission, Skeletons enabled, Global skeleton coordinates, Motive bone names, Y-axis up. Start this client outside a restricted sandbox if socket access fails. No additional Python packages are required for the pipe.

The connection receives full frame packets at the source rate. `work/motive-status.json` contains connection status and discovered assets; `work/motive-latest-frame.json` publishes the latest normalized frame twice per second for inspection. That status snapshot is not a full-rate transport. Full-rate recording is available with:

```powershell
./connect-motive.ps1 -Skeleton 'Skeleton 002' -Output 'work/motive-session.jsonl'
```

The output refuses overwrite, preserves tracked flags and xyzw rotations, and uses the bounded disk queue. Out-of-order timestamps or changed asset definitions stop the pipe; restart for a new session. Disconnects are detected after three seconds without frames.

The default selects the currently observed `Skeleton 002`; it does not automatically identify a person. Verify player ownership and bone placement before using recordings to train. Motive LUArm/RUArm origins map to shoulders, LFArm/RFArm to elbows, and hands to wrists; other bones remain named with `bone:` prefixes. These are bone origins and require anatomical validation. The SDK path can be changed with `python camera/motive.py --sdk <PythonClient-folder> ...`.

Validation on October 14, 2026: five-second recording test saved 518 selected-skeleton frames at 120 Hz, no source gaps or queue overflow. Nineteen selected bones were tracked at the end; the right shoulder, elbow, and wrist were present. The test recording is in work/ and has not entered personal training data.

Earlier prototype modules have been copied into this repository's `camera/` folder so the live adapter shares its schema and recorder. The original camera recordings and environments remain at their earlier workspace location.

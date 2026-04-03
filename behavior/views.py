import os
import sys
import json
import subprocess
from django.conf import settings
from django.shortcuts import render, redirect
from django.contrib import messages


def behavior_page(request):
    log = request.session.pop("behavior_log", None)
    return render(request, "behavior.html", {"log": log})


def start_monitor(request):
    """
    Runs scan_behavior.py as a subprocess so cv2.imshow has the process
    main thread (required on macOS — inline imshow in a Django worker
    thread raises 'Unknown C++ exception' or 'os is not defined').
    """
    if request.method != "POST":
        return redirect("behavior")

    duration    = int(request.POST.get("duration", 60))
    script_path = os.path.join(settings.BASE_DIR, "scan_behavior.py")

    if not os.path.exists(script_path):
        messages.error(request,
            "scan_behavior.py not found in project root (next to manage.py).")
        return redirect("behavior")

    try:
        # macOS: pass environment to properly initialize GUI
        env = os.environ.copy()
        
        result = subprocess.run(
            [sys.executable, script_path, str(duration)],
            capture_output=True, text=True,
            timeout=duration + 30,
            env=env,
        )

        summary = None
        for line in result.stdout.strip().splitlines():
            if line.startswith("RESULT:"):
                summary = json.loads(line[len("RESULT:"):])
                break

        if summary:
            request.session["behavior_log"] = summary
            a = summary["attentive"]
            p = summary["using_phone"]
            d = summary["distracted"]
            messages.success(request,
                f"Session done — Attentive: {a}s | Phone: {p}s | Distracted: {d}s")
        else:
            messages.warning(request, "Session ended but no data was captured.")

        if result.returncode != 0 and result.stderr:
            print("[behavior stderr]", result.stderr[:500])

    except subprocess.TimeoutExpired:
        messages.error(request, "Session timed out unexpectedly.")
    except Exception as e:
        messages.error(request, f"Monitor failed: {e}")

    return redirect("behavior")
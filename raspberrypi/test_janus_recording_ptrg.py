#!/usr/bin/env python3

import os
import time
import subprocess
import sys

# ==== CONFIG ====
JANUS_PATH = "/home/arratialab/Documents/Janus_5202_4.2.4_20251007_linux/bin/JanusC"
SESSION_NAME = "janus_ptrg"

# === Helper functions ===

def send_tmux_cmd(cmd, delay=0.2):
    subprocess.run(["tmux", "send-keys", "-t", SESSION_NAME, cmd, "C-m"])
    time.sleep(delay)

def janus_startup():
    print("[*] Starting Janus...")
    subprocess.run([
        "tmux", "new-session", "-d", "-s", SESSION_NAME, f"./JanusC"
    ], cwd=os.path.dirname(JANUS_PATH))
    time.sleep(5)

    print("[*] Sending voltage on commands (h, H, q)...")
    for key in ['h', 'H', 'q']:
        send_tmux_cmd(key, delay=2)

    print("[*] Waiting 20s for voltage to ramp up...")
    time.sleep(20)

def janus_start_recording():
    print("[*] Starting recording (s)...")
    send_tmux_cmd('s')

def janus_stop_recording():
    print("[*] Stopping recording and shutting down...")
    for key in ['S', 'h', 'H', 'm', 'q', 'q']:
        send_tmux_cmd(key, delay=5)

    time.sleep(5)
    subprocess.run(["tmux", "kill-session", "-t", SESSION_NAME])
    print("[*] Janus session terminated.")

# === Main ===

def main():
    janus_startup()
    janus_start_recording()
    print(f"[*] Recording for {int(sys.argv[1])} seconds...")
    time.sleep(int(sys.argv[1]))
    janus_stop_recording()

if __name__ == "__main__":
    main()

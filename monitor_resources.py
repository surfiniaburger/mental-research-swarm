import time
import sys
import os
sys.path.append(os.getcwd())
from swarm.telemetry import get_mac_hardware_stats, log_telemetry

def main():
    print("Starting Standalone Resource Monitor...")
    print("Logging hardware stats to telemetry.log every 30 seconds.")
    print("Press Ctrl+C to stop.")
    
    try:
        while True:
            stats = get_mac_hardware_stats()
            log_telemetry("standalone_heartbeat", stats)
            print(f"[{time.strftime('%H:%M:%S')}] CPU: {stats['cpu_util']}% | Mem: {stats['mem_util_gb']}GB | VRAM: {stats['vram_free_mb']}MB")
            time.sleep(30)
    except KeyboardInterrupt:
        print("\nStopping monitor.")

if __name__ == "__main__":
    main()

import taipy.gui.builder as tgb
import taipy as tp
from taipy.gui import Gui, State
import pandas as pd
import json
import os
import time
import threading

# --- Paths ---
LOG_FILE = "telemetry.log"
CRITIQUE_FILE = "research_env/docs/last_critique.md"
SESSION_FILE = "research_env/docs/session_state.json"

def load_data():
    data = []
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "r") as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        if "metrics" in entry:
                            row = {
                                "timestamp": entry.get("timestamp"),
                                "event_type": entry.get("event_type"),
                                **entry["metrics"]
                            }
                            data.append(row)
                    except:
                        continue
        except Exception as e:
            print(f"Error reading log: {e}")
    return pd.DataFrame(data)

def load_scenarios():
    try:
        scenarios = tp.get_scenarios()
        data = []
        for s in scenarios:
            res = s.research_result.read()
            data.append({
                "id": s.id,
                "name": s.name,
                "iteration": s.iteration.read(),
                "val_bpb": res.val_bpb if res else 0.0,
                "status": res.status if res else "pending"
            })
        return pd.DataFrame(data)
    except:
        return pd.DataFrame(columns=["id", "name", "iteration", "val_bpb", "status"])

def load_critique():
    if os.path.exists(CRITIQUE_FILE):
        try:
            with open(CRITIQUE_FILE, "r") as f:
                return f.read()
        except:
            pass
    return "### Evaluation: Normal Sequence\nNo active rejections."

def load_session():
    if os.path.exists(SESSION_FILE):
        try:
            with open(SESSION_FILE, "r") as f:
                return json.load(f)
        except:
            pass
    return {"iteration": 0, "global_best_bpb": 0.0}

# --- State Variables ---
df = load_data()
scenario_df = load_scenarios()
cpu_val = "0.0%"
mem_val = "0.0GB"
vram_val = "0.0MB"
iter_val = 0
best_bpb = 0.0
status = "Operational"
critique_md = load_critique()

def update_all_clients(state: State):
    """
    Callback invoked by broadcast_callback.
    Taipy calls this once per connected client, passing their State object.
    We refresh the data and assign it directly to the state.
    """
    new_df = load_data()
    new_scenarios = load_scenarios()
    session = load_session()
    critique = load_critique()

    state.df = new_df
    state.scenario_df = new_scenarios
    if not new_df.empty:
        latest = new_df.iloc[-1]
        state.cpu_val = f"{latest.get('cpu_util', 0.0)}%"
        state.mem_val = f"{latest.get('mem_util_gb', 0.0)}GB"
        state.vram_val = f"{latest.get('vram_free_mb', 0.0)}MB"

    state.iter_val = session.get("iteration", 0)
    state.best_bpb = session.get("global_best_bpb", 0.0)
    state.critique_md = critique

    if os.path.exists(LOG_FILE):
        last_update = os.path.getmtime(LOG_FILE)
        state.status = "Active" if (time.time() - last_update) < 60 else "Stale"

def refresh_background(gui: Gui):
    """Background thread that triggers a broadcast refresh every 10 seconds."""
    while True:
        try:
            gui.broadcast_callback(update_all_clients)
        except Exception as e:
            print(f"Refresh error: {e}")
        time.sleep(10)

# --- Layout ---
with tgb.Page() as page:
    with tgb.layout(columns="300px 1"):
        # Sidebar
        with tgb.part(class_name="sidebar-part"):
            tgb.text("NEXUS AI", variant="h2")
            tgb.text("MISSION CONTROL", variant="h5")
            tgb.html("hr")
            tgb.text("SYSTEM STATUS", variant="h6")
            tgb.text("{status}", variant="h4")
            tgb.html("br")
            tgb.text("CURRENT ITERATION", variant="h6")
            tgb.text("{iter_val}", variant="h3")
            tgb.html("br")
            tgb.text("GLOBAL BEST BPB", variant="h6")
            tgb.text("{best_bpb}", variant="h3")
            tgb.html("hr")
            tgb.text("CRITIC FEEDBACK", variant="h6")
            tgb.text("{critique_md}", mode="md")

        # Main Content
        with tgb.part():
            with tgb.layout(columns="1 1 1"):
                with tgb.part(class_name="card"):
                    tgb.text("CPU UTIL", variant="h6")
                    tgb.text("{cpu_val}", variant="h2")
                with tgb.part(class_name="card"):
                    tgb.text("RAM UTIL", variant="h6")
                    tgb.text("{mem_val}", variant="h2")
                with tgb.part(class_name="card"):
                    tgb.text("VRAM FREE", variant="h6")
                    tgb.text("{vram_val}", variant="h2")

            with tgb.expandable("Live Metrics Flux"):
                tgb.chart("{df}", type="line", x="timestamp", y=["cpu_util", "mem_util_gb"], title="System Resource Flux")

            with tgb.expandable("Scenario Ledger (Taipy Core)", expanded=True):
                tgb.table("{scenario_df}", title="Historical Trials", filterable=True)

            tgb.table("{df}", title="Activity Log", page_size=10, filterable=True)

# --- Styling ---
stylekit = {
    "color_primary": "#00FF7F",
    "color_secondary": "#1E1E2E",
    "font_family": "Inter, system-ui",
    "background_color_dark": "#0B0B14",
    "card_background_color_dark": "#161625"
}

if __name__ == "__main__":
    gui = Gui(page=page)
    t = threading.Thread(target=refresh_background, args=(gui,), daemon=True)
    t.start()
    gui.run(use_reloader=False, port=8081, title="Nexus AI Dashboard", stylekit=stylekit, dark_mode=True)

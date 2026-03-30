#!/bin/bash
# brain — SessionStart hook: boots brain, prints context + consciousness signals.
# Output: full brain state for Claude's context (injected via SessionStart stdout)
#
# Brain DB resolution order:
# 1. BRAIN_DB_DIR env var (explicit override)
# 2. /sessions/*/mnt/AgentsContext/brain/ (Cowork mounted paths)
# 3. $HOME/AgentsContext/brain/ (local Claude Code via symlink)
# If none found, boot fails cleanly (no /tmp fallback — silent data loss is worse).

source "$(dirname "$0")/resolve-brain-db.sh"

# No DB found — guide the user
if [ -z "$BRAIN_DB_DIR" ]; then
  echo ""
  echo "brain: No brain.db found."
  echo ""
  echo "Two options:"
  echo ""
  echo "  1. CONNECT TO EXISTING BRAIN — Set the path to your brain folder:"
  echo "     In Claude Code settings or .claude/settings.json, add to env:"
  echo '       "BRAIN_DB_DIR": "/path/to/your/brain/folder"'
  echo "     The folder should contain (or will contain) brain.db."
  echo ""
  echo "  2. START FRESH — Create a new brain:"
  echo "     mkdir -p ~/AgentsContext/brain"
  echo "     Then restart this session. The brain will initialize automatically."
  echo ""
  echo "Searched locations:"
  echo "  - \$BRAIN_DB_DIR env var (not set)"
  echo "  - /sessions/*/mnt/AgentsContext/brain/ (Cowork — not found)"
  echo "  - \$HOME/AgentsContext/brain/ (not found)"
  echo ""
  exit 0
fi

# ── Start daemon FIRST (foreground) — single source of truth ──
# The daemon keeps Brain + embedder loaded in memory.
# Boot waits for daemon, then uses it for context formatting.
# No direct Brain() instantiation — daemon owns the model.
python3 -c "
import sys, os, socket, json, subprocess, time

parent = os.path.dirname(os.environ.get('BRAIN_SERVER_DIR', ''))
if parent:
    sys.path.insert(0, parent)

db_dir = os.environ.get('BRAIN_DB_DIR', '')
db_path = os.path.join(db_dir, 'brain.db')
port = 47200 + (os.getuid() % 100)

# Check if daemon is already running
def ping():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(2)
        s.connect(('127.0.0.1', port))
        s.sendall((json.dumps({'cmd': 'ping'}) + '\n').encode())
        data = s.recv(4096)
        s.close()
        return json.loads(data.decode().strip()).get('ok', False)
    except Exception:
        return False

if ping():
    sys.stderr.write('[brain-boot] Daemon already running on port %d\n' % port)
else:
    # Start daemon in background
    sys.stderr.write('[brain-boot] Starting daemon...\n')
    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    startup = (
        'import sys, os; '
        'sys.path.insert(0, %r); '
        'os.environ[\"BRAIN_DB_DIR\"] = %r; '
        'from servers.daemon_server import BrainDaemon; '
        'd = BrainDaemon(%r); d.start()'
    ) % (project_dir, db_dir, db_path)
    subprocess.Popen([sys.executable, '-c', startup],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     start_new_session=True)
    # Wait for daemon to be ready
    for i in range(8):
        time.sleep(1)
        if ping():
            sys.stderr.write('[brain-boot] Daemon ready (took %ds)\n' % (i + 1))
            break
    else:
        sys.stderr.write('[brain-boot] WARNING: Daemon did not start within 8s\n')
"

exec python3 "$(dirname "$0")/boot_brain.py"

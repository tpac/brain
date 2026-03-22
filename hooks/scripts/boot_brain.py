"""SessionStart hook — boots brain, prints context + consciousness signals.
Thin wrapper: resolves env, calls Brain.format_boot_context(), prints result.
All formatting logic lives in BrainSurfaceMixin.format_boot_context().
"""
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from hook_common import db_path

server_dir = os.environ.get("BRAIN_SERVER_DIR", "")
db_dir = os.environ.get("BRAIN_DB_DIR", "")

if server_dir:
    parent = os.path.dirname(server_dir)
    if parent not in sys.path:
        sys.path.insert(0, parent)

try:
    from servers.brain import Brain
except ImportError as e:
    print("brain: Failed to import brain module: " + str(e), file=sys.stderr)
    sys.exit(1)

try:
    brain = Brain(db_path)
except Exception as e:
    print("brain: Failed to initialize: " + str(e), file=sys.stderr)
    sys.exit(1)

brain.reset_session_activity()

# Validate config (stderr for critical warnings)
config_warnings = brain.validate_config()
for w in config_warnings:
    level = w.get("level", "warning").upper()
    msg = w.get("message", "")
    if level == "CRITICAL":
        print("CRITICAL: " + msg, file=sys.stderr)
    else:
        print("WARNING: " + msg, file=sys.stderr)

# Resolve user/project from env or brain config
user = os.environ.get("BRAIN_USER", "User")
project = os.environ.get("BRAIN_PROJECT", "default")
if user == "User":
    stored_user = brain.get_config("default_user", "User")
    if stored_user and stored_user != "User":
        user = stored_user
if project == "default":
    stored_project = brain.get_config("default_project", "default")
    if stored_project and stored_project != "default":
        project = stored_project

# Export debug mode for child hooks
debug_enabled = brain.get_config("debug_enabled", "0") == "1"
if debug_enabled:
    os.environ["BRAIN_DEBUG"] = "1"

# Single call — all formatting lives in Brain
print(brain.format_boot_context(user=user, project=project, db_dir=db_dir))

brain.close()

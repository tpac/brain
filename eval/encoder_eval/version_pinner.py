"""S1eVersion — context manager that pins the s1e active version via
`set_interaction_active`, restores on exit.

Cleaner than `diff_encoding.py`'s re-register-as-latest hack: doesn't
pollute the interaction table with v23, v24, v25, ... on every eval run.
Captures the prior active version at entry so restoration is faithful even
if the version was changed mid-context by something else.

Usage:
    with S1eVersion(22) as pinner:
        # s1e active_version is now 22 for any encoder call
        replay_item(brain, ...)
    # Prior active version restored on exit (even if body raised)
"""
from typing import Optional


class S1eVersion:
    """Context manager for pinning the s1e active version during an eval arm."""

    INTERACTION_NAME = 's1e'

    def __init__(self, version: int, dry_run: bool = False):
        """
        Args:
            version: target s1e version (must already be registered).
            dry_run: if True, capture prior but don't actually flip — for
                     test scaffolding. Real eval runs use False.
        """
        self.version = int(version)
        self.dry_run = dry_run
        self.prior: Optional[int] = None
        self.target_template_length: Optional[int] = None

    def __enter__(self):
        from servers.daemon_client import send_command

        # Capture prior active version + verify target exists
        info = send_command('list_interactions', {})
        if not info.get('ok'):
            raise RuntimeError(f"list_interactions failed: {info}")
        s1e_entry = next(
            (e for e in info.get('result', []) if e.get('name') == self.INTERACTION_NAME),
            None)
        if s1e_entry is None:
            raise RuntimeError(f"interaction '{self.INTERACTION_NAME}' not registered")
        self.prior = int(s1e_entry.get('active_version'))
        if self.version > int(s1e_entry.get('max_version', 0)):
            raise RuntimeError(
                f"s1e v{self.version} not registered (max_version="
                f"{s1e_entry.get('max_version')})")

        # Fetch target template length for logging
        target = send_command('get_interaction', {
            'name': self.INTERACTION_NAME, 'version': self.version})
        if target.get('ok'):
            self.target_template_length = len(
                target.get('result', {}).get('template') or '')

        # Flip
        if not self.dry_run:
            r = send_command('set_interaction_active', {
                'name': self.INTERACTION_NAME, 'version': self.version})
            if not r.get('ok'):
                raise RuntimeError(
                    f"set_interaction_active failed for {self.INTERACTION_NAME} "
                    f"v{self.version}: {r}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.prior is None or self.dry_run:
            return False  # nothing to restore
        try:
            from servers.daemon_client import send_command
            send_command('set_interaction_active', {
                'name': self.INTERACTION_NAME, 'version': self.prior})
        except Exception as e:
            # Restoration failure is bad but should not mask a body exception.
            # Log loudly; raise only if body succeeded.
            import sys
            print(f"[version_pinner] RESTORE FAILED — s1e left at v{self.version}, "
                  f"prior was v{self.prior}. Error: {e}", file=sys.stderr,
                  flush=True)
            if exc_type is None:
                raise
        return False  # don't swallow exceptions

    def __repr__(self):
        return (f"S1eVersion(version={self.version}, prior={self.prior}, "
                f"template_length={self.target_template_length})")


def current_active_version() -> int:
    """Convenience: return the current s1e active_version, no flip."""
    from servers.daemon_client import send_command
    info = send_command('list_interactions', {})
    if not info.get('ok'):
        raise RuntimeError(f"list_interactions failed: {info}")
    for e in info.get('result', []):
        if e.get('name') == S1eVersion.INTERACTION_NAME:
            return int(e['active_version'])
    raise RuntimeError(f"interaction '{S1eVersion.INTERACTION_NAME}' not registered")

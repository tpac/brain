"""Codex-owned hook review: inspect definitions and open the native review UI.

No Entity imports, brain access, hook execution, trust writes, or approval
keystrokes. A trusted definition is not proof of a working memory pipeline.
The MCP caller owns elicitation and must get consent before open_review().
"""
import json
import os
from pathlib import Path
import selectors
import shlex
import shutil
import subprocess
import sys
import tempfile
import time


class ReviewError(RuntimeError):
    pass


def find_codex(executable=None):
    """Prefer the desktop's bundled runtime to a possibly different PATH CLI."""
    candidates = [executable] if executable else [
        '/Applications/ChatGPT.app/Contents/Resources/codex',
        '/Applications/Codex.app/Contents/Resources/codex',
        str(Path.home() / 'Applications/ChatGPT.app/Contents/Resources/codex'),
        shutil.which('codex'),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file() and os.access(candidate, os.X_OK):
            return str(Path(candidate).absolute())
    raise ReviewError('Codex was not found. Open or install the desktop app, then try again.')


class _Client:
    def __init__(self, executable, codex_home, cwd, timeout=12):
        self.timeout = timeout
        self.seq = 0
        self.buffer = b''
        self.process = subprocess.Popen(
            [executable, 'app-server', '--stdio'], cwd=cwd,
            env={**os.environ, 'CODEX_HOME': str(codex_home)},
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)

    def send(self, message):
        self.process.stdin.write((json.dumps(message) + '\n').encode())
        self.process.stdin.flush()

    def call(self, method, params):
        self.seq += 1
        request_id = self.seq
        self.send({'id': request_id, 'method': method, 'params': params})
        deadline = time.monotonic() + self.timeout
        while True:
            while b'\n' in self.buffer:
                line, self.buffer = self.buffer.split(b'\n', 1)
                if not line.strip():
                    continue
                message = json.loads(line)
                if 'method' in message:
                    if 'id' in message:
                        raise ReviewError('Codex requested an unexpected interaction during the status check.')
                    continue
                if message.get('id') == request_id:
                    if 'error' in message:
                        raise ReviewError('This Codex build could not inspect hooks. Use its native hook review.')
                    return message['result']
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not self.selector.select(remaining):
                raise ReviewError('Codex hook status timed out. Try again after Codex finishes starting.')
            chunk = os.read(self.process.stdout.fileno(), 65536)
            if not chunk:
                raise ReviewError('Codex stopped before returning hook status.')
            self.buffer += chunk

    def close(self):
        self.process.terminate()
        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        self.selector.close()
        self.process.stdin.close()
        self.process.stdout.close()


def summarize_hooks(reply, plugin_root=None):
    entries = reply.get('data', [])
    if len(entries) != 1 or entries[0].get('errors'):
        raise ReviewError('Codex reported hook configuration errors. Review them in Codex before continuing.')
    root = Path(plugin_root).resolve() if plugin_root else None
    hooks = []
    for hook in entries[0].get('hooks', []):
        if hook.get('source') != 'plugin' or not (hook.get('pluginId') or '').startswith('entity@'):
            continue
        if root is not None:
            try:
                Path(hook['sourcePath']).resolve().relative_to(root)
            except (KeyError, ValueError):
                continue
        hooks.append({key: hook.get(key) for key in
                      ('key', 'pluginId', 'eventName', 'enabled', 'trustStatus', 'currentHash')})
    pending = [h for h in hooks if h['trustStatus'] not in ('trusted', 'managed')]
    disabled = [h for h in hooks if h['enabled'] is not True]
    state = ('not_found' if not hooks else 'review_required' if any(h['enabled'] is True for h in pending) else
             'disabled' if disabled else 'definitions_trusted')
    return {'state': state, 'trust_complete': bool(hooks) and not pending and not disabled,
            'hooks': hooks, 'review_count': len(pending), 'disabled_count': len(disabled),
            'warnings': entries[0].get('warnings', []),
            'runtime_verified': False}


def hook_status(*, executable=None, codex_home=None, cwd=None, plugin_root=None):
    """Ask Codex for its current hashes and trust. Never infer trust from a stamp."""
    binary = find_codex(executable)
    home = Path(codex_home or os.environ.get('CODEX_HOME') or Path.home() / '.codex').absolute()
    directory = Path(cwd or Path.home()).absolute()
    client = _Client(binary, home, directory)
    try:
        client.call('initialize', {'clientInfo': {'name': 'entity-hook-review', 'version': '1'},
                                   'capabilities': {'experimentalApi': True}})
        client.send({'method': 'initialized'})
        return summarize_hooks(client.call('hooks/list', {'cwds': [str(directory)]}), plugin_root)
    finally:
        client.close()


def review_script(executable, codex_home, cwd):
    """A terminal receives the exact host configuration inspected by hook_status."""
    command = shlex.join([str(executable), '--no-alt-screen', '--cd', str(cwd)])
    return ('#!/bin/sh\n'
            'printf "%s\\n" "Entity automatic memory — Codex hook review" '
            '"Choose Review hooks in Codex and approve only the hooks you intend to run." '
            '"Codex may first ask you to confirm the working directory." '
            '"When finished, close this window and return to Entity setup to check again."\n'
            'export CODEX_HOME=' + shlex.quote(str(codex_home)) + '\n'
            'exec ' + command + '\n')


def open_review(*, executable=None, codex_home=None, cwd=None):
    """Launch native Codex after the caller obtains explicit user consent.

    macOS opens a .command document in Terminal. Other hosts get manual
    instructions. No input is sent to Codex, even if its startup UI changes.
    """
    if sys.platform != 'darwin':
        raise ReviewError('Automatic review launch is available on macOS. Open Codex CLI to review Entity hooks on this operating system.')
    binary = find_codex(executable)
    home = Path(codex_home or os.environ.get('CODEX_HOME') or Path.home() / '.codex').absolute()
    directory = Path(cwd or Path.home()).absolute()
    # Terminal reads this document after launch; the OS temp directory owns its
    # lifetime. It contains paths and no credentials.
    folder = Path(tempfile.mkdtemp(prefix='entity-hook-review-'))
    path = folder / 'Review Entity hooks.command'
    path.write_text(review_script(binary, home, directory))
    path.chmod(0o700)
    try:
        result = subprocess.run(['/usr/bin/open', '-a', 'Terminal', str(path)],
                                env={**os.environ, 'CODEX_HOME': str(home)},
                                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL, timeout=8, check=False)
        if result.returncode:
            raise ReviewError('Could not open Codex hook review. Your hook permissions were not changed.')
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ReviewError('Could not open Codex hook review. Your hook permissions were not changed.') from error
    return {'state': 'review_launched', 'trust_granted': False,
            'message': 'Complete the review in Codex, then check Entity setup again. Opening the review does not grant permission.'}

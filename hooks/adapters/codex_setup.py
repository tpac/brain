"""Session-local setup interactions, independent of the brain daemon.

The stdin owner routes responses here; waiting for consent never blocks that
reader. Only the matching, unexpired acceptance may open native hook review.
"""
import json
import sys
import threading
import uuid

# codex_onboarding.py owns host discovery, status, and native review launch.
import codex_onboarding as codex

SETUP_TOOL = {
    'name': 'setup',
    'description': ('Check Entity automatic-memory setup in Codex. Use action=status '
                    'when memory context or session attribution is missing. '
                    'action=review asks the user in-app before opening Codex native '
                    'hook review in a terminal window; no commands need to be typed. '
                    'Then call status again. This tool never grants hook trust.'),
    'inputSchema': {'type': 'object', 'properties': {
        'action': {'type': 'string', 'enum': ['status', 'review'], 'default': 'status'}},
        },
    'annotations': {'destructiveHint': False, 'openWorldHint': False},
}

IDENTITY_NOTICE = (
    'Entity could not verify session identity for this call. Automatic memory '
    'capture may be unavailable: hooks may need approval, be disabled, or be '
    'failing. Tell the user; in Codex call setup(action="status") to diagnose '
    'and setup(action="review") to offer native hook review. Do not claim '
    'automatic memory is active from an MCP connection alone.'
)


class SetupSession:
    tools = [SETUP_TOOL]
    instructions = (
        "If automatic memory context is absent or a tool reports missing session "
        "identity, tell the user and call setup(action='status'). Offer "
        "setup(action='review') when review is needed. An accepted confirmation "
        "does not grant hook trust. Check status after the user completes review."
    )
    notice = IDENTITY_NOTICE

    def handles(self, name):
        return name == 'setup'

    def __init__(self, send, plugin_root, timeout=120, log=None):
        self.send, self.plugin_root, self.timeout, self.log = send, plugin_root, timeout, log
        self.form_supported = False
        self.current = None
        self.lock = threading.RLock()

    def initialize(self, params):
        self.close()
        # The proxy has already filtered capabilities by the negotiated version.
        capability = params.get('capabilities', {}).get('elicitation')
        self.form_supported = (isinstance(capability, dict) and
                               (not capability or 'form' in capability))

    def _reply(self, request_id, payload, error=False):
        response = {'content': [{'type': 'text', 'text': json.dumps(payload)}]}
        if error:
            response['isError'] = True
            message = payload.get('message', payload.get('state', 'setup failed'))
            sys.stderr.write('[entity-setup] %s\n' % message)
            if self.log:
                self.log('mcp_setup', message, 'host setup', level='warning')
        self.send({'jsonrpc': '2.0', 'id': request_id, 'result': response})

    def result(self, op, payload, error=False):
        with self.lock:
            if self.current is not op:
                return
            self.close()
            self._reply(op['id'], payload, error)

    def _dispatch(self, job):
        worker = threading.Thread(target=job, daemon=True)
        worker.start()
        return worker

    def start(self, request_id, arguments, identity_verified=False):
        with self.lock:
            if self.current is not None:
                self._reply(request_id, {'state': 'busy', 'message': 'Entity setup is already checking or waiting for confirmation.'})
                return
            op = self.current = {'id': request_id, 'eid': None, 'timer': None}
        def inspect():
            try:
                self._start(op, arguments, identity_verified)
            except Exception as error:
                self.result(op, {'state': 'unknown', 'message': str(error)}, True)
        self._dispatch(inspect)

    def _start(self, op, arguments, identity_verified=False):
        action = arguments.get('action', 'status')
        if action not in ('status', 'review'):
            self.result(op, {'state': 'error', 'message': 'Use action=status or action=review.'}, True)
            return
        status = codex.hook_status(plugin_root=self.plugin_root)
        status['caller_identity_verified'] = identity_verified
        if action == 'status' or status['state'] in ('not_found', 'definitions_trusted', 'disabled'):
            status['message'] = {
                'not_found': 'Codex did not find this Entity installation’s hooks. Check that the plugin is installed and enabled.',
                'definitions_trusted': 'Codex reports these hook definitions trusted and enabled. Verify automatic recall and capture in the app before declaring memory ready.',
                'disabled': 'Some Entity hooks are disabled. Open Codex CLI and use /hooks to review their settings. The automatic startup review only covers new or changed enabled hooks.',
                'review_required': 'Entity hooks need review. Offer setup(action="review") to open Codex’s own approval screen.',
            }[status['state']]
            self.result(op, status)
            return
        if not self.form_supported:
            self.result(op, {'state': 'confirmation_unavailable', 'trust_granted': False,
                        'message': 'This connection does not support in-app confirmation. Open Codex CLI and review Entity under /hooks; this tool changed no permissions.'})
            return
        eid = 'entity-setup-' + uuid.uuid4().hex
        timer = threading.Timer(self.timeout, self._expire, args=(op,))
        timer.daemon = True
        request = {'jsonrpc': '2.0', 'id': eid, 'method': 'elicitation/create', 'params': {
            'mode': 'form',
            'message': ('Entity needs Codex hook approval for automatic memory. Open Codex’s '
                        'review in a terminal window? You will choose which hooks to trust '
                        'in Codex; accepting this form does not grant permission. No shell '
                        'commands need to be typed. Afterwards, return here to check setup.'),
            'requestedSchema': {'type': 'object', 'properties': {
                'open_review': {'type': 'boolean', 'title': 'Open Codex hook review', 'default': False}},
                'required': ['open_review']}}}
        with self.lock:
            if self.current is not op:
                return
            op.update(eid=eid, timer=timer)
            self.send(request)
            timer.start()

    def _expire(self, op):
        with self.lock:
            if self.current is op and op['eid']:
                self.result(op, {'state': 'confirmation_expired', 'trust_granted': False})

    def receive(self, message):
        with self.lock:
            op = self.current
            if op is None or not op['eid'] or message.get('id') != op['eid']:
                return False
            op['eid'] = None  # Consume before dispatch: replay and expiry cannot launch again.
            op['timer'].cancel()
        reply = message.get('result') or {}
        content = reply.get('content') or {} if isinstance(reply, dict) else {}
        accepted = (isinstance(reply, dict) and reply.get('action') == 'accept' and
                    isinstance(content, dict) and content.get('open_review') is True and
                    'error' not in message)
        if not accepted:
            self.result(op, {'state': 'review_cancelled', 'trust_granted': False})
            return True
        def launch():
            with self.lock:
                if self.current is not op:
                    return
            try:
                self.result(op, codex.open_review())
            except Exception as error:
                self.result(op, {'state': 'launch_failed', 'trust_granted': False, 'message': str(error)}, True)
        self._dispatch(launch)
        return True

    def cancel(self, request_id):
        with self.lock:
            if self.current is not None and self.current['id'] == request_id:
                self.close()

    def close(self):
        with self.lock:
            if self.current is not None and self.current['timer']:
                self.current['timer'].cancel()
            self.current = None


def create_extension(send, plugin_root, log=None):
    return SetupSession(send, plugin_root, log=log)

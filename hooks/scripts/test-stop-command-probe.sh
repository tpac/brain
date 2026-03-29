#!/bin/bash
# Minimal probe: does this command hook fire on Stop alongside the existing one?
echo "COMMAND_HOOK_FIRED $(date)" >> /tmp/brain-stop-command-probe.txt
echo '{"decision": "allow"}'

This is a diagnostic probe for the Stop agent hook.

YOUR TASK: Report what you can do. Execute these steps IN ORDER:

STEP 1: Write to a file to prove you're alive.
Use whatever file-writing tool you have to write the text "AGENT_ALIVE" followed by the current timestamp to /tmp/brain-stop-agent-probe.txt

STEP 2: List your available tools.
Report what tools you have access to. Try listing them or describe what you can do.

STEP 3: Try to reach the brain daemon.
Run this bash command:
python3 -c "import socket,json,time; s=socket.socket(); s.settimeout(3); s.connect(('127.0.0.1',47203)); s.sendall((json.dumps({'cmd':'ping'})+chr(10)).encode()); print(s.recv(4096).decode()); s.close()" 2>&1

STEP 4: Try to call a brain MCP tool.
If you have access to any tools with "brain" or "plugin" in the name, call recall with query "test probe".

STEP 5: Write your findings to /tmp/brain-stop-agent-results.txt
Include: what tools you found, whether bash worked, whether daemon responded, whether MCP tools were available.

After all steps, respond: {"decision": "allow"}

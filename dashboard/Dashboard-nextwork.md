# Dashboard — Next Work (Updated 2026-04-11)

## Done

### 3D Graph Visualization ✅
- **Live data** from brain.db via `/api/graph3d` endpoint
- **ForceGraph3D** from CDN — community-colored nodes, hub nodes larger
- **Community filtering** — only shows nodes in communities + hubs, hides orphans
- **Legend panel** — slide out from right with community names and member counts
- **Hover tooltips** — node name, type, community
- **Click → node detail** — loads full content, connections, metadata

### Dashboard Tabs ✅
- **Decoding tab** (was "Surface") — shows S1 recalls + S2 decode traces
- **Encoding tab** — shows S1 encoding activity
- **Scale filter** — "All scales" / "S1 Turn" / "S2 Graph"
- **S2 community traces** — O/K/Δ grouped by chain in the decoding feed

## Still To Do

### Timeline Slider
- Slider shows brain development over time
- Slide it and nodes pop in based on created_at
- Shows growth rate visually — when were bursts of encoding?

### Queryable
- Search bar — filter nodes by keyword in the graph view
- Filter by community, type, confidence range
- Highlight paths between two selected nodes

### Visual Polish
- Tighter community spacing — communities are spread out, needs tuning
- Locked nodes with gold ring
- Edge opacity based on weight
- Click community in legend → fly camera to that cluster
- Pulse animation on recently created/revised nodes
- Edge coloring by relation type (correction=red, extension=green, etc.)

### S2 Encoding Display
- Show S2CE community creation in the Encoding tab
- Badge for new communities created since last view
- Expand to see community content, members, narrative

### Node Detail Panel
- Currently uses the existing loadNodeDetail from the 2D graph era
- Needs update: show community membership, community narrative when viewing a community node
- Show "part of community: X" when viewing a member node

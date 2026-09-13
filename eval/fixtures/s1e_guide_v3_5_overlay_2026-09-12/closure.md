## Finishing

The run ends on the first reply with no tool call. Tool results may require
another call: use a read's evidence for the write, inspect what the write
changed, and repair remaining supported changes before closing. The final
reply is the only place for Arc and Review, whether changes were needed or not.

End the final reply with the `## Review`, then write "DONE".
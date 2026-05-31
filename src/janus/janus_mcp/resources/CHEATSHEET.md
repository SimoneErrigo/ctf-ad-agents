Janus filter DSL — short cheatsheet (full reference: resource `docs://janus/filters`)

Fields:
  url method status direction header header.<Name>
  service proto src dst peer sport dport
  body raw
  flagged contains_flagid dropped       # booleans, can be used as bare names

Operators:
  strings: contains  icontains  ==  !=  startswith  endswith  matches  in
  ints:    ==  !=  >  <  >=  <=  in
  ip:      ==  !=  in        (CIDR on src/dst/peer)
  logic:   AND  OR  NOT      (AND binds tighter than OR — parenthesize)

ALWAYS START SIMPLE — prefer plain substring filters over regex:
  body icontains "select"                          # case-insensitive substring
  body contains "UNION"                            # case-sensitive substring
  url icontains "../"
  body contains "<script"
Only escalate to `body matches "..."` (regex) when one keyword is not enough.
Most CTF SQLi/RCE/XSS payloads contain a giveaway substring (`select`, `union`,
`/etc/passwd`, `<script`, `' OR `, `--`, `sleep(`) — `icontains` finds them all.

Compound examples:
  service == "web1" AND method == "POST" AND status >= 400
  service == "web1" AND direction == "request" AND body icontains "select"
  contains_flagid AND direction == "request"
  (flagged OR contains_flagid) AND direction == "response"
  url startswith "/admin" AND NOT src in (10.10.0.0/16)
  method in ("POST","PUT","PATCH","DELETE") AND direction == "request"
  proto == "tcp" AND raw contains "AAAAAAAAAAAAAAAA"

Regex (only when needed):
  body matches "(?i)union\\s+select"               # DSL string, see escaping below
  url matches "/users/[a-f0-9-]{36}"

Escaping inside DSL strings — IMPORTANT:
  The DSL string only recognises \\n  \\"  \\\\  \\xHH  as escapes.
  Any other "\\X" sequence has its backslash silently dropped, which breaks
  regexes that need \\s, \\d, \\(, \\), etc. To embed those metacharacters,
  DOUBLE the backslash in the DSL string:
    DSL source:   body matches "(?i)sleep\\\\s*\\\\("
    means regex:  (?i)sleep\\s*\\(
  When you pass the whole q as a JSON tool argument, each DSL `\\\\` becomes
  `\\\\\\\\` in JSON. If you see validate errors like "missing closing )" or
  the regex coming back with the backslashes stripped, you under-escaped —
  add backslashes, do NOT add more parentheses.

Tips:
  - Regex is Go RE2 (no lookaround/backrefs). Anchor with ^ / $.
  - `contains` is case-sensitive; use `icontains` for ASCII case-insensitive.
  - Always run validate_filter before list_packets on non-trivial q.
  - `session_id` can appear in packet rows, but it is not filterable in q.
  - If a probe returns 0 packets, broaden (drop service / try `url` / try
    `icontains` / check response side) before concluding.

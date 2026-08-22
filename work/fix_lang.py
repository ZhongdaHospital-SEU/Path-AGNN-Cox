# -*- coding: utf-8 -*-
import io
p = "manuscript/Path-AGNN-Cox_manuscript_template.md"
s = io.open(p, encoding="utf-8").read()

# 1) discussion: remove parenthetical
old = "was not significant ({{ENRICH_LUAD_HITS}} hits, {{ENRICH_LUAD_P}}); an equivalent test"
new = "was not significant with {{ENRICH_LUAD_HITS}} hits at {{ENRICH_LUAD_P}}; an equivalent test"
assert old in s
s = s.replace(old, new, 1)

# 2) hedge: potentially in PathMoG comparison
old = "These differences position Path-AGNN-Cox as a lightweight, broadly deployable alternative"
new = "These differences position Path-AGNN-Cox as a lightweight alternative that is potentially more broadly deployable"
assert old in s
s = s.replace(old, new, 1)

# 3) hedge: seemingly in DCA sentence
old = "The incremental benefit was modest and threshold-dependent, consistent with the comparable discrimination"
new = "The incremental benefit was modest and threshold-dependent, seemingly consistent with the comparable discrimination"
assert old in s
s = s.replace(old, new, 1)

io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("language tweaks applied")

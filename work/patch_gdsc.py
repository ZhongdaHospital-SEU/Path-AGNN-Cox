import io
p = r"work/gdsc_validation.py"
t = io.open(p, encoding="utf-8").read()

old = '''templates = {}
for pw in sig_pw:
    genes = [g for g in gmt[pw] if g in keep]
    if len(genes) < 3:
        continue
    Ch = np.corrcoef(zscore_rows(hi[genes].to_numpy(dtype=float)).T)
    Cl = np.corrcoef(zscore_rows(lo[genes].to_numpy(dtype=float)).T)
    templates[pw] = (genes, Ch - Cl)
print("templates:", len(templates))

gene_list = list(expr_z.index)
col_idx = {g: i for i, g in enumerate(gene_list)}
expr_arr = expr_z.T
scores = {}
for pw, (genes, D) in templates.items():
    idx = [col_idx[g] for g in genes if g in col_idx]
    if len(idx) < 3:
        continue
    Xc = expr_arr[:, idx]
    Ds = D[np.ix_(idx, idx)]
    scores[pw] = np.einsum("ij,jk,ik->i", Xc, Ds, Xc)'''
new = '''def corr_matrix(X):
    Z = zscore_rows(X)
    C = Z.T @ Z / (Z.shape[0] - 1)
    return np.nan_to_num(C, nan=0.0)

templates = {}
for pw in sig_pw:
    genes = [g for g in gmt[pw] if g in keep]
    if len(genes) < 3:
        continue
    Ch = corr_matrix(hi[genes].to_numpy(dtype=float))
    Cl = corr_matrix(lo[genes].to_numpy(dtype=float))
    templates[pw] = (genes, Ch - Cl)
print("templates:", len(templates))

gdsc_genes = [g for g in expr_z.index if g in keep]
print("GDSC pathway genes present:", len(gdsc_genes))
expr_arr = expr_z.T  # cells x genes
scores = {}
for pw, (genes, D) in templates.items():
    genes = [g for g in genes if g in expr_arr.columns]
    if len(genes) < 3:
        continue
    Xc = expr_arr[genes].to_numpy()
    Ds = D[np.ix_([g for g in gmt[pw] if g in keep].index if False else [genes.index(g) for g in genes], [genes.index(g) for g in genes])]
    scores[pw] = np.einsum("ij,jk,ik->i", Xc, Ds, Xc)'''
assert t.count(old) == 1
t = t.replace(old, new)
io.open(p, "w", encoding="utf-8").write(t)
print("patched")

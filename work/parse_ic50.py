import csv
p = r"work/pkg/GDSC2/DataFiles/DataFiles/GLDS/GDSCv2/complete_matrix_output GDSCv2.txt"
with open(p, newline="", encoding="utf-8") as f:
    rows = list(csv.reader(f, delimiter=" ", quotechar='"'))
print("n rows:", len(rows))
print("hdr n:", len(rows[0]), "first 6:", rows[0][:6])
print("row1 n:", len(rows[1]), "first 3:", rows[1][:3])

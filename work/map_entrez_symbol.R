suppressMessages(library(org.Hs.eg.db))
args <- commandArgs(trailingOnly = TRUE)
fin <- args[1]; fout <- args[2]
ids <- readLines(fin, warn = FALSE)
m <- select(org.Hs.eg.db, keys = ids, keytype = 'ENTREZID', columns = 'SYMBOL')
write.csv(m, fout, row.names = FALSE, quote = FALSE)
cat('mapped', nrow(m), '
')

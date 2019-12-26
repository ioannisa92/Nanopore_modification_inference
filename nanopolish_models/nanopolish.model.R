cpg <- read.delim("./original/r9.4_450bps.cpg.6mer.template.model", header=FALSE, comment.char="#", stringsAsFactors=FALSE)
native <- read.delim("../ont_models/r9.4_180mv_450bps_6mer_DNA.model", header=FALSE, comment.char="#", stringsAsFactors=FALSE)

flag <- rep(0, nrow(cpg))
flag[grepl("MG", cpg$V1) & !grepl("MA", cpg$V1) & !grepl("MT", cpg$V1) & !grepl("MM", cpg$V1) & !grepl("MC", cpg$V1)] <- 1
flag[flag == 1 & grepl("C", cpg$V1)] <- 0

pdf("nanopolish.model.pdf", width = 10, height = 5)
par(mfrow = c(1, 2), mar = c(5, 5, 5, 3))
plot(cpg$V2[flag == 1], cpg$V2[match(gsub("M", "C", cpg$V1[flag == 1]), cpg$V1)],
     xlab = "MG kmer", ylab = "CG kmer", pch = 16, cex = 0.5, cex.lab = 2, main = "nanopolish model", cex.main = 2)
abline(0, 1, NULL, NULL, col = 2)
plot(cpg$V2[match(gsub("M", "C", cpg$V1[flag == 1]), cpg$V1)], native$V2[match(gsub("M", "C", cpg$V1[flag == 1]), native$V1)],
     xlab = "nanopolish model", ylab = "ont model", pch = 16, cex = 0.5, cex.lab = 2, main = "CG kmer", cex.main = 2)
abline(0, 1, NULL, NULL, col = 2)
dev.off()

table <- cpg[flag == 1, ]
write.table(table, file = "./processed/r9.4_450bps.mpg.6mer.template.model", quote = F, row.names = F, col.names = F)
table <- cpg[match(gsub("M", "C", cpg$V1[flag == 1]), cpg$V1), ]
write.table(table, file = "./processed/r9.4_450bps.cpg.6mer.template.model", quote = F, row.names = F, col.names = F)


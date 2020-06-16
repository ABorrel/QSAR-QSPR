#!/usr/bin/env Rscript
library(ggplot2)


################
#     MAIN     #
################
args <- commandArgs(TRUE)
pImportance = args[1]
nbconsidered = as.integer(args[2])

#pImportance = "C:/Users/Aborrel/research/ILS/HERG/results/QSAR/Merge_involvedDesc/RF/Av_importance"
#nbconsidered = 10

dimportance = read.table(pImportance, sep = "\t", header = TRUE)
dimportance$Desc = as.factor(dimportance$Desc)
dimportance$Run = as.character(dimportance$Run)
dimportance$val = as.double(scale(dimportance$val))

# order
ldesc = unique(dimportance$Desc)
lMd = NULL
for(desc in ldesc){
  md = median(dimportance$val[which(dimportance$Desc == desc)])
  lMd = append(lMd, abs(md))
}
names(lMd) = ldesc
lMd = sort(lMd, decreasing = TRUE)
ldesc = names(lMd)[1:nbconsidered]

dtop = NULL
for(desc in ldesc){
  dtop = rbind(dtop, dimportance[which(dimportance$Desc == desc),])
}




ggplot(dtop, aes(val, Desc))+
  geom_point(aes(color = Run))+labs(x = "Importance", y = "Descriptor") +
  theme(axis.text.y = element_text(size = 12, hjust = 0.5, vjust =0.1), axis.text.x = element_text(size = 12, hjust = 0.5, vjust =0.1), axis.title.y = element_text(size = 14, hjust = 0.5, vjust =0.1), axis.title.x =  element_text(size = 14, hjust = 0.5, vjust =0.1))

ggsave(paste(pImportance, ".png", sep = ""), dpi=300, height = 7, width = 6)

#!/usr/bin/env Rscript
source("performance.R")
library(svglite)




################
#     MAIN     #
################
args <- commandArgs(TRUE)
ppred = args[1]
nameout = args[2]
prout = args[3]

#ppred = "/home/borrela2/imatinib/DL/Split/Lig3D-Lig2D-Lig-BS_IC50/trainfit_testpredictions.csv"
#nameout = "Train"
#prout = "/home/borrela2/imatinib/DL/Split/Lig3D-Lig2D-Lig-BS_IC50/"

# openning
dpred = read.csv(ppred)
rownames(dpred) = dpred[,1]
colnames(dpred) = c("ID", "Yreal", "Ypredict")
vreal =  dpred[,2]
vpred = dpred[,3]

# performance
corpred = cor(vreal, vpred)
rmsep = vrmsep(vreal, vpred)
R2pred = calR2(vreal, vpred)
MAEpred = MAE(vreal, vpred)
R02pred = R02(vreal, vpred)

lperf = c(round(R2pred,2), round(R02pred,2), round(MAEpred,2), round(corpred,2))
names(lperf) = c("R2", "R02", "MAE", "r")
write.csv(t(lperf), paste(prout, "perf", nameout, ".csv", sep = ""))


# plot prediction
theme_set(theme_grey())
p = ggplot(dpred, aes(Yreal, Ypredict))+
  geom_point(size=1.5, colour="black", shape=21) + 
  geom_text(x=4.8, y=9.7, label = paste("R2=",round(R2pred,2), sep = ""), size = 8)+
  labs(x = "Experimental pAff", y = "Predicted pAff") +
  theme(axis.text.y = element_text(size = 25, hjust = 0.5, vjust =0.1), axis.text.x = element_text(size = 25, hjust = 0.5, vjust =0.1), axis.title.y = element_text(size = 25, hjust = 0.5, vjust =0.1), axis.title.x =  element_text(size = 25, hjust = 0.5, vjust =0.1))+
  xlim (c(4, 12)) +
  geom_segment(aes(x = 4, y = 4, xend = 12, yend = 12), linetype=1, size = 0.1) + 
  ylim (c(4, 12))
ggsave(paste(prout, "plotPoint", nameout, ".png", sep = ""), width = 6,height = 6, dpi = 300)


theme_set(theme_grey())
p = ggplot(dpred, aes(Yreal, Ypredict))+
  #geom_point(size=1.5, colour="black", shape=21) + 
  geom_text(x=4.8, y=9.7, label = paste("R2=",round(R2pred,2), sep = ""), size = 8)+
  geom_text(x=dpred$Yreal, y=dpred$Ypredict, label=dpred$ID)+
  labs(x = "Experimental pAff", y = "Predicted pAff") +
  theme(axis.text.y = element_text(size = 25, hjust = 0.5, vjust =0.1), axis.text.x = element_text(size = 25, hjust = 0.5, vjust =0.1), axis.title.y = element_text(size = 25, hjust = 0.5, vjust =0.1), axis.title.x =  element_text(size = 25, hjust = 0.5, vjust =0.1))+
  xlim (c(4, 12)) +
  geom_segment(aes(x = 4, y = 4, xend = 12, yend = 12), linetype=1, size = 0.1) + 
  ylim (c(4, 12))
ggsave(paste(prout, "plotName", nameout, ".svg", sep = ""), width = 10,height = 10, dpi = 300)



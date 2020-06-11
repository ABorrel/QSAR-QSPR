#!/usr/bin/env Rscript
source("performance.R")




################
#     MAIN     #
################
args <- commandArgs(TRUE)
pprob = args[1]
pout = args[2]

#pprob = "/home/borrela2/interference/testing/588_resorufin/HepG2/cell_red_n/CARTclass/sumProb"
#pout = "/home/borrela2/interference/testing/588_resorufin/HepG2/cell_red_n/CARTclass/sumProb_perf.csv"

dprob = read.csv(pprob, sep = ",", header =TRUE)
rownames(dprob) = dprob[,1]
dprob = dprob[,-1]
dprob = na.omit(dprob)

lclass = dprob[,1]
lclass[which(dprob[,1] < 0.5)] = 0
lclass[which(dprob[,1] >= 0.5)] = 1

# criteria quality
criteriaQuality = perftable(lclass, dprob[,3])
print(criteriaQuality)
TP = criteriaQuality[[1]]
TN = criteriaQuality[[2]]
FP = criteriaQuality[[3]]
FN = criteriaQuality[[4]]


lperf = classPerf(dprob[,3], lclass)
acc = lperf[[1]]
se = lperf[[2]]
sp = lperf[[3]]
mcc = lperf[[4]]

MpbTN = mean(dprob[which(dprob[,3] == 0 & lclass == 0), 1])
SDpbTN = sd(dprob[which(dprob[,3] == 0 & lclass == 0), 1])
MpbFP = mean(dprob[which(dprob[,3] == 0 & lclass == 1), 1])
SDpbFP = sd(dprob[which(dprob[,3] == 0 & lclass == 1), 1])

MpbTP = mean(dprob[which(dprob[,3] == 1 & lclass == 1), 1])
SDpbTP = sd(dprob[which(dprob[,3] == 1 & lclass == 1), 1])
MpbFN = mean(dprob[which(dprob[,3] == 1 & lclass == 0), 1])
SDpbFN = sd(dprob[which(dprob[,3] == 1 & lclass == 0), 1])


dout = rbind(c("TP", "TN", "FP", "FN", "acc", "se", "sp", "mcc", "MpbTP", "SDpbTP", "MpbTN", "SDpbTN", "MpbFP", "SDpbFP", "MpbFN", "SDpbFN"), c(TP, TN, FP, FN, acc, se, sp, mcc, MpbTP, SDpbTP, MpbTN, SDpbTN, MpbFP, SDpbFP, MpbFN, SDpbFN ))
colnames(dout) = dout[1,]
dout=dout[-1,]
write.csv(dout, pout)

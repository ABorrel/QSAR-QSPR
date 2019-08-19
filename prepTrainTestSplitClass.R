#!/usr/bin/env Rscript
source ("~/development/Rglobal/source/dataManager.R")


################
#     MAIN     #
################

args <- commandArgs(TRUE)
pdesc = args[1]
splitFact = as.double(args[2])
prout = args[3]

din = read.csv(pdesc, sep = "\t", header = TRUE)

# first split data by class
ldclasses = separeData (din, "Aff")
d1 = ldclasses[[1]]
d2 = ldclasses[[2]]

ld1split = samplingDataFraction(d1, splitFact)
ld2split = samplingDataFraction(d2, splitFact)

dtrain = rbind(ld1split[[1]], ld2split[[1]])
dtest = rbind(ld1split[[2]], ld2split[[2]])

write.csv(dtrain, paste(prout, "train.csv", sep = ""))
write.csv(dtest, paste(prout, "test.csv", sep = ""))


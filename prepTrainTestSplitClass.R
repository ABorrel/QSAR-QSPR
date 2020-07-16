#!/usr/bin/env Rscript
source("./../R_toolbox/dataManager.R")


################
#     MAIN     #
################

args <- commandArgs(TRUE)
pdesc = args[1]
splitFact = as.double(args[2])
prout = args[3]

#pdesc = "C:/Users/Aborrel/research/ILS/HERG/results/QSAR/1/desc_Class.csv"
#prout = "C:/Users/Aborrel/research/ILS/HERG/results/QSAR/1/"
#splitFact = 0.15

din = read.csv(pdesc, sep = "\t", header = TRUE)

# in case of split equal 0 => rewrite train
if (splitFact == 0){
  write.csv(din, paste(prout, "train.csv", sep = ""), row.names = FALSE)
}else{
  # first split data by class
  ldclasses = separeData (din, "Aff")
  d1 = ldclasses[[1]]
  d2 = ldclasses[[2]]
  
  ld1split = samplingDataFraction(d1, splitFact)
  ld2split = samplingDataFraction(d2, splitFact)
  
  dtrain = rbind(ld1split[[1]], ld2split[[1]])
  dtest = rbind(ld1split[[2]], ld2split[[2]])
  rownames(dtrain) = dtrain[,1]
  rownames(dtest) = dtest[,1]
  
  write.csv(dtrain, paste(prout, "train.csv", sep = ""), row.names = FALSE)
  write.csv(dtest, paste(prout, "test.csv", sep = ""), row.names = FALSE)
}




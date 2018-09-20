#!/usr/bin/env Rscript
source("MLClassification.R")


################
#     MAIN     #
################
args <- commandArgs(TRUE)
pmodel = args[1]
ML = args[2]
ptrain = args[3]
prout = args[4]


#pmodel = "/home/borrela2/interference/spDataAnalysis/QSARclassCrossColor/1/crossColor/LDAclass/model.RData"
#ML = "LDA"
#prout = "/home/borrela2/interference/spDataAnalysis/QSARclassCrossColor/1/crossColor/LDAclass/"
#ptrain = "/home/borrela2/interference/spDataAnalysis/QSARclassCrossColor/1/crossColor/trainSet.csv"


load(pmodel)
model = outmodel$model

if(ML == "LDA"){
  
  #open train
  dtrain = read.csv(ptrain, header = TRUE)
  rownames(dtrain) = dtrain[,1]
  dtrain = dtrain[,-1]
  
  # need normalized desc
  lscaling = model$scaling[,1]
  
  lcoefNorm = normalizationScalingLDA(lscaling, dtrain)

  write.table(lcoefNorm, file = paste(prout, "ImportanceDesc", sep = ""), sep = "\t")
  
}
if(ML == "RF"){
  
  timportance = model$importance[,1]
  write.table(timportance, paste(prout, "ImportanceDesc", sep = ""), sep = "\t")
  
}
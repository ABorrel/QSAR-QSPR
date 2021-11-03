#!/usr/bin/env Rscript
library(randomForest)
library(MASS)
library(rpart)
library(rpart.plot)
library(e1071)
#library(neuralnet)
library(nnet)
library(clusterGeneration)
library(stringr)
library(reshape2)



################
#     MAIN     #
################
args <- commandArgs(TRUE)
pdesc = args[1]
pmodel = args[2]
ML = args[3]
pout = args[4]


#pdesc = "/home/borrela2/interference/testing/588_resorufin/descMat"
#pmodel = "/home/borrela2/interference/testing/QSARmodel/HepG2/cell_red_n/RFclass/model4.Rdata"
#ML = "RFclass"
#pout = "/home/borrela2/interference/testing/588_resorufin/HepG2/cell_red_n/RFclass/perf_RFclass_model4.csv"

#pdesc = "/home/borrela2/interference/testing/588_resorufin/descMat"
#pmodel = "/home/borrela2/interference/testing/QSARmodel/HepG2/cell_red_n/CARTclass/model1.Rdata"
#ML = "CARTclass"
#prout = "/home/borrela2/interference/testing/588_resorufin/HepG2/cell_red_n/CARTclass/"

#pdesc = "/home/borrela2/interference/testing/588_resorufin/descMat"
#pmodel ="/home/borrela2/interference/testing/QSARmodel/HepG2/cell_red_n/LDAclass/model3.Rdata" 
#ML = "LDAclass" 
#pout = "/home/borrela2/interference/testing/588_resorufin/HepG2/cell_red_n/LDAclass/perf_LDAclass_model3.csv"



load(pmodel)
model = outmodel$model
din = read.csv(pdesc, sep = ",", header = TRUE)
if(dim(din)[2] == 1){
  din = read.csv(pdesc, sep = "\t", header = TRUE)  
}
rownames(din) = din[,1]
din = din[,-1]

# if not aff in colnames
if("Aff" %in% colnames(din) == FALSE ){
  Aff = rep(1, dim(din)[1])
  din = cbind(din, Aff)
}


if(ML == "RFclass"){
  lpred = predict(model, din)
  dout = cbind(lpred, din$Aff)
}else if (ML == "CARTclass"){
  lpred = predict(model, din)
  dout = cbind(lpred[,2], din$Aff)
}else if(ML == "LDAclass"){
  lpred = predict(model, din)
  dout = cbind(lpred$posterior[,2], din$Aff)
}else if(ML == "NNclass"){
  lpred = predict(model, din[,-c(which(colnames(din) == "Aff"))])
  lpred = as.double(as.character(lpred))
  dout = cbind(lpred, din$Aff) 
}else if(ML == "SVMclass"){
  lpred = predict(model, din)
  lpred = as.vector(lpred)
  dout = cbind(lpred, din$Aff)
}



colnames(dout) = c("Pred", "Aff")
rownames(dout) = rownames(din)

write.csv(dout, pout)

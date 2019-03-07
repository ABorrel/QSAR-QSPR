#!/usr/bin/env Rscript
source ("~/development/Rglobal/source/dataManager.R")
source("MachinLearning.R")
source("performance.R")

library(chemmodlab)
library (rpart)



################
#     MAIN     #
################
args <- commandArgs(TRUE)
ptrain = args[1]
ptest = args[2]
pcluster = args[3]
prout = args[4]
nbCV = as.integer(args[5])
internalCV = as.integer(args[6])


# to test
#ptrain = "/home/borrela2/imatinib/results/analysis/QSARs/Lig2D/trainSet.csv"
#ptest = "/home/borrela2/imatinib/results/analysis/QSARs/Lig2D/testSet.csv"
#pcluster = "0"
#prout = "/home/borrela2/imatinib/results/analysis/QSARs/Lig2D/"
# cross validation 10
#nbCV = 10


# model regression #
####################
modelPCRreg = 1 
modelPLSreg = 1
modelSVMreg = 1
modelRFreg = 1
modelCartreg = 1
modelNNreg = 1
modelDLreg = 0 #old creation of DNN using R
chemmodlabreg = 0



#########################
#    PRINT PARAMETERS   #
#########################

print("=====PARAMETERS=====")
print (paste("Train csv: ", ptrain, sep = ""))
print (paste("Test csv: ", ptest, sep = ""))
print (paste("Folder out: ", prout, sep = ""))
print(paste("Nb of CV: ", nbCV, sep = ""))
print("")

print("=====Machine learning=====")
print("---Regression model----")
print(paste("PCR: ", modelPCRreg, sep = ""))
print(paste("PLS: ", modelPLSreg, sep = ""))
print(paste("SVM: ", modelSVMreg, sep = ""))
print(paste("CART: ", modelCartreg, sep = ""))
print(paste("RF: ", modelRFreg, sep = ""))
print(paste("NN: ", modelNNreg, sep = ""))
print(paste("DP: ", modelDLreg, sep = ""))
print(paste("Chemmodlab: ", chemmodlabreg, sep = ""))
print("")



##############################
# Process descriptors matrix #
##############################

# training set
dtrain = read.csv(ptrain, header = TRUE)
rownames(dtrain) = dtrain[,1]
dtrain = dtrain[,-1]

# test set
dtest = read.csv(ptest, header = TRUE)
rownames(dtest) = dtest[,1]
dtest = dtest[,-1]

# dglobal for CV
if(internalCV == 1){
  dglobal = dtrain
}else{
  dglobal = rbind(dtrain, dtest[,colnames(dtrain)])  
}


# cluster
if (pcluster != "0"){
  dcluster = read.csv(pcluster, header = TRUE)
  namescpd = dcluster[,1]
  dcluster = dcluster[,-1]
  names(dcluster) = namescpd
}else{
  namescpd = cbind(rownames(dtrain), rownames(dtest))
  dcluster = rep(1, length(namescpd))
  names(dcluster) = namescpd
}


print("==== Dataset ====")
print(paste("Data train: dim = ", dim(dtrain)[1], dim(dtrain)[2], sep = " "))
print(paste("Data test: dim = ", dim(dtest)[1], dim(dtest)[2], sep = " "))
print("")

# sampling data for CV #
########################
lgroupCV = samplingDataNgroup(dglobal, nbCV)
controlDatasets(lgroupCV, paste(prout, "ChecksamplingCV", nbCV, sep = ""))


##### REGRESSION MODELS #########
#################################
print("**************************")
print("*****  REGRESSSION   *****")
print("**************************")

### PCR ####
############

if (modelPCRreg == 1){
  
  # control number of descriptors VS number of data
  nbdesc = dim(dtrain)[2]
  nbchemical = dim(dtrain)[1]
  print ((100-nbCV)/100*(nbchemical))
  if(((100-nbCV)/100*(nbchemical)) <= nbdesc){
    ldesc = reduceNBdesc(dtrain, nbCV)
  }else{
    ldesc = colnames(dtrain)
  }
  dtrainPCR = dtrain[,ldesc]
  dglobalPCR = dglobal[,ldesc]
  dtestPCR = dtest[,ldesc]
  lgroupCVPCR = samplingDataNgroup(dglobalPCR, nbCV)
  
  print(paste("Nb descriptors PCR: ", dim(dtrainPCR)[2], sep = ""))
  print(paste("Nb chemical PCR: ", dim(dtrainPCR)[1], sep = ""))
  
  proutPCR = paste(prout, "PCRreg/", sep = "")
  dir.create(proutPCR)
  nbCp = PCRgridCV(lgroupCVPCR, proutPCR)
  outPCRCV = PCRCV(lgroupCVPCR, nbCp, dcluster, proutPCR)
  outPCR = PCRTrainTest(dtrainPCR, dtestPCR, dcluster, nbCp, proutPCR)
}

### PLS  ####
#############

if (modelPLSreg == 1){
  
  # control number of descriptors VS number of data
  nbdesc = dim(dtrain)[2]
  nbchemical = dim(dtrain)[1]
  if(((100-nbCV)/100*nbchemical) <= nbdesc){
    ldesc = reduceNBdesc(dtrain, nbCV)
  }else{
    ldesc = colnames(dtrain)
  }
  dtrainPLS = dtrain[,ldesc]
  dglobalPLS = dglobal[,ldesc]
  dtestPLS = dtest[,ldesc]
  lgroupCVPLS = samplingDataNgroup(dglobalPLS, nbCV)
  
  print(paste("Nb descriptors PLS: ", dim(dtrainPLS)[2], sep = ""))
  print(paste("Nb chemical PLS: ", dim(dtrainPLS)[1], sep = ""))
  
  proutPLS = paste(prout, "PLSreg/", sep = "")
  dir.create(proutPLS)
  outPLSCV = PLSCV(lgroupCVPLS, dcluster, proutPLS)
  #have to finish
  outPLS = PLSTrainTest(dtrainPLS, dtestPLS, dcluster, outPLSCV$nbcp, proutPLS)
}

### SVM ###
###########

if(modelSVMreg == 1){
  proutSVM = paste(prout, "SVMreg/", sep = "")
  dir.create(proutSVM)
  vgamma = 2^(-1:1)
  vcost = 2^(2:8)
  outSVMCV = SVMRegCV(lgroupCV, vgamma, vcost, dcluster, proutSVM)
  outSVM = SVMRegTrainTest(dtrain, dtest, vgamma, vcost, dcluster, proutSVM)
}

######
# RF #
######

if (modelRFreg == 1){
  proutRF = paste(prout, "RFreg/", sep = "")
  dir.create(proutRF)
  
  vntree = c(10,50,100,200,500, 1000)
  vmtry = c(1,2,3,4,5,10,15,20, 25, 30)
  
  #RFregCV(lgroupCV, 50, 5, dcluster, prout)# for test
  parameters = RFGridRegCV(vntree, vmtry, lgroupCV,  proutRF)
  outRFCV = RFregCV(lgroupCV, parameters[[1]], parameters[[2]], dcluster, proutRF)
  outRF = RFreg(dtrain, dtest, parameters[[1]], parameters[[2]], dcluster, proutRF)
}



############
#   CART   #
############

if(modelCartreg == 1){
  proutCART = paste(prout, "CARTreg/", sep = "")
  dir.create(proutCART)
  outCARTCV = CARTRegCV(lgroupCV, dcluster, proutCART)
  outCART = CARTreg(dtrain, dtest, dcluster, proutCART)
}


#############
# CHEMMOLAB #
#############

if(chemmodlabreg == 1){
  dchem = cbind(dtrain[,dim(dtrain)[2]],dtrain[,-dim(dtrain)[2]] )
  colnames(dchem)[1] = "Aff"
  
  pdf(paste(prout, "Reg_chemmolab.pdf", sep = ""))
  fit = ModelTrain(dchem, ids = FALSE)
  CombineSplits(fit, metric = "R2")
  CombineSplits(fit, metric = "rho")
  dev.off()
}



####################
#  NEURAL NETWORK  #
####################


if(modelNNreg ==1){
  vsize = c(1,2,5,10)
  vdecay = c(1e-6, 1e-4, 1e-2, 1e-1, 1)
  proutNN = paste(prout, "NNreg/", sep = "")
  dir.create(proutNN)
  vsize = c(1,2,5,10)
  vdecay = c(1e-6, 1e-4)
  
  print("AAAAA")
  
  outNNCV = NNRegCV(lgroupCV, dcluster, vdecay, vsize, proutNN)
  outNN = NNReg(dtrain, dtest, dcluster,vdecay, vsize, proutNN)
}


####################
#  DEEP LEARNING   #
####################

# treated with keras in python

#if(modelDLreg ==1){
#  #DLRegCV(lgroupCV, dcluster, prout)
#  DLReg(dtrain, dtest, dcluster, prout)
#}


#############################
# merge table of perfomance #
#############################

perfCV = NULL
perftrain = NULL
perfTest = NULL
rownameTable = NULL

if(modelPCRreg==1){
  perfCV = rbind(perfCV, outPCRCV$CV)
  perftrain = rbind(perftrain, outPCR$train)
  perfTest = rbind(perfTest, outPCR$test)
  rownameTable = append(rownameTable, "PCR")
}

if(modelPLSreg==1){
  perfCV = rbind(perfCV, outPLSCV$CV)
  perftrain = rbind(perftrain, outPLS$train)
  perfTest = rbind(perfTest, outPLS$test)
  rownameTable = append(rownameTable, "PLS")
}

if(modelSVMreg==1){
  perfCV = rbind(perfCV, outSVMCV$CV)
  perftrain = rbind(perftrain, outSVM$train)
  perfTest = rbind(perfTest, outSVM$test)
  rownameTable = append(rownameTable, "SVM")
}

if(modelRFreg==1){
  perfCV = rbind(perfCV, outRFCV$CV)
  perftrain = rbind(perftrain, outRF$train)
  perfTest = rbind(perfTest, outRF$test)
  rownameTable = append(rownameTable, "RF")
}

if(modelCartreg==1){
  perfCV = rbind(perfCV, outCARTCV$CV)
  perftrain = rbind(perftrain, outCART$train)
  perfTest = rbind(perfTest, outCART$test)
  rownameTable = append(rownameTable, "CART")
}

if(modelNNreg==1){
  perfCV = rbind(perfCV, outNNCV$CV)
  perftrain = rbind(perftrain, outNN$train)
  perfTest = rbind(perfTest, outNN$test)
  rownameTable = append(rownameTable, "NN")
}

rownames(perfTest) = rownameTable
rownames(perftrain) = rownameTable
rownames(perfCV) = rownameTable

# round
perfTest = round(perfTest,3)
perftrain = round(perftrain,3)
perfCV = round(perfCV,3)

colnames(perfTest) = c("R2", "R02", "MAE", "r", "RMSEP")
colnames(perftrain) = c("R2", "R02", "MAE", "r", "RMSEP")
colnames(perfCV) = c("R2", "R02", "MAE", "r", "RMSEP")

write.csv(perfTest, file = paste(prout, "perfTest.csv", sep = ""))
write.csv(perftrain, file = paste(prout, "perfTrain.csv", sep = ""))
write.csv(perfCV, file = paste(prout, "perfCV.csv", sep = ""))







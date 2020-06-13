#!/usr/bin/env Rscript
source("./../R_toolbox/dataManager.R")
source("MLClassification.R")
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


#ptrain = "C:/Users/Aborrel/research/ILS/HERG/results/QSAR/1/train.csv" 
#ptest = "C:/Users/Aborrel/research/ILS/HERG/results/QSAR/1/test.csv" 
#pcluster = "0"
#prout = "C:/Users/Aborrel/research/ILS/HERG/results/QSAR/1/"
#nbCV = 10


# to test
#ptrain = "/home/borrela2/imatinib/results/analysis/QSARs/Lig2D/trainSet.csv"
#ptest = "/home/borrela2/imatinib/results/analysis/QSARs/Lig2D/testSet.csv"
#pcluster = "0"
#prout = "/home/borrela2/imatinib/results/analysis/QSARs/Lig2D/"

# cross validation 10
#nbCV = 10


# model classification #
########################
modelSVMclass = 1
modelRFclass = 1
modelLDAclass = 1
modelCartclass = 1
modelNNclass = 1
chemmodlabclass = 0



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
print("---Classification model----")
print(paste("SVM: ", modelSVMclass, sep = ""))
print(paste("CART: ", modelCartclass, sep = ""))
print(paste("LDA: ", modelLDAclass, sep = ""))
print(paste("RF: ", modelRFclass, sep = ""))
print(paste("NN: ", modelNNclass, sep = ""))
print(paste("Chemmodlab: ", chemmodlabclass, sep = ""))
print("")



##############################
# Process descriptors matrix #
##############################

# training set
dtrain = read.csv(ptrain, header = TRUE)
rownames(dtrain) = dtrain[,1]
dtrain = dtrain[,-1]
# have to check the null variance
dtrain = delSDNull(dtrain)

# test set
dtest = read.csv(ptest, header = TRUE)
rownames(dtest) = dtest[,1]
dtest = dtest[,-1]

# dglobal for CV
dglobal = rbind(dtrain, dtest[,colnames(dtrain)])

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


##### CLASSIF  MODELS #########
###############################
print("*****************************")
print("*****  CLASSIFICATION   *****")
print("*****************************")



if(modelCartclass == 1){
  prCART = paste(prout, "CARTclass/", sep = "")
  dir.create(prCART)
  outCART = CARTclass(dtrain, dtest, prCART)
  outCARTCV = CARTClassCV(lgroupCV, prCART)
}


if(modelSVMclass == 1){
  
  prSVM = paste(prout, "SVMclass_linear/", sep = "")
  dir.create(prSVM)
  vgamma = 2^(-1:1)
  vcost = 2^(2:8)
  ksvm = "linear"
  outSVMCV_linear = SVMClassCV(lgroupCV, vgamma, vcost, ksvm, prSVM)
  outSVM_linear = SVMClassTrainTest(dtrain, dtest, vgamma, vcost, ksvm, prSVM)
  
  prSVM = paste(prout, "SVMclass_radial/", sep = "")
  dir.create(prSVM)
  ksvm = "radial"
  outSVMCV_radial = SVMClassCV(lgroupCV, vgamma, vcost, ksvm, prSVM)
  outSVM_radial = SVMClassTrainTest(dtrain, dtest, vgamma, vcost, ksvm, prSVM)
  
  prSVM = paste(prout, "SVMclass_sigmoid/", sep = "")
  dir.create(prSVM)
  ksvm = "sigmoid"
  outSVMCV_sigmoid = SVMClassCV(lgroupCV, vgamma, vcost, ksvm, prSVM)
  outSVM_sigmoid = SVMClassTrainTest(dtrain, dtest, vgamma, vcost, ksvm, prSVM)
  
}



if(modelRFclass == 1){
  prRF = paste(prout, "RFclass/", sep = "")
  dir.create(prRF)
  vntree = c(10,50,100,200,500)
  vmtry = c(1,2,3,4,5,10,15,20,25,30)
  
  parameters = RFGridClassCV(vntree, vmtry, lgroupCV,  prRF)
  outRFCV = RFClassCV(lgroupCV, parameters[[1]], parameters[[2]], prRF)
  outRF = RFClassTrainTest(dtrain, dtest, parameters[[1]], parameters[[2]], prRF)
}


if(modelLDAclass == 1){
  prLDA = paste(prout, "LDAclass/", sep = "")
  dir.create(prLDA)
  outLDACV = LDAClassCV(lgroupCV, prLDA)
  outLDA = LDAClassTrainTest(dtrain, dtest, prLDA)
}


if(modelNNclass == 1){
  prNN = paste(prout, "NNclass/", sep = "")
  dir.create(prNN)
  #vsize = c(1,2,5,10)
  #vdecay = c(1e-6, 1e-4, 1e-2, 1e-1, 1)
  vsize = c(1,2,5)
  vdecay = c(1e-1, 0.5, 1)
  outNNCV = NNClassCV(lgroupCV, vsize, vdecay, prNN)
  outNN = NNClassTrainTest(dtrain, dtest, vsize, vdecay, prNN)
}


#############################
# merge table of perfomance #
#############################

perfCV = NULL
perftrain = NULL
perfTest = NULL
rownameTable = NULL

if(modelCartclass==1){
  perfCV = rbind(perfCV, outCARTCV$CV)
  perftrain = rbind(perftrain, outCART$train)
  perfTest = rbind(perfTest, outCART$test)
  rownameTable = append(rownameTable, "CART")
}


if(modelNNclass == 1){
  perfCV = rbind(perfCV, outNNCV$CV)
  perftrain = rbind(perftrain, outNN$train)
  perfTest = rbind(perfTest, outNN$test)
  rownameTable = append(rownameTable, "NN")
}


if(modelSVMclass == 1){
  # for 3 kernels
  perfCV = rbind(perfCV, outSVMCV_linear$CV)
  perftrain = rbind(perftrain, outSVM_linear$train)
  perfTest = rbind(perfTest, outSVM_linear$test)
  rownameTable = append(rownameTable, "SVM-linear")
  
  perfCV = rbind(perfCV, outSVMCV_radial$CV)
  perftrain = rbind(perftrain, outSVM_radial$train)
  perfTest = rbind(perfTest, outSVM_radial$test)
  rownameTable = append(rownameTable, "SVM-radial")
  
  perfCV = rbind(perfCV, outSVMCV_sigmoid$CV)
  perftrain = rbind(perftrain, outSVM_sigmoid$train)
  perfTest = rbind(perfTest, outSVM_sigmoid$test)
  rownameTable = append(rownameTable, "SVM-sigmoid")
  
}


if(modelRFclass == 1){
  perfCV = rbind(perfCV, outRFCV$CV)
  perftrain = rbind(perftrain, outRF$train)
  perfTest = rbind(perfTest, outRF$test)
  rownameTable = append(rownameTable, "RF")
}

if(modelLDAclass == 1){
  perfCV = rbind(perfCV, outLDACV$CV)
  perftrain = rbind(perftrain, outLDA$train)
  perfTest = rbind(perfTest, outLDA$test)
  rownameTable = append(rownameTable, "LDA")
}

rownames(perfTest) = rownameTable
rownames(perftrain) = rownameTable
rownames(perfCV) = rownameTable

colnames(perfTest) = c("Acc", "Se", "Sp", "MCC")
colnames(perftrain) = c("Acc", "Se", "Sp", "MCC")
colnames(perfCV) = c("Acc", "Se", "Sp", "MCC")

write.csv(perfTest, file = paste(prout, "perfTest.csv", sep = ""))
write.csv(perftrain, file = paste(prout, "perfTrain.csv", sep = ""))
write.csv(perfCV, file = paste(prout, "perfCV.csv", sep = ""))


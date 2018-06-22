#!/usr/bin/env Rscript
source ("tool.R")
source("MLClassification.R")
source("performance.R")
source("dataManager.R")

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



# to test
#ptrain = "/home/borrela2/imatinib/results/analysis/QSARs/Lig2D/trainSet.csv"
#ptest = "/home/borrela2/imatinib/results/analysis/QSARs/Lig2D/testSet.csv"
#pcluster = "0"
#prout = "/home/borrela2/imatinib/results/analysis/QSARs/Lig2D/"

# cross validation 10
#nbCV = 10


# model classification #
########################
modelPCRclass = 0 
modelPLSclass = 0
modelSVMclass = 0
modelRFclass = 0
modelCartclass = 1
modelNNclass = 0
modelDLclass = 0
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

# test set
dtest = read.csv(ptest, header = TRUE)
rownames(dtest) = dtest[,1]
dtest = dtest[,-1]

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
lgroupCV = samplingDataNgroup(dtrain, nbCV)
controlDatasets(lgroupCV, paste(prout, "ChecksamplingCV", nbCV, sep = ""))


##### CLASSIF  MODELS #########
###############################
print("*****************************")
print("*****  CLASSIFICATION   *****")
print("*****************************")

############
#   CART   #
############

if(modelCartclass == 1){
  prCART = paste(prout, "CARTclass/", sep = "")
  dir.create(prCART)
  outCART = CARTclass(dtrain, dtest, prCART)
  outCARTCV = CARTClassCV(lgroupCV, prCART)
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

rownames(perfTest) = rownameTable
rownames(perftrain) = rownameTable
rownames(perfCV) = rownameTable

colnames(perfTest) = c("Acc", "Se", "Sp", "MCC")
colnames(perftrain) = c("Acc", "Se", "Sp", "MCC")
colnames(perfCV) = c("Acc", "Se", "Sp", "MCC")

write.csv(perfTest, file = paste(prout, "perfTest.csv", sep = ""))
write.csv(perftrain, file = paste(prout, "perfTrain.csv", sep = ""))
write.csv(perfCV, file = paste(prout, "perfCV.csv", sep = ""))


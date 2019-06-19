#!/usr/bin/env Rscript
source("~/development/Rglobal/source/dataManager.R")
source("MLMultiClass.R")
source("performance.R")

################
#     MAIN     #
################


args <- commandArgs(TRUE)
pall = args[1]
splitVal = as.double(args[2])
prout = args[3]
nbCV = as.integer(args[4])


#pall = "/home/borrela2/cancer/BBN/RFmodels/RF_HM_KC_muta/descAll"
#prout = "/home/borrela2/cancer/BBN/RFmodels/RF_HM_KC_muta/"
#splitVal = 0.20
#nbCV = 10

# model classification #
########################
modelRFMulticlass = 1


#######################
# prep train and test #
#######################
dall = read.csv(pall, sep = "\t", header = TRUE)
rownames(dall) = dall[,1]
dall = dall[,-1]

ldata = sampligDataMulticlass(dall, splitVal, "Aff")
dtrain = ldata[[1]]
dtest = ldata[[2]]

vntree = c(10,50,100,200,500)
vmtry = c(1,2,3,4,5,10,15,20,25,30)

parameters = RFGridMultiClassCV(vntree, vmtry, dtrain, prout)
RFMultiClassTrainTest(dtrain, dtest, parameters[[1]], parameters[[2]], prout)


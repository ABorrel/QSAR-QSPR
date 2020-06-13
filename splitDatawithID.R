#!/usr/bin/env Rscript
source ("./../R_toolbox/dataManager.R")


################
#     MAIN     #
################

args <- commandArgs(TRUE)
pdesc = args[1]
pdesclabeltrain = args[2] #to ID
pdesclabeltest = args[3] 
paff = args[4]
prout = args[5]
valcor = args[6]
maxquantile = as.double(args[7])
logaff = as.integer(args[8])
typeAff = args[9]
nbNA = as.integer(args[10])



##############################
# Process descriptors matrix #
##############################

dglobal = openData(pdesc, valcor, prout, NbmaxNA = nbNA)
dglobal = dglobal[[1]]

print("==== Preprocessing ====")
print(paste("Data initial: dim = ", dim(dglobal)[1], dim(dglobal)[2], sep = " "))

##########
# filter #
##########

dglobal = delnohomogeniousdistribution(dglobal, maxquantile)
print(paste("Data after filtering: dim = ", dim(dglobal)[1], dim(dglobal)[2], sep = " "))

#######################
# order with affinity #
#######################
# Opening
daffinity = read.csv(paff, sep = "\t", header = TRUE)
rownames(daffinity) = daffinity[,1]

#select data by type of affinity
if(typeAff != "All"){
  iselect = which(daffinity[,which(colnames(daffinity) == "Type")] == typeAff)
  daffinity = daffinity[iselect,]  
}

if(!is.null(colnames(daffinity)) == "Type"){# remove type col
  daffinity = daffinity[,-which(colnames(daffinity) == "Type")]
}

#Remove NA on affinity
daffinity = na.omit(daffinity)


# transform #
if(logaff == 1){
  namerow = daffinity[,1]
  daffinitylog = -log10(daffinity[,-1])
  daffinity = cbind(namerow, daffinitylog)
  rownames(daffinity) = namerow
}


# merge with data descriptors and remove data remove from the manual curation
#lID = intersect(rownames(daffinity), rownames(dglobal))
#print(paste("NB ID selected, intersect aff and global: ", length(lID), sep = ""))
#dglobal = dglobal[lID,]
#daffinity = daffinity[lID,]



##################
# divide dataset #
##################

# write global set
write.csv(dglobal, paste(prout, "globalSet.csv", sep = ""))

#load ID for train and test
dtrainID = read.csv(pdesclabeltrain, sep=",", header = TRUE)
rownames(dtrainID) = dtrainID[,1]

dtestID = read.csv(pdesclabeltest, sep=",", header = TRUE)
rownames(dtestID) = dtestID[,1]


dtrain = dglobal[rownames(dtrainID),]
dtest = dglobal[rownames(dtestID),]


# rapply null sd and correlation
dtrain = delSDNull(dtrain)
dtest = delSDNull(dtest)
ldesc = intersect(colnames(dtrain), colnames(dtest))

dtrain = dtrain[,ldesc]
dtest = dtest[,ldesc]

#################
# Add affinity  #
#################


# training set
Aff = daffinity[rownames(dtrain),2]
dtrainglobal = cbind(dtrain, Aff)
write.csv(dtrainglobal, paste(prout, "trainSet.csv", sep = ""))
  
# test set
Aff = daffinity[rownames(dtest),2]
dtestglobal = cbind(dtest, Aff)
write.csv(dtestglobal, paste(prout, "testSet.csv", sep = ""))
  
lcontrol = list(dtrainglobal, dtestglobal)
controlDatasets(lcontrol, paste(prout, "qualitySplit", sep = ""))

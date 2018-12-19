#!/usr/bin/env Rscript
source ("~/development/Rglobal/source/dataManager.R")


################
#     MAIN     #
################

args <- commandArgs(TRUE)
pdesc = args[1]
pdata = args[2] #to take affinity or class
prout = args[3]
valcor = args[4]
maxquantile = as.double(args[5])
proptraintest = as.double(args[6])
logaff = as.integer(args[7])
typeAff = args[8]
nbNA = as.integer(args[9])



#pdesc = "/home/borrela2/interference/PUBCHEM/411/descMat"
#pdata = "/home/borrela2/interference/PUBCHEM/411/QSAR/Aff/actClass.txt"
#prout = "/home/borrela2/interference/PUBCHEM/411/QSAR/Aff/"
#valcor = "0.9"
#maxquantile = 90 
#proptraintest =0.15
#logaff =0 
#typeAff = "All" 
#nbNA =1000

#pdesc = "/home/borrela2/interference/Desc/tableDesc1D2DOpera"
#pdata = "/home/borrela2/interference/spDataAnalysis/QSARclassCrossColor/1/crossColor/AC50_all"
#prout = "/home/borrela2/interference/spDataAnalysis/QSARclassCrossColor/1/crossColor/"
#valcor = 0.9
#maxquantile = 90 
#proptraintest = 0.15
#logaff = 1
#typeAff = "All"
#nbNA = 1000


#pdesc = "/home/borrela2/imatinib/results/analysis/QSARs/Lig-FPI-BS/descGlobal"
#pdata = "/home/borrela2/imatinib/results/CHEMBL/AffAllcurated"
#prout = "/home/borrela2/imatinib/results/analysis/QSARs/Lig-FPI-BS/"
#valcor = 0.80
#logaff = 0
#maxquantile = 80
#proptraintest = 0.15
#typeAff = "All"


#pdesc = "/home/borrela2/interference/PUBCHEM/411/descMat"
#pdata = "/home/borrela2/interference/PUBCHEM/411/QSAR/Aff/actClass.txt"
#prout = "/home/borrela2/interference/PUBCHEM/411/QSAR/Aff/"
#valcor = 0.85
#maxquantile = 90 
#proptraintest = 0.15
#logaff = 0
#typeAff = "All"
#nbNA = 10000


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
daffinity = read.csv(pdata, sep = "\t", header = TRUE)
rownames(daffinity) = daffinity[,1]
#select data by type of affinity

if(typeAff != "All"){
  iselect = which(daffinity[,which(colnames(daffinity) == "Type")] == typeAff)
  daffinity = daffinity[iselect,]  
}

if(is.integer0(which(colnames(daffinity) == "Type")) == FALSE){# remove type col
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
lID = intersect(rownames(daffinity), rownames(dglobal))
print(paste("NB ID selected, intersect aff and global: ", length(lID), sep = ""))
dglobal = dglobal[lID,]

daffinity = daffinity[lID,]


##################
# divide dataset #
##################

# write global set
write.csv(dglobal, paste(prout, "globalSet.csv", sep = ""))



ltraintest = samplingDataFraction(dglobal, proptraintest)
dtrain = ltraintest[[1]]
dtest = ltraintest[[2]]

# rapply null sd and correlation
dtrain = delSDNull(dtrain)
dtest = delSDNull(dtest)
ldesc = intersect(colnames(dtrain), colnames(dtest))

dtrain = dtrain[,ldesc]
dtest = dtest[,ldesc]

#################
# Add affinity  #
#################


if(dim(daffinity)[2] == 2){
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
}else{
  for(afftype in colnames(daffinity)[-1]){
    print (afftype)
    # training set
    Aff = daffinity[rownames(dtrain),afftype]
    dtrainglobal = cbind(dtrain, Aff)
    write.csv(dtrainglobal, paste(prout, afftype, "_trainSet.csv", sep = ""))
    
    # test set
    Aff = daffinity[rownames(dtest),afftype]
    dtestglobal = cbind(dtest, Aff)
    write.csv(dtestglobal, paste(prout, afftype, "_testSet.csv", sep = ""))
    
    lcontrol = list(dtrainglobal, dtestglobal)
    controlDatasets(lcontrol, paste(prout, afftype, "_qualitySplit", sep = ""))
    
  }
}



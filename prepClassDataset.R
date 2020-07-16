#!/usr/bin/env Rscript
source("./../R_toolbox/dataManager.R")

# script use to define a class dataset based on a rate of active chemicals


################
#     MAIN     #
################

args <- commandArgs(TRUE)
p_desc = args[1]
p_aff = args[2]
rate_act = as.double(args[3])
pr_out = args[4]

#p_aff = "C:/Users/Aborrel/research/ILS/HERG/results/Cleaned_Data/AC50_cleaned.csv"
#p_desc = "C:/Users/Aborrel/research/ILS/HERG/results/Cleaned_Data/desc1D2D_cleaned.csv"
#rate_act = 0.3
#pr_out = "C:/Users/Aborrel/research/ILS/HERG/results/QSAR/1/"


p_desc = 'C:\\Users\\Aborrel\\research\\ILS\\HERG\\results\\QSAR\\trainGlobal.csv'
p_aff = 'C:\\Users\\Aborrel\\research\\ILS\\HERG\\results\\Cleaned_Data\\AC50_cleaned.csv'
rate_act = 0.3
pr_out = 'C:\\Users\\Aborrel\\research\\ILS\\HERG\\results\\QSAR\\1\\/'




# open files
############

ddesc = read.csv(p_desc, sep = ",", header = TRUE)
rownames(ddesc) = ddesc[,1]
ddesc = ddesc[,-1]

# be sure Aff is not in ddesc matrix
if(exists("Aff", where=ddesc)){
  ddesc = ddesc[,-which(colnames(ddesc)=="Aff")]
}



daff = read.csv(p_aff, sep = ",", header = TRUE)
rownames(daff) = daff[,1]


l_inter = intersect(rownames(ddesc), rownames(daff))
daff = daff[l_inter,]
ddesc = ddesc[l_inter,]


# format affinity in class
#################
Aff = rep(0, dim(daff)[1])
Aff[which(!is.na(daff$Aff))] = 1

# add aff 
#########
ID = rownames(ddesc)
ddesc = cbind(ID, ddesc)
ddesc = cbind(ddesc, Aff)

ldclasses = separeData (ddesc, "Aff")
d0 = ldclasses[[1]]
d1 = ldclasses[[2]]

nb_active = dim(d1)[1]

# use to limit the dataset
if(rate_act != 0){
  nb_inactive = 100*nb_active/(rate_act*100)
  print(nb_inactive)
}else{
  nb_inactive = dim(d0)[1]# we take all acitve
}


dout = rbind(d1, d0[sample(dim(d0)[1])[0:nb_inactive],])
write.table(dout, file=paste(pr_out, "desc_Class.csv", sep=""),  sep = "\t")

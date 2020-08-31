#!/usr/bin/env Rscript
source("performance.R")
library("pls")
source("./../R_toolbox/dataManager.R")
library(randomForest)
library(MASS)
library(rpart)
library(rpart.plot)
library(e1071)
library(ggplot2)
#library(neuralnet)
library(caret)
library(nnet)
library(clusterGeneration)
library(stringr)
library(reshape2)


######################################################################################################################################
######################################################################################################################################


#####################
#  classification   #
#####################


##########
#  SVM   #
##########

SVMClassTrainTest = function(dtrain, dtest, vgamma, vcost, ksvm, prout){
  
  pmodel = paste(prout, "model.RData", sep = "")
  if(file.exists(pmodel)){
    load(pmodel)
    return(outmodel) 
  }
  
  
  print(paste("====SVM-", ksvm, " in train-test --- Automatic optimization CV-10====", sep = ""))
  
  # optimisation on CV-10
  modelsvm = SVMTuneClass(dtrain, vgamma, vcost, ksvm, 10)
  
  predsvmtest = predict(modelsvm, dtest[,-c(which(colnames(dtest) == "Aff"))])
  predsvmtrain = predict(modelsvm, dtrain[,-c(which(colnames(dtrain) == "Aff"))])
  
  names(predsvmtrain) = rownames(dtrain)
  names(predsvmtest) = rownames(dtest)
  
  # performances = train
  dpredtrain = cbind(dtrain[,"Aff"], as.character(predsvmtrain))
  colnames(dpredtrain) = c("Real", "Predict")
  write.csv(dpredtrain, paste(prout, "TrainPred.csv", sep = ""))
  
  lpreftrain = classPerf(dpredtrain[,1], dpredtrain[,2])
  acctrain = lpreftrain[[1]]
  setrain = lpreftrain[[2]]
  sptrain = lpreftrain[[3]]
  mcctrain = lpreftrain[[4]]
  
  # performances = test
  dpredtest = cbind(dtest[,"Aff"], as.character(predsvmtest))
  colnames(dpredtest) = c("Real", "Predict")
  write.csv(dpredtest, paste(prout, "TestPred.csv", sep = ""))
  
  lpreftest = classPerf(dtest[,"Aff"], predsvmtest)
  acctest = lpreftest[[1]]
  setest = lpreftest[[2]]
  sptest = lpreftest[[3]]
  mcctest = lpreftest[[4]]
  
  print("===== SVM model train-Test =====")
  #print(modelpls$coefficients)
  print(paste("Perf training (dim= ", dim(dtrain)[1], "*", dim(dtrain)[2], "):", sep = ""))
  print(paste("ACC train=", acctrain))
  print(paste("Se train=", setrain))
  print(paste("Sp train=", sptrain))
  print(paste("MCC train=", mcctrain))
  print("")
  print("")
  
  
  print(paste("Perf test (dim=", dim(dtest)[1], "*", dim(dtest)[2], "):", sep = ""))
  print(paste("ACC test=", acctest))
  print(paste("Se test=", setest))
  print(paste("Sp test=", sptest))
  print(paste("MCC test=", mcctest))
  print("")
  print("")
  
  
  perftrain = c(acctrain, setrain, sptrain, mcctrain)
  names(perftrain) = c("ACC", "SE", "SP", "MCC")
  
  perftest = c(acctest, setest, sptest, mcctest)
  names(perftest) = c("ACC", "SE", "SP", "MCC")
  
  outmodel = list()
  outmodel$train = perftrain
  outmodel$test = perftest
  outmodel$model = modelsvm
  
  save(outmodel, file = paste(prout, "model.RData", sep = ""))
  return(outmodel)
} 
  
  
SVMClassCV = function(lgroupCV, vgamma, vcost, ksvm, prout){  
  
  pmodel = paste(prout, "modelCV.RData", sep = "")
  if(file.exists(pmodel) == TRUE){
    load(pmodel)
    return(outmodelCV)
  }
  
  
  
  print(paste("== SVM-", ksvm , " in CV with ", length(lgroupCV), " Automatic optimization by folds", sep = ""))
  
  # data combination
  k = 1
  kmax = length(lgroupCV)
  y_predict = NULL
  y_real = NULL
  y_proba = NULL
  while(k <= kmax){
    dtrain = NULL
    dtest = NULL
    for (m in seq(1:kmax)){
      if (m == k){
        dtest = lgroupCV[[m]]
      }else{
        dtrain = rbind(dtrain, lgroupCV[[m]])
      }
    }
    
    dtrain$Aff = as.factor(dtrain$Aff)
    
    
    modtune = tryCatch(tune(svm, Aff~., data = dtrain, ranges = list(gamma = vgamma, cost = vcost, type="nu-classification", kernel=ksvm), tunecontrol = tune.control(sampling = "fix")),
                       error = function(e) {return("NA")})
    
    
    if(is.character(modtune)){
      modtune = tune(svm, Aff~., data = dtrain, ranges = list(gamma = vgamma, cost = vcost, type="C-classification", kernel=ksvm), tunecontrol = tune.control(sampling = "fix"))
    }
    
    vpred = predict (modtune$best.model, dtest, decision.values = TRUE)
    vpred = as.double(vpred)
    vpred[which(vpred == 1)] = 0
    vpred[which(vpred == 2)] = 1
    y_proba = append(y_proba, vpred)
    
    
    y_predict = append(y_predict, vpred)
    y_real = append(y_real, dtest[,"Aff"])
    k = k + 1
  }
  
  # performances
  lpref = classPerf(y_real, y_predict)
  acc = lpref[[1]]
  se = lpref[[2]]
  sp = lpref[[3]]
  mcc = lpref[[4]]
  
  dpred = cbind(y_proba, y_real)
  colnames(dpred) = c("Predict", "Real")
  write.csv(dpred, file = paste(prout, "CVPred-", length(lgroupCV), ".csv", sep = ""))
  
  print("Perfomances in CV")
  print(paste("acc=", acc, sep = ""))
  print(paste("se=", se, sep = ""))
  print(paste("sp=", sp, sep = ""))
  print(paste("mcc=", mcc, sep = "")) 
  print("")
  print("")
  
  outmodelCV = list()
  lscore = c(acc, se, sp, mcc)
  names(lscore) = c("ACC", "SE", "SP", "MCC")
  outmodelCV$CV = lscore
  save(outmodelCV, file = paste(prout, "modelCV.RData", sep = ""))
  return(outmodelCV)  
}


SVMTuneClass = function(dtrain, vgamma, vcost, ksvm, nbCV){
  
  
  lfolds = samplingDataNgroup(dtrain, nbCV)
  lmodel = list()
  lMCCbest = NULL
  
  k = 1
  kmax = length(lfolds)
  while(k <= kmax){
    dtrain = NULL
    dtest = NULL
    for (m in seq(1:kmax)){
      lcpd = rownames(lfolds[[m]])
      if (m == k){
        dtest = as.data.frame(lfolds[[m]])
      }else{
        dtrain = rbind(dtrain, lfolds[[m]])
      }
    }
    
    #dtrain = as.data.frame(scale(dtrain))
    ddestrain = dtrain[,-c(which(colnames(dtrain) == "Aff"))]
    #ddestrain = scale(ddestrain)
    Aff = dtrain[,c("Aff")]
    dtrain$Aff = as.factor(dtrain$Aff)
    
    
    #dtest = as.data.frame(scale(dtest))
    dtestAff = dtest[,"Aff"]
    #ddesctest = dtest[,-c(which(colnames(dtest) == "Aff"))]
    #ddesctest = scale(ddesctest, center = attr(ddestrain, 'scaled:center'), scale = attr(ddestrain, 'scaled:scale'))
    
    modelsvm = tryCatch(tune(svm, Aff~., data = dtrain, ranges = list(gamma = vgamma, cost = vcost, type="nu-classification", kernel=ksvm), tunecontrol = tune.control(sampling = "fix")),
                       error = function(e) {return("NA")})
    
    
    if(is.character(modelsvm)){
      modelsvm = tune(svm, Aff~., data = dtrain, ranges = list(gamma = vgamma, cost = vcost, type="C-classification", kernel=ksvm), tunecontrol = tune.control(sampling = "fix"))
    }
    
    modelsvm = modelsvm$best.model
    vpred = predict(modelsvm, dtest, decision.values = TRUE)
    
    vpred = as.double(vpred)
    vpred[which(vpred == 1 )] = 0
    vpred[which(vpred == 2 )] = 1
    
    
    lpreftest = classPerf(dtestAff, vpred)
    acctest = lpreftest[[1]]
    setest = lpreftest[[2]]
    sptest = lpreftest[[3]]
    mcctest = lpreftest[[4]]
    
    # optimize with MCC
    lMCCbest = append(lMCCbest, mcctest)
    lmodel[[k]] =  modelsvm
    k = k + 1 
  }
  #print(lMCCbest)
  #print(which(lMCCbest == max(lMCCbest, na.rm = TRUE)))
  return(lmodel[[which(lMCCbest == max(lMCCbest, na.rm = TRUE))]])
}  



#####################
#  Neural network   #
#####################

NNClassTrainTest = function(dtrain, dtest, vsize, vdecay, prout){
  
  pmodel = paste(prout, "model.RData", sep = "")
  if(file.exists(pmodel)){
    load(pmodel)
    return(outmodel) 
  }
  
  
  print(paste("====NN in train-test --- Automatic optimization CV-10====", sep = ""))
  
  # optimisation on CV-10
  modelNN = NNTuneClass(dtrain, vsize, vdecay, 10)
  
  predsvmtest = predict(modelNN, dtest[,-c(which(colnames(dtest) == "Aff"))])
  predsvmtrain = predict(modelNN, dtrain[,-c(which(colnames(dtrain) == "Aff"))])
  
  #fix proba => error
  
  predsvmtrain[which(predsvmtrain < 0.5)] = 0
  predsvmtrain[which(predsvmtrain >= 0.5)] = 1
  predsvmtest[which(predsvmtest < 0.5)] = 0
  predsvmtest[which(predsvmtest >= 0.5)] = 1
  
  names(predsvmtrain) = rownames(dtrain)
  names(predsvmtest) = rownames(dtest)
  
  # performances = train
  dpredtrain = cbind(dtrain[,"Aff"], as.character(predsvmtrain))
  colnames(dpredtrain) = c("Real", "Predict")
  write.csv(dpredtrain, paste(prout, "TrainPred.csv", sep = ""))
  
  lpreftrain = classPerf(dpredtrain[,1], dpredtrain[,2])
  acctrain = lpreftrain[[1]]
  setrain = lpreftrain[[2]]
  sptrain = lpreftrain[[3]]
  mcctrain = lpreftrain[[4]]
  
  # performances = test
  dpredtest = cbind(dtest[,"Aff"], as.character(predsvmtest))
  colnames(dpredtest) = c("Real", "Predict")
  write.csv(dpredtest, paste(prout, "TestPred.csv", sep = ""))
  
  lpreftest = classPerf(dtest[,"Aff"], predsvmtest)
  acctest = lpreftest[[1]]
  setest = lpreftest[[2]]
  sptest = lpreftest[[3]]
  mcctest = lpreftest[[4]]
  
  print("===== NN model train-Test =====")
  #print(modelpls$coefficients)
  print(paste("Perf training (dim= ", dim(dtrain)[1], "*", dim(dtrain)[2], "):", sep = ""))
  print(paste("ACC train=", acctrain))
  print(paste("Se train=", setrain))
  print(paste("Sp train=", sptrain))
  print(paste("MCC train=", mcctrain))
  print("")
  print("")
  
  
  print(paste("Perf test (dim=", dim(dtest)[1], "*", dim(dtest)[2], "):", sep = ""))
  print(paste("ACC test=", acctest))
  print(paste("Se test=", setest))
  print(paste("Sp test=", sptest))
  print(paste("MCC test=", mcctest))
  print("")
  print("")
  
  
  perftrain = c(acctrain, setrain, sptrain, mcctrain)
  names(perftrain) = c("ACC", "SE", "SP", "MCC")
  
  perftest = c(acctest, setest, sptest, mcctest)
  names(perftest) = c("ACC", "SE", "SP", "MCC")
  
  outmodel = list()
  outmodel$train = perftrain
  outmodel$test = perftest
  outmodel$model = modelNN
  
  save(outmodel, file = paste(prout, "model.RData", sep = ""))
  return(outmodel)
} 


NNClassCV = function(lgroupCV, vsize, vdecay, prout){  
  
  pmodel = paste(prout, "modelCV.RData", sep = "")
  if(file.exists(pmodel) == TRUE){
    load(pmodel)
    return(outmodelCV)
  }
  
  
  print(paste("== NN in CV with ", length(lgroupCV), " Automatic optimization by folds ==", sep = ""))
  
  # data combination
  k = 1
  my.grid <- expand.grid(.decay = vdecay, .size = vsize)
  kmax = length(lgroupCV)
  y_predict = NULL
  y_real = NULL
  y_proba = NULL
  while(k <= kmax){
    dtrain = NULL
    dtest = NULL
    for (m in seq(1:kmax)){
      if (m == k){
        dtest = lgroupCV[[m]]
      }else{
        dtrain = rbind(dtrain, lgroupCV[[m]])
      }
    }
    
    #dtrain$Aff = as.factor(dtrain$Aff)
    nnfit = train(Aff ~ ., data = dtrain,
                  method = "nnet", maxit = 75, tuneGrid = my.grid, trace = F, linout = 1) 
    
    vpred = predict (nnfit, dtest)
    #vpred = as.double(vpred)
    
    y_proba = append(y_proba, vpred)
    vpred[which(vpred >= 0.5)] = 1
    vpred[which(vpred <= 0.5)] = 0
    
    y_predict = append(y_predict, vpred)
    y_real = append(y_real, dtest[,"Aff"])
    k = k + 1
  }
  
  # performances
  lpref = classPerf(y_real, y_predict)
  acc = lpref[[1]]
  se = lpref[[2]]
  sp = lpref[[3]]
  mcc = lpref[[4]]
  
  dpred = cbind(y_proba, y_real)
  colnames(dpred) = c("Predict", "Real")
  write.csv(dpred, file = paste(prout, "CVPred-", length(lgroupCV), ".csv", sep = ""))
  
  print("Perfomances in CV")
  print(paste("acc=", acc, sep = ""))
  print(paste("se=", se, sep = ""))
  print(paste("sp=", sp, sep = ""))
  print(paste("mcc=", mcc, sep = "")) 
  print("")
  print("")
  
  outmodelCV = list()
  lscore = c(acc, se, sp, mcc)
  names(lscore) = c("ACC", "SE", "SP", "MCC")
  outmodelCV$CV = lscore
  save(outmodelCV, file = paste(prout, "modelCV.RData", sep = ""))
  return(outmodelCV)  
}


NNTuneClass = function(dtrain, vsize, vdecay, nbCV){
  
  
  lfolds = samplingDataNgroup(dtrain, nbCV)
  lmodel = list()
  lMCCbest = NULL
  
  my.grid <- expand.grid(.decay = vdecay, .size = vsize)
  k = 1
  kmax = length(lfolds)
  while(k <= kmax){
    dtrain = NULL
    dtest = NULL
    for (m in seq(1:kmax)){
      lcpd = rownames(lfolds[[m]])
      if (m == k){
        dtest = as.data.frame(lfolds[[m]])
      }else{
        dtrain = rbind(dtrain, lfolds[[m]])
      }
    }
    
    nnfit = train(Aff ~ ., data = dtrain,
                  method = "nnet", maxit = 75, tuneGrid = my.grid, trace = F, linout = 1) 
    
    vpred = predict (nnfit, dtest)
    #vpred = as.double(vpred)
    
    vpred[which(vpred >= 0.5)] = 1
    vpred[which(vpred <= 0.5)] = 0
    
    lpreftest = classPerf(dtest$Aff, vpred)
    acctest = lpreftest[[1]]
    setest = lpreftest[[2]]
    sptest = lpreftest[[3]]
    mcctest = lpreftest[[4]]
    
    # optimize with MCC
    lMCCbest = append(lMCCbest, mcctest)
    lmodel[[k]] =  nnfit
    k = k + 1 
  }
  print(lMCCbest)
  return(lmodel[[which(lMCCbest == max(lMCCbest, na.rm = TRUE))]])
}  




#####################
#  Random forest    #
#####################

RFGridClassCV = function(lntree, lmtry, lfolds, prout){
  
  pgrid = paste(prout, "grid.RData", sep = "")
  if(file.exists(pgrid) == TRUE){
    load(pgrid)
    return(list(rownames (gridOpt)[which(gridOpt==max(gridOpt), arr.ind=T)[1]],colnames (gridOpt)[which(gridOpt==max(gridOpt), arr.ind=T)[2]] ))
  }
  
  
  gridOpt = data.frame ()
  i = 0
  for (ntree in lntree){
    i = i + 1
    j = 0
    for (mtry in lmtry){
      j = j + 1
      
      # data combination
      k = 1
      kmax = length(lfolds)
      y_predict = NULL
      y_real = NULL
      while(k <= kmax){
        dtrain = NULL
        dtest = NULL
        for (m in seq(1:kmax)){
          if (m == k){
            dtest = lfolds[[m]]
          }else{
            dtrain = rbind(dtrain, lfolds[[m]])
          }
        }
        
        modelRF = randomForest( Aff~., data = dtrain, mtry=mtry, ntree = ntree, type = "class",  importance=TRUE)
        vpred = predict (modelRF, dtest)
        vpred[which(vpred < 0.5)] = 0
        vpred[which(vpred >= 0.5)] = 1
        
        y_predict = append(y_predict, vpred)
        y_real = append(y_real, dtest[,"Aff"])
        k = k + 1
      }
      
      
      # MCC for grid
      #print(y_predict)
      l_perf = classPerf(y_real, y_predict)
      gridOpt[i,j] = l_perf[[4]]
      
      # R conversion 
    }
  }
  colnames (gridOpt) = lmtry
  rownames (gridOpt) = lntree
  
  write.table (gridOpt, paste(prout, "RFclassMCC.grid", sep = ""))
  print(which(gridOpt == max(gridOpt), arr.ind = TRUE))
  save(gridOpt, file = paste(prout, "grid.RData", sep = ""))
  
  print(paste("=== RF grid optimisation in CV = ", length(lfolds), " ntree = ", rownames (gridOpt)[which(gridOpt==max(gridOpt), arr.ind=T)[1]], " mtry=", colnames (gridOpt)[which(gridOpt==max(gridOpt), arr.ind=T)[2]], sep = ""))
  return (list(rownames (gridOpt)[which(gridOpt==max(gridOpt), arr.ind=T)[1]],colnames (gridOpt)[which(gridOpt==max(gridOpt), arr.ind=T)[2]] ))
}

RFClassCV = function(lfolds, ntree, mtry, prout){
  
  pmodel = paste(prout, "modelCV.RData", sep = "")
  if(file.exists(pmodel) == TRUE){
    load(pmodel)
    return(outmodelCV)
  }
  
  print(paste("== RF in CV with ", length(lfolds), " folds ntree = ", ntree, " mtry = ", mtry, sep = ""))
  
  # data combination
  k = 1
  kmax = length(lfolds)
  y_predict = NULL
  y_real = NULL
  timportance = NULL
  y_proba = NULL
  while(k <= kmax){
    dtrain = NULL
    dtest = NULL
    for (m in seq(1:kmax)){
      if (m == k){
        dtest = lfolds[[m]]
      }else{
        dtrain = rbind(dtrain, lfolds[[m]])
      }
    }
    
    modelRF = randomForest( Aff~., data = dtrain, mtry=as.integer(mtry), ntree=as.integer(ntree), type = "class",  importance=TRUE)
    vpred = predict (modelRF, dtest)
    vproba = vpred
    vpred[which(vpred < 0.5)] = 0
    vpred[which(vpred >= 0.5)] = 1
    
    timportance = cbind(timportance, modelRF$importance[,1])
    y_predict = append(y_predict, vpred)
    y_proba = append(y_proba, vproba)
    y_real = append(y_real, dtest[,"Aff"])
    k = k + 1
  }
  
  # performances
  lpref = classPerf(y_real, y_predict)
  acc = lpref[[1]]
  se = lpref[[2]]
  sp = lpref[[3]]
  mcc = lpref[[4]]
  
  png(paste(prout, "PerfRFClassCV", length(lfolds), ".png", sep = ""), 800, 800)
  plot(y_real, y_proba, type = "n")
  text(y_real, y_proba, labels = names(y_predict), cex = 0.8)
  abline(a = 0.5, b = 0, col = "red", cex = 3)
  dev.off()
  
  dpred = cbind(y_proba, y_real)
  colnames(dpred) = c("Predict", "Real")
  write.table(dpred, file = paste(prout, "PerfRFClassCV", length(lfolds), ".txt", sep = ""), sep = "\t")
  
  print("Perfomances in CV")
  print(paste("acc=", acc, sep = ""))
  print(paste("se=", se, sep = ""))
  print(paste("sp=", sp, sep = ""))
  print(paste("mcc=", mcc, sep = ""))
  print("")
  print("")
  
  outmodelCV = list()
  lscore = c(acc, se, sp, mcc)
  names(lscore) = c("ACC", "SE", "SP", "MCC")
  outmodelCV$CV = lscore
  
  
  # importance descriptors
  Mimportance = apply(timportance, 1, mean)
  SDimportance = apply(timportance, 1, sd)
  
  dimportance = cbind(Mimportance, SDimportance)
  rownames(dimportance) = rownames(timportance)
  dimportance = dimportance[order(dimportance[,1], decreasing = TRUE),]
  
  write.table(dimportance, paste(prout, "ImportanceDescCV", length(lfolds), sep = ""), sep = "\t")
  
  outmodelCV$importance = dimportance
  
  png(paste(prout, "ImportanceRFClassCV", length(lfolds), ".png", sep = ""), 1000, 800)
  par( mar=c(10,4,4,4))
  plot(dimportance[,1], xaxt ="n", xlab="", pch = 19, ylab="M importance")
  axis(1, 1:length(dimportance[,1]), labels = rownames(dimportance), las = 2, cex.axis = 0.7, cex = 2.75)
  for (i in 1:(dim(dimportance)[1])){
    segments(i, dimportance[i,1] - dimportance[i,2], i, dimportance[i,1] + dimportance[i,2])
  }
  dev.off()
  
  save(outmodelCV, file = paste(prout, "modelCV.RData", sep = ""))
  return(outmodelCV)
}


RFClassTrainTest = function (dtrain, dtest, ntree, mtry, prout){
  
  pmodel = paste(prout, "model.RData", sep = "")
  if(file.exists(pmodel)){
    load(pmodel)
    return(outmodel) 
  }
  
  
  modelRF = randomForest( Aff~., data = dtrain, mtry=as.integer(mtry), ntree=as.integer(ntree), type = "class",  importance=TRUE)
  vpredtrain = predict (modelRF, dtrain, type = "class")
  vpredtest = predict (modelRF, dtest, type = "class")
  
  vpredtrainprob = vpredtrain
  vpredtestprob = vpredtest
  
  vpredtest[which(vpredtest < 0.5)] = 0
  vpredtest[which(vpredtest >= 0.5)] = 1
  vpredtrain[which(vpredtrain < 0.5)] = 0
  vpredtrain[which(vpredtrain >= 0.5)] = 1
  
  
  vperftrain = classPerf(dtrain[,c("Aff")], vpredtrain)
  acctrain = vperftrain[1]
  setrain = vperftrain[2]
  sptrain = vperftrain[3]
  mcctrain = vperftrain[4]
  
  vperftest = classPerf(dtest[,c("Aff")], vpredtest)
  acctest = vperftest[1]
  setest = vperftest[2]
  sptest = vperftest[3]
  mcctest = vperftest[4]
  
  
  print("===Perf RF===")
  print(paste("Dim train: ", dim(dtrain)[1]," ", dim(dtrain)[2], sep = ""))
  print(paste("Dim test: ", dim(dtest)[1]," ", dim(dtest)[2], sep = ""))
  
  print("==Train==")
  print(paste("acc=", vperftrain[[1]], sep = ""))
  print(paste("se=", vperftrain[[2]], sep = ""))
  print(paste("sp=", vperftrain[[3]], sep = ""))
  print(paste("mcc=", vperftrain[[4]], sep = ""))
  
  
  print("==Test==")
  print(paste("acc=", vperftest[[1]], sep = ""))
  print(paste("se=", vperftest[[2]], sep = ""))
  print(paste("sp=", vperftest[[3]], sep = ""))
  print(paste("mcc=", vperftest[[4]], sep = ""))
  print("")
  print("")
  
  
  perftrain = c(acctrain, setrain, sptrain, mcctrain)
  names(perftrain) = c("ACC", "SE", "SP", "MCC")
  
  perftest = c(acctest, setest, sptest, mcctest)
  names(perftest) = c("ACC", "SE", "SP", "MCC")
  
  outmodel = list()
  outmodel$train = perftrain
  outmodel$test = perftest
  outmodel$model = modelRF
  
  # importance
  timportance = modelRF$importance[,1]
  write.table(timportance, paste(prout, "ImportanceDesc", sep = ""), sep = "\t")
  outmodel$importance = timportance
  
  save(outmodel, file = paste(prout, "model.RData", sep = ""))
  
  png(paste(prout, "PerfTrainTest.png", sep = ""), 1600, 800)
  par(mfrow = c(1,2))
  plot(dtrain[,"Aff"], vpredtrainprob, type = "n")
  text(dtrain[,"Aff"], vpredtrainprob, labels = names(vpredtrainprob))
  abline(a = 0.5, b = 0, col = "red", cex = 3)
  dev.off()
  
  plot(dtest[,"Aff"], vpredtest, type = "n")
  text(dtest[,"Aff"], vpredtestprob, labels = names(vpredtestprob))
  abline(a = 0.5, b = 0, col = "red", cex = 3)
  dev.off()
  
  
  write.csv(vpredtestprob, paste(prout,"classTest.csv", sep = ""))
  write.csv(vpredtrainprob, paste(prout,"classTrain.csv", sep = ""))
  
  return(outmodel)
}



##########
#  CART  #
##########

CARTClassCV = function(lfolds, prout){
  
  pmodel = paste(prout, "modelCV.RData", sep = "")
  if(file.exists(pmodel) == TRUE){
    load(pmodel)
    return(outmodelCV)
  }
  
  print(paste("== CART in CV with ", length(lfolds), "==", sep = ""))
  
  # data combination
  k = 1
  kmax = length(lfolds)
  y_predict = NULL
  y_real = NULL
  y_proba = NULL
  pdf(paste(prout, "TreeCARTClass-CV", length(lfolds), ".pdf",sep = ""))
  while(k <= kmax){
    dtrain = NULL
    dtest = NULL
    for (m in seq(1:kmax)){
      if (m == k){
        dtest = lfolds[[m]]
      }else{
        dtrain = rbind(dtrain, lfolds[[m]])
      }
    }
    
    dtrain$Aff = as.factor(as.character(dtrain$Aff))
    modelCART = best.rpart( Aff~., data = dtrain, minsplit = c(1,5,1))
    vpred = predict(modelCART, dtest)
    vpred = vpred[,2]
    vproba = vpred
    vpred[which(vpred < 0.5)] = 0
    vpred[which(vpred >= 0.5)] = 1
    
    plotcp(modelCART)
    # plot tree in pdf
    rpart.plot( modelCART , # middle graph
                extra=104, box.palette="GnBu",
                branch.lty=3, shadow.col="gray", nn=TRUE)
    
    y_predict = append(y_predict, vpred)
    y_proba = append(y_proba, vproba)
    y_real = append(y_real, dtest[,"Aff"])
    
    k = k + 1
  }
  dev.off()
  
  # performances
  lpref = classPerf(y_real, y_predict)
  acc = lpref[[1]]
  se = lpref[[2]]
  sp = lpref[[3]]
  mcc = lpref[[4]]
  
  outmodelCV = list()
  lscore = c(acc, se, sp, mcc)
  names(lscore) = c("Acc", "Se", "Sp", "MCC")
  outmodelCV$CV = lscore
  
  png(paste(prout, "PerfCARTClassCV", length(lfolds), ".png", sep = ""), 800, 800)
  plot(y_real, y_proba, type = "n")
  text(y_real, y_proba, labels = names(y_predict), cex = 0.8)
  abline(a = 0.5, b = 0, col = "red", cex = 3)
  dev.off()
  
  dpred = cbind(y_proba, y_real)
  colnames(dpred) = c("Predict", "Real")
  write.table(dpred, file = paste(prout, "PerfCARTClassCV", length(lfolds), ".txt", sep = ""), sep = "\t")
  
  print("Perfomances in CV")
  print(paste("acc=", acc, sep = ""))
  print(paste("se=", se, sep = ""))
  print(paste("sp=", sp, sep = ""))
  print(paste("mcc=", mcc, sep = ""))
  print("")
  print("")
  
  save(outmodelCV, file = paste(prout, "modelCV.RData", sep = ""))
  return(outmodelCV)
}



CARTclass = function (dtrain, dtest, prout){
  
  pmodel = paste(prout, "model.RData", sep = "")
  if(file.exists(pmodel)){
    load(pmodel)
    return(outmodel) 
  }
  
  
  print("== CART in train/test ==")
  
  # model and apply
  dtrain$Aff = as.factor(as.character(dtrain$Aff))
  
  modelCART = best.rpart( Aff~., data = dtrain, minsplit = c(1,5,1))
  
  
  vpredtrain = predict(modelCART, dtrain)
  vpredtest = predict(modelCART, dtest)
  
  # draw tree
  pdf(paste(prout, "TreeCARTClass-TrainTest.pdf",sep = ""))
  plotcp(modelCART)
  # plot tree in pdf
  rpart.plot( modelCART , # middle graph
              extra=104, box.palette="GnBu",
              branch.lty=3, shadow.col="gray", nn=TRUE)
  dev.off()
  
  
  #result
  vpredtrain = vpredtrain[,2]
  vprobatrain = vpredtrain
  
  vpredtest = vpredtest[,2]
  vprobatest = vpredtest
  
  vrealtrain = dtrain[,c("Aff")]
  vrealtest = dtest[,c("Aff")]
  
  # ROC curve
  drawROCCurve(vrealtrain, vprobatrain, paste(prout, "ROCcurvetrain", sep = ""))
  drawROCCurve(vrealtest, vprobatest, paste(prout, "ROCcurvetest", sep = ""))
  
  
  # format pred
  vprobatrain[which(vprobatrain < 0.5)] = 0
  vprobatrain[which(vprobatrain >= 0.5)] = 1
  vprobatest[which(vprobatest < 0.5)] = 0
  vprobatest[which(vprobatest >= 0.5)] = 1
  
  # performances
  lpreftrain = classPerf(vrealtrain, vprobatrain)
  lpreftest = classPerf(vrealtest, vprobatest)
  
  print("==Perfomances in train/test==")
  print("===Perfomances in train===")
  print(paste("acc=", lpreftrain[[1]], sep = ""))
  print(paste("se=", lpreftrain[[2]], sep = ""))
  print(paste("sp=", lpreftrain[[3]], sep = ""))
  print(paste("mcc=", lpreftrain[[4]], sep = ""))
  print("")
  print("===Perfomances in test===")
  print(paste("acc=", lpreftest[[1]], sep = ""))
  print(paste("se=", lpreftest[[2]], sep = ""))
  print(paste("sp=", lpreftest[[3]], sep = ""))
  print(paste("mcc=", lpreftest[[4]], sep = ""))
  print("")
  print("")
  
  outmodel = list()
  ltrain = c(lpreftrain[[1]],  lpreftrain[[2]],  lpreftrain[[3]],  lpreftrain[[4]])
  names(ltrain) = c("Acc", "Se", "Sp", "MCC")
  outmodel$train = ltrain
  
  ltest = c(lpreftest[[1]], lpreftest[[2]], lpreftest[[3]], lpreftest[[4]])
  names(ltest) = c("Acc", "Se", "Sp", "MCC")
  outmodel$test = ltest
  outmodel$model = modelCART
  
  save(outmodel, file = paste(prout, "model.RData", sep = ""))
  
  return(outmodel)
}



#########
#  LDA  #
#########

LDAClassCV = function(lfolds, prout){
  
 
  pmodel = paste(prout, "modelCV.RData", sep = "")
  if(file.exists(pmodel) == TRUE){
    load(pmodel)
    return(outmodelCV)
  }
  
  print(paste("==== LDA in CV with ", length(lfolds), "=====", sep = ""))
  
  # data combination
  k = 1
  kmax = length(lfolds)
  y_predict = NULL
  y_real = NULL
  timportance = NULL
  y_proba = NULL
  while(k <= kmax){
    dtrain = NULL
    dtest = NULL
    for (m in seq(1:kmax)){
      if (m == k){
        dtest = lfolds[[m]]
      }else{
        dtrain = rbind(dtrain, lfolds[[m]])
      }
    }
    
    #dtrain$Aff = as.factor(dtrain$Aff)
    dtrain = delSDNull(dtrain) # check variance null to remove colinear variable
    modelLDA = lda( Aff~., data = dtrain, type = "class")
    vpred = predict (modelLDA, dtest)
    vproba = vpred$posterior[,2]
    vpred = vproba
    vpred[which(vpred < 0.5)] = 0
    vpred[which(vpred >= 0.5)] = 1
    
    y_predict = append(y_predict, vpred)
    y_proba = append(y_proba, vproba)
    y_real = append(y_real, dtest[,"Aff"])
    k = k + 1
  }
  
  # performances
  lpref = classPerf(y_real, y_predict)
  acc = lpref[[1]]
  se = lpref[[2]]
  sp = lpref[[3]]
  mcc = lpref[[4]]
  
  png(paste(prout, "PerfLDAClassCV", length(lfolds), ".png", sep = ""), 800, 800)
  plot(y_real, y_proba, type = "n")
  text(y_real, y_proba, labels = names(y_predict), cex = 0.8)
  abline(a = 0.5, b = 0, col = "red", cex = 3)
  dev.off()
  
  dpred = cbind(y_proba, y_real)
  colnames(dpred) = c("Predict", "Real")
  write.table(dpred, file = paste(prout, "PerfLDAClassCV", length(lfolds), ".txt", sep = ""), sep = "\t")
  
  print("Perfomances in CV")
  print(paste("acc=", acc, sep = ""))
  print(paste("se=", se, sep = ""))
  print(paste("sp=", sp, sep = ""))
  print(paste("mcc=", mcc, sep = ""))
  print("")
  print("")
  
  outmodelCV = list()
  lscore = c(acc, se, sp, mcc)
  names(lscore) = c("ACC", "SE", "SP", "MCC")
  outmodelCV$CV = lscore
  
  save(outmodelCV, file = paste(prout, "modelCV.RData", sep = ""))
  return(outmodelCV)
}


LDAClassTrainTest = function (dtrain, dtest, prout){#, name_barplot, draw_plot, name_ACP, graph){
  
  pmodel = paste(prout, "model.RData", sep = "")
  if(file.exists(pmodel)){
    load(pmodel)
    return(outmodel) 
  }
  
  
  modelLDA = lda(Aff~., dtrain)
  
  vpredtrain = predict (modelLDA, dtrain, type = "class")
  vpredtest = predict (modelLDA, dtest, type = "class")
  
  vpredtestprob = vpredtest$posterior[,2]
  vpredtrainprob = vpredtrain$posterior[,2]
  
  vpredtrain = vpredtrainprob
  vpredtest = vpredtestprob 

  vpredtest[which(vpredtest < 0.5)] = 0
  vpredtest[which(vpredtest >= 0.5)] = 1
  vpredtrain[which(vpredtrain < 0.5)] = 0
  vpredtrain[which(vpredtrain >= 0.5)] = 1


  vperftrain = classPerf(dtrain[,c("Aff")], vpredtrain)
  acctrain = vperftrain[1]
  setrain = vperftrain[2]
  sptrain = vperftrain[3]
  mcctrain = vperftrain[4]

  vperftest = classPerf(dtest[,c("Aff")], vpredtest)
  acctest = vperftest[1]
  setest = vperftest[2]
  sptest = vperftest[3]
  mcctest = vperftest[4]

  
  print("===Perf LDA===")
  print(paste("Dim train: ", dim(dtrain)[1]," ", dim(dtrain)[2], sep = ""))
  print(paste("Dim test: ", dim(dtest)[1]," ", dim(dtest)[2], sep = ""))
  
  print("==Train==")
  print(paste("acc=", vperftrain[[1]], sep = ""))
  print(paste("se=", vperftrain[[2]], sep = ""))
  print(paste("sp=", vperftrain[[3]], sep = ""))
  print(paste("mcc=", vperftrain[[4]], sep = ""))
  
  
  print("==Test==")
  print(paste("acc=", vperftest[[1]], sep = ""))
  print(paste("se=", vperftest[[2]], sep = ""))
  print(paste("sp=", vperftest[[3]], sep = ""))
  print(paste("mcc=", vperftest[[4]], sep = ""))
  print("")
  print("")
  
  perftrain = c(acctrain, setrain, sptrain, mcctrain)
  names(perftrain) = c("ACC", "SE", "SP", "MCC")
  
  perftest = c(acctest, setest, sptest, mcctest)
  names(perftest) = c("ACC", "SE", "SP", "MCC")
  
  outmodel = list()
  outmodel$train = perftrain
  outmodel$test = perftest
  outmodel$model = modelLDA
  
  save(outmodel, file = paste(prout, "model.RData", sep = ""))
  
  
  png(paste(prout, "PerfTrainTest.png", sep = ""), 1600, 800)
  par(mfrow = c(1,2))
  plot(dtrain[,"Aff"], vpredtrainprob, type = "n")
  text(dtrain[,"Aff"], vpredtrainprob, labels = names(vpredtrainprob))
  abline(a = 0.5, b = 0, col = "red", cex = 3)
  
  plot(dtest[,"Aff"], vpredtestprob, type = "n")
  text(dtest[,"Aff"], vpredtestprob, labels = names(vpredtestprob))
  abline(a = 0.5, b = 0, col = "red", cex = 3)
  dev.off()
  
  
  write.csv(vpredtestprob, paste(prout,"classTest.csv", sep = ""))
  write.csv(vpredtrainprob, paste(prout,"classTrain.csv", sep = ""))
  
  
  barplotDescriptor(modelLDA, prout, dtrain)
  return(outmodel)
}


barplotDescriptor = function(res.lda, prout, data_train){
  
  aa = dim (data_train)
  coef = res.lda$scaling[,1]
  
  # Standardized
  coef = normalizationScalingLDA (coef, data_train)
  
  vcol = rep(2,length=length(coef))
  names(vcol) = names(coef)
  vcol[which(coef<0)]=4
  coef.sort = sort (abs(coef),decreasing = TRUE)
  png (paste (prout, "coefDesc.png", sep = ""), 2300, 1300)
  par( mar=c(45,10,5,5))
  barplot(coef.sort, names.arg = names(coef.sort), las=2, cex.names = 2.5, col = vcol[names(coef.sort)], main = "", cex.main = 3, cex.axis = 2.7, ylab = "Significativité des descripteurs (%)", cex.lab = 3)
  #legend("topright",legend=c("coef >0","coef <0"), col=c(2,4),lty=2, cex = 2.75)
  dev.off ()
  
  write.table(coef.sort, file = paste(prout, "ImportanceDesc", sep = ""), sep = "\t")
  
}


normalizationScalingLDA = function (scalingLDA, d){
  
  l_out = NULL
  l_des = names (scalingLDA)
  
  for (desc in l_des){
    l_out = append (l_out, normalizationCoef (scalingLDA[desc], d[,c(desc,"Aff")]))
  }
  names (l_out) = names (scalingLDA)
  return (l_out)
}


normalizationCoef = function (coef, data_train){
  
  d_class1 = data_train[which(data_train[,"Aff"]==0),]
  d_class2 = data_train[which(data_train[,"Aff"]==1),]
  
  m_class1 = mean (d_class1[,1])
  m_class2 = mean (d_class2[,1])
  
  v_c1 =  sum((d_class1[,1]-m_class1) * (d_class1[,1]-m_class1))
  v_c2 =  sum((d_class2[,1]-m_class2) * (d_class2[,1]-m_class2))
  
  
  v_out = sqrt((v_c1 + v_c2) / (dim(data_train)[1] - 2))
  
  return (coef*v_out)
  
}


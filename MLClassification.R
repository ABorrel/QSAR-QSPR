#!/usr/bin/env Rscript
source("performance.R")
library("pls")
source("tool.R")
source("dataManager.R")
library (randomForest)
library (MASS)
library(rpart)
library(rpart.plot)
library(e1071)
library(ggplot2)
#library(neuralnet)
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

SVMClassCV = function(lfolds, vgamma, vcost, prout){
  
  print(paste("== SVM in CV with ", length(lfolds), " Automatic optimization by folds", sep = ""))
  
  # data combination
  k = 1
  kmax = length(lfolds)
  y_predict = NULL
  y_real = NULL
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
    
    modtune = tune(svm, Aff~., data = dtrain, ranges = list(gamma = vgamma, cost = vcost), tunecontrol = tune.control(sampling = "fix"))
    
    vpred = predict (modtune$best.model, dtest, type = "class")
    y_proba = append(y_proba, vpred)
    
    vpred[which(vpred < 0.5)] = 0
    vpred[which(vpred >= 0.5)] = 1
    
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
  
  png(paste(prout, "PerfSVMClassCV", length(lfolds), ".png", sep = ""), 800, 800)
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
  
  tperf = cbind(y_predict, y_real)
  write.table(tperf, paste(prout, "perfSVMRegCV_", length(lfolds), ".txt", sep = ""), sep = "\t")
}




#####################
#  Random forest    #
#####################

RFGridClassCV = function(lntree, lmtry, lfolds, prout){
  
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
      
      
      # R2 for grid
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
  
  print(paste("=== RF grid optimisation in CV = ", length(lfolds), " ntree = ", rownames (gridOpt)[which(gridOpt==max(gridOpt), arr.ind=T)[1]], " mtry=", colnames (gridOpt)[which(gridOpt==max(gridOpt), arr.ind=T)[2]], sep = ""))
  return (list(rownames (gridOpt)[which(gridOpt==max(gridOpt), arr.ind=T)[1]],colnames (gridOpt)[which(gridOpt==max(gridOpt), arr.ind=T)[2]] ))
}

RFClassCV = function(lfolds, ntree, mtry, prout){
  
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
  
  # importance descriptors
  Mimportance = apply(timportance, 1, mean)
  SDimportance = apply(timportance, 1, sd)
  
  dimportance = cbind(Mimportance, SDimportance)
  rownames(dimportance) = rownames(timportance)
  dimportance = dimportance[order(dimportance[,1], decreasing = TRUE),]
  
  write.table(dimportance, paste(prout, "ImportanceDescClassCV_", length(lfolds), ".txt", sep = ""), sep = "\t")
  
  png(paste(prout, "ImportanceRFClassCV_", length(lfolds), ".png", sep = ""), 1000, 800)
  par( mar=c(10,4,4,4))
  plot(dimportance[,1], xaxt ="n", xlab="", pch = 19, ylab="M importance")
  axis(1, 1:length(dimportance[,1]), labels = rownames(dimportance), las = 2, cex.axis = 0.7, cex = 2.75)
  for (i in 1:(dim(dimportance)[1])){
    segments(i, dimportance[i,1] - dimportance[i,2], i, dimportance[i,1] + dimportance[i,2])
  }
  dev.off()
  return(y_predict)
}


RFClass = function (dtrain, dtest, ntree, mtry, prout){
  
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
  vperftest = classPerf(dtest[,c("Aff")], vpredtest)
  
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
  
  
  png(paste(prout, "PerfTrainTest.png", sep = ""), 1600, 800)
  par(mfrow = c(1,2))
  plot(dtrain[,"Aff"], vpredtrainprob, type = "n")
  text(dtrain[,"Aff"], vpredtrainprob, labels = names(vpredtrainprob))
  abline(a = 0.5, b = 0, col = "red", cex = 3)
  
  plot(dtest[,"Aff"], vpredtest, type = "n")
  text(dtest[,"Aff"], vpredtestprob, labels = names(vpredtestprob))
  abline(a = 0.5, b = 0, col = "red", cex = 3)
  dev.off()
  write.csv(vpredtestprob, paste(prout,"classTest.csv", sep = ""))
  
}



##########
#  CART  #
##########

CARTClassCV = function(lfolds, prout){
  
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
    
    modelCART = rpart( Aff~., data = dtrain, method = "class")
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
  
  lperf = list()
  lscore = c(acc, se, sp, mcc)
  names(lscore) = c("Acc", "Se", "Sp", "MCC")
  lperf$CV = lscore
  
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
  
  return(lperf)
}



CARTclass = function (dtrain, dtest, prout){
  
  print("== CART in train/test ==")
  
  # model and apply
  modelCART = rpart( Aff~., data = dtrain, method = "class")
  vpredtrain = predict(modelCART, dtrain)
  vpredtest = predict(modelCART, dtest)
  
  #print(vpredtest)
  
  # draw tree
  pdf(paste(prout, "TreeCARTClass-TrainTest.pdf",sep = ""))
  plotcp(modelCART)
  # plot tree in pdf
  rpart.plot( modelCART , # middle graph
              extra=104, box.palette="GnBu",
              branch.lty=3, shadow.col="gray", nn=TRUE)
  dev.off()
  
  
  #result
  vpredtrain = vpredtrain[,1]
  vprobatrain = vpredtrain
  
  vpredtest = vpredtest[,1]
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


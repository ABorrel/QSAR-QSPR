source("performance.R")
library (randomForest)


RFGridMultiClassCV = function(lntree, lmtry, dtrain, prout){
  
  pgrid = paste(prout, "grid.RData", sep = "")
  #if(file.exists(pgrid) == TRUE){
  #  load(pgrid)
  #  return(list(rownames (gridOpt)[which(gridOpt==max(gridOpt), arr.ind=T)[1]],colnames (gridOpt)[which(gridOpt==max(gridOpt), arr.ind=T)[2]] ))
  #}
  
  lfolds = sampligDataMulticlassCV(dtrain, 10, "Aff")
  
  gridOpt = data.frame ()
  gridErr = data.frame ()
  gridModel = list()
  
  i = 0
  for (ntree in lntree){
    i = i + 1
    j = 0
    for (mtry in lmtry){
      j = j + 1
      
      # extrat the bext model
      # data combination
      lmodel = list()
      lperfMCC = NULL
      lperfErr = NULL
      
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
        
        dtrain$Aff = as.factor(dtrain$Aff)
        modelRF = randomForest( Aff~., data = dtrain, mtry=mtry, ntree = ntree, type = "classification",  importance=TRUE)
        vpredtrain = predict (modelRF, dtrain)
        vpredtest = predict (modelRF, dtest)
        
        dclass = multiClassPerf(dtest$Aff, vpredtest)
        MMCC = mean(dclass[,c("MCC")])
        ERR = modelRF$err.rate[dim(modelRF$err.rate)[1], 1]
        #print(modelRF$err.rate)
      
        lperfErr = append(lperfErr, ERR)
        lperfMCC = append(lperfMCC, MMCC)
        lmodel[[k]] = modelRF
      
        k = k + 1
      } 
      
      
      iopt = which(min(lperfErr) == lperfErr)
      if(length(iopt) > 1){
        iopt = iopt[1]
      }
      gridOpt[i,j] = lperfMCC[iopt]
      gridErr[i,j] = lperfErr[iopt]
      
      
      
      if(length(gridModel) == 0){
        gridModel[[i]] = list()
      }
      
      if(length(gridModel) < i){
        gridModel[[i]] = list()
      }
      
      gridModel[[i]][[j]] = lmodel[[iopt]]
     
    }
  }
  colnames (gridOpt) = lmtry
  rownames (gridOpt) = lntree
  
  colnames (gridErr) = lmtry
  rownames (gridErr) = lntree
  
  
  write.table (gridOpt, paste(prout, "RFclassMCC.grid", sep = ""))
  write.table (gridErr, paste(prout, "RFclassERR.grid", sep = ""))
  
  print(which(gridOpt == max(gridOpt), arr.ind = TRUE))
  save(gridOpt, file = paste(prout, "gridMCC.RData", sep = ""))
  save(gridErr, file = paste(prout, "gridERR.RData", sep = ""))
  
  print(paste("=== RF grid optimisation in train, ntree by MCC = ", rownames (gridOpt)[which(gridOpt==max(gridOpt), arr.ind=T)[1]], " mtry=", colnames (gridOpt)[which(gridOpt==max(gridOpt), arr.ind=T)[2]], sep = ""))
  print(paste("=== RF grid optimisation in train, ntree by Err = ", rownames (gridErr)[which(gridErr==max(gridErr), arr.ind=T)[1]], " mtry=", colnames (gridErr)[which(gridErr==max(gridErr), arr.ind=T)[2]], sep = ""))
  
  
  return (gridModel[[which(gridErr==max(gridErr), arr.ind=T)[1]]][[which(gridErr==max(gridErr), arr.ind=T)[2]]])
  return (gridModel[[which(gridOpt==max(gridOpt), arr.ind=T)[1]]][[which(gridOpt==max(gridOpt), arr.ind=T)[2]]])
  
  #return (list(rownames (gridErr)[which(gridErr==min(gridErr), arr.ind=T)[1]],colnames (gridErr)[which(gridErr==min(gridErr), arr.ind=T)[2]] ))
  #return (list(rownames (gridOpt)[which(gridOpt==max(gridOpt), arr.ind=T)[1]],colnames (gridOpt)[which(gridOpt==max(gridOpt), arr.ind=T)[2]] ))

}



RFMultiClassTrainTest = function (dtrain, dtest, modelRF, prout){
  
  pmodel = paste(prout, "model.RData", sep = "")
  #if(file.exists(pmodel)){
  #  load(pmodel)
  #  return(outmodel) 
  #}
  
  
  #dtrain$Aff = as.factor(dtrain$Aff)
  #modelRF = randomForest( Aff~., data = dtrain, mtry=as.integer(mtry), ntree=as.integer(ntree), type = "classification",  importance=TRUE)
  
  vpredtrain = predict (modelRF, dtrain, type = "class")
  vpredtest = predict (modelRF, dtest, type = "class")
  
  dclassTrain = multiClassPerf(dtrain$Aff, vpredtrain)
  dclassTest = multiClassPerf(dtest$Aff, vpredtest)
  
  write.csv(dclassTrain, paste(prout, "perfTrain.csv", sep =""))
  write.csv(dclassTest, paste(prout, "perfTest.csv", sep =""))
  
  
  print("===Perf RF===")
  print(paste("Dim train: ", dim(dtrain)[1]," ", dim(dtrain)[2], sep = ""))
  print(paste("Dim test: ", dim(dtest)[1]," ", dim(dtest)[2], sep = ""))
  
  print("==Train==")
  print(dclassTrain)
  
  
  print("==Test==")
  print(dclassTest)
  print("")
  print("")
  
  
  outmodel = list()
  outmodel$train = dtrain
  outmodel$test = dtest
  outmodel$model = modelRF
  
  # importance
  timportance = modelRF$importance[,1]
  write.table(timportance, paste(prout, "ImportanceDesc", sep = ""), sep = "\t")
  outmodel$importance = timportance
  
  dimportance = read.table(paste(prout, "ImportanceDesc", sep = ""))
  
  ORDER = order(dimportance[,1], decreasing = T)
  NAME = rownames(dimportance)
  
  dimportance = cbind(dimportance, NAME)
  dimportance = cbind (dimportance, ORDER)
  #dimportance = dimportance[ORDER[seq(1,10)],]
  dimportance = as.data.frame(dimportance)
  
  p = ggplot(dimportance, aes(NAME, x, fill = 1)) + 
    geom_bar(stat = "identity", show.legend = FALSE) + 
    #scale_x_continuous(breaks = -dimportance$ORDER, labels = dimportance$NAME)+
    theme(axis.text.y = element_text(size = 15, hjust = 0.5, vjust =0.1), axis.text.x = element_text(size = 15, hjust = 0.5, vjust =0.1), axis.title.y = element_text(size = 15, hjust = 0.5, vjust =0.1), axis.title.x =  element_text(size = 15, hjust = 0.5, vjust =0.1))+
    labs(y = "", x = "")+ 
    #ylim (c(0, 0.5)) +
    coord_flip()
  
  ggsave(paste(prout, "ImportanceDescRF.png", sep = ""), width = 6,height = 6, dpi = 300)
  
  
  save(outmodel, file = paste(prout, "model.RData", sep = ""))
  
  png(paste(prout, "PerfTrainTest.png", sep = ""), 1600, 800)
  par(mfrow = c(1,2))
  plot(as.double(as.vector(dtrain[,"Aff"])),as.double(as.vector(dtrain[,"Aff"])), type = "n")
  text(as.double(as.vector(dtrain[,"Aff"])), as.double(as.vector(vpredtrain)), labels = names(vpredtrain))
  
  plot(as.double(as.vector(dtest[,"Aff"])), as.double(as.vector(dtest[,"Aff"])), type = "n")
  text(as.double(as.vector(dtest[,"Aff"])), as.double(as.vector(vpredtest)), labels = names(vpredtest))
  abline(a = 0.5, b = 0, col = "red", cex = 3)
  dev.off()
  
  write.csv(vpredtest, paste(prout,"classTest.csv", sep = ""))
  write.csv(vpredtrain, paste(prout,"classTrain.csv", sep = ""))
  
  return(outmodel)
}



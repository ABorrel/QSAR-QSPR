source("performance.R")
library (randomForest)


RFGridMultiClassCV = function(lntree, lmtry, dtrain, prout){
  
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
      
      modelRF = randomForest( Aff~., data = dtrain, mtry=mtry, ntree = ntree, type = "class",  importance=TRUE)
      vpredtrain = round(predict (modelRF, dtrain))
      
      dclass = multiClassPerf(dtrain$Aff, vpredtrain)
      MMCC = mean(dclass[,c("MCC")])
      #print(MMCC)

      gridOpt[i,j] = MMCC
      
      # R conversion 
    }
  }
  
  colnames (gridOpt) = lmtry
  rownames (gridOpt) = lntree
  
  write.table (gridOpt, paste(prout, "RFclassMCC.grid", sep = ""))
  #print(which(gridOpt == max(gridOpt), arr.ind = TRUE))
  save(gridOpt, file = paste(prout, "grid.RData", sep = ""))
  
  print(paste("=== RF grid optimisation in train, ntree = ", rownames (gridOpt)[which(gridOpt==max(gridOpt), arr.ind=T)[1]], " mtry=", colnames (gridOpt)[which(gridOpt==max(gridOpt), arr.ind=T)[2]], sep = ""))
  return (list(rownames (gridOpt)[which(gridOpt==max(gridOpt), arr.ind=T)[1]],colnames (gridOpt)[which(gridOpt==max(gridOpt), arr.ind=T)[2]] ))
}



RFMultiClassTrainTest = function (dtrain, dtest, ntree, mtry, prout){
  
  pmodel = paste(prout, "model.RData", sep = "")
  if(file.exists(pmodel)){
    load(pmodel)
    return(outmodel) 
  }
  
  
  modelRF = randomForest( Aff~., data = dtrain, mtry=as.integer(mtry), ntree=as.integer(ntree), type = "class",  importance=TRUE)
  
  vpredtrainprob = predict (modelRF, dtrain, type = "class")
  vpredtestprob = predict (modelRF, dtest, type = "class")
  vpredtrain = round(vpredtrainprob)
  vpredtest = round(vpredtestprob)
  
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
  dimportance = dimportance[ORDER[seq(1,10)],]
  dimportance = as.data.frame(dimportance)
  print(dimportance)
  
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
  plot(dtrain[,"Aff"], vpredtrainprob, type = "n")
  text(dtrain[,"Aff"], vpredtrainprob, labels = names(vpredtrainprob))
  
  plot(dtest[,"Aff"], vpredtest, type = "n")
  text(dtest[,"Aff"], vpredtestprob, labels = names(vpredtestprob))
  abline(a = 0.5, b = 0, col = "red", cex = 3)
  dev.off()
  
  write.csv(vpredtestprob, paste(prout,"classTest.csv", sep = ""))
  write.csv(vpredtrainprob, paste(prout,"classTrain.csv", sep = ""))
  
  return(outmodel)
}



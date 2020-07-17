#!/usr/bin/env Rscript
source("performance.R")

rectangle <- function(x, y, width, height, density=12, angle=-45, ...) 
  polygon(c(x,x,x+width,x+width), c(y,y+height,y+height,y), 
          density=density, angle=angle, ...)


# draw ROC
ROCCurve = function(vreal, vpred){
  
  dROC = NULL
  for(i in seq(0,1,0.05)){
    vpredROC = vpred
    vpredROC[which(vpred < i)] = 0
    vpredROC[which(vpred >= i)] = 1
    
    vperf = classPerf(vreal, vpredROC)
    dROC = rbind(dROC, c(i, vperf[[2]], vperf[[3]]))
  }
  colnames(dROC) = c("Sep", "Sensitivity", "Specificity")
  dROC = as.data.frame(dROC)
  
  return(dROC)
}


simple_auc <- function(TPR, FPR){
  # inputs already sorted, best scores first 
  dFPR <- c(diff(FPR), 0)
  dTPR <- c(diff(TPR), 0)
  return(sum(TPR * dFPR) + sum(dTPR * dFPR)/2)
}

##########
# main   #
##########
args <- commandArgs(TRUE)
p_pred = args[1]

p_pred = "./../../HERG/results/QSAR/Merge_probRF/Prob_test"



d_pred = read.csv(p_pred, sep = "\t")


category = d_pred$Real
category[which(is.na(d_pred$Real))] = 0
category[which(d_pred$Real!= 0)] = 1

prediction <- rev(seq_along(category))
dROC = ROCCurve(category, d_pred$Mpred)


roc_df <- data.frame(
  TPR=rev(dROC$Sensitivity), 
  FPR=rev(1 - dROC$Specificity)) 



# plot 
roc_df <- transform(roc_df, 
                    dFPR = c(diff(FPR), 0),
                    dTPR = c(diff(TPR), 0))


AUC = with(roc_df, simple_auc(TPR, FPR))


png(paste(p_pred, "_AUC.png", sep = ""))
plot(0:10/10, 0:10/10, type='n', xlab="FPR", ylab="TPR")
abline(h=0:10/10, col="lightblue")
abline(v=0:10/10, col="lightblue")

with(roc_df, {
  mapply(rectangle, x=FPR, y=0,   
         width=dFPR, height=TPR, col="green", lwd=2)
  mapply(rectangle, x=FPR, y=TPR, 
         width=dFPR, height=dTPR, col="blue", lwd=2)
  lines(FPR, TPR, type='b', lwd=3, col="red")
  text(0.65, 0.5, paste("AUC: ", round(AUC,2), sep = ""), cex = 1.8)
})

dev.off()








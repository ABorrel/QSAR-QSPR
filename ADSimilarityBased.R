#!/usr/bin/env Rscript
library(ggfortify)
library(tripack)


################
#     MAIN     #
################

args <- commandArgs(TRUE)
p_sim_matrix = args[1]
p_matrix_chem = args[2]
pr_out = args[3]

#p_sim_matrix = "/mnt/c/Users/AlexandreBorrel/research/SSI/E2up_P4up/results/QSAR_P4_H295R_variable-sampling_singledosecheck_noborderline/mat_sim_MACCS_Tanimoto.csv"
#p_matrix_chem = "/mnt/c/Users/AlexandreBorrel/research/SSI/E2up_P4up/results/QSAR_P4_H295R_variable-sampling_singledosecheck_noborderline/rdkit-OPERA-toxprint_0.9-0/classQSAR/6/AD/chem_similarity/chem.csv"
#pr_out = "/mnt/c/Users/AlexandreBorrel/research/SSI/E2up_P4up/results/QSAR_P4_H295R_variable-sampling_singledosecheck_noborderline/rdkit-OPERA-toxprint_0.9-0/classQSAR/6/AD/chem_similarity/"



d_chem = read.csv(p_matrix_chem, sep = "\t", row.names = 1)
d_sim = read.csv(p_sim_matrix, sep = "\t", row.names = 1)
colnames(d_sim) = rownames(d_sim)
# define col vector


# remove row with more than n NA
cnt_na <- apply(d_sim, 1, function(z) sum(is.na(z)))
nb_NA = min(cnt_na) + 1

d_sim = d_sim[cnt_na < nb_NA,]
d_sim = d_sim[,cnt_na < nb_NA]

l_casrn = intersect(rownames(d_sim), rownames(d_chem))
d_sim = d_sim[l_casrn, ]
d_sim = d_sim[,l_casrn]
d_chem = d_chem[l_casrn,]

# disimilarity
d_sim = 1-d_sim

l_set = rep("Train Act", dim(d_chem)[1])
l_set[which(d_chem$Aff == 0 & d_chem$set=="train")] = "Train Inact"
l_set[which(d_chem$Aff == 0 & d_chem$set=="test")] = "Test Inact"
l_set[which(d_chem$Aff == 1 & d_chem$set=="test")] = "Test Act"

l_color = rep("#8dbfe4", dim(d_chem)[1])
l_color[which(d_chem$Aff == 0 & d_chem$set=="train")] = "#6475e8"
l_color[which(d_chem$Aff == 0 & d_chem$set=="test")] = "#a62a9a"
l_color[which(d_chem$Aff == 1 & d_chem$set=="test")] = "#f378b5"


# reduce matrix
fit_MDS <- cmdscale(d_sim, eig=TRUE, k=2)
fit_MDS = fit_MDS$points
fit_MDS = cbind(fit_MDS, l_set)
colnames(fit_MDS) = c("Dim1", "Dim2", "Set")
fit_MDS = as.data.frame(fit_MDS)
fit_MDS$Dim1 = as.double(as.character(fit_MDS$Dim1))
fit_MDS$Dim2 = as.double(as.character(fit_MDS$Dim2))


cmd_plot<- ggplot(fit_MDS, aes(x=Dim1, y=Dim2)) +
  geom_text(aes(colour=Set, label = rownames(fit_MDS)),  size=2) +
  scale_colour_manual(name = "Set", values = c("Train Act" = "#8dbfe4", "Train Inact" = "#6475e8", "Test Act" = "#f378b5", "Test Inact" = "#a62a9a"))  
ggsave(paste(pr_out, "CP_similarity_text.png", sep = ""),  width = 8, height = 6, dpi = 300, bg="transparent")

cmd_plot<- ggplot(fit_MDS, aes(x=Dim1, y=Dim2)) +
  geom_point(aes(colour=Set, label = rownames(fit_MDS)),  size=2) +
  scale_colour_manual(name = "Set", values = c("Train Act" = "#8dbfe4", "Train Inact" = "#6475e8", "Test Act" = "#f378b5", "Test Inact" = "#a62a9a"))  
ggsave(paste(pr_out, "CP_similarity_point.png", sep = ""),  width = 8, height = 6, dpi = 300, bg="transparent")

# mosaic plot
#plot(voronoi.mosaic(fit_MDS$Dim1, fit_MDS$Dim2, duplicate="remove"), colors=l_color)


# Traing and Test Scores
test_acc = pobj$scores_dt[pobj$scores_dt$score_var == "xentropy" & pobj$scores_dt$data == "test"]
train_acc = pobj$scores_dt[pobj$scores_dt$score_var == "xentropy" & pobj$scores_dt$data == "train"]
write.csv(test_acc, file = "/home/julian/Uni/MasterThesis/code/experiments/03_Acinar_R_analysis/test_scores.csv")
write.csv(train_acc, file = "/home/julian/Uni/MasterThesis/code/experiments/03_Acinar_R_analysis/train_scores.csv")

# Weights
write.csv(data.frame(as.matrix(pobj$glmnet_best$beta)), file="Uni/MasterThesis/code/experiments/03_Acinar_R_analysis/weights.csv")


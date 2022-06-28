library(reshape2)
library(dplyr)
library(dbarts)
library(tidyverse)

set.seed(123456)

new_amines = c('HJFYRMFYQMIZDG-UHFFFAOYSA-N',
               'JMXLWMIFDJCGBV-UHFFFAOYSA-N',
               'NJQKYAASWUGXIT-UHFFFAOYSA-N',
               'ZKRCWINLLKOVCL-UHFFFAOYSA-N')

# select amine
new_amine = new_amines[1]

# pick model 0 or 1
model = 0

# k is the active learning step
k = 1

# set directories
data_dir = '../../../../data/'
stateset_dir = paste(data_dir, 'stateset/', new_amine,  '/stateset.csv', sep = '')
volume_dir = paste(data_dir, 'stateset/', new_amine,  '/stateset_volumes.csv', sep = '')
training_dir = paste(data_dir, "training/historical/raw_data.csv", sep = '')
draw_dir = paste(data_dir, "training/initialization/", new_amine, "/training_draw", model, ".csv", sep = '')

# read in training data
dat_train = read.csv(training_dir, header = T, fill = T, stringsAsFactors = F)
# read in stateset data
dat_stateset = read.csv(stateset_dir, header = T, stringsAsFactors = F)
# read in warmstart data
dat_new_amine = read.csv(draw_dir, header = T, stringsAsFactors = F)

# get dictionary to map inchikeys to smiles
inventory = read.csv("data/inventory.csv", header = T, stringsAsFactors = F)
inventory = inventory[-1, c(6, 8)]
names(inventory) = c("X_rxn_organic.inchikey", "X_rxn_organic.smile")

# add smiles to datasets
dat_train = left_join(dat_train, inventory, by = "X_rxn_organic.inchikey")
dat_new_amine = left_join(dat_new_amine, inventory, by = "X_rxn_organic.inchikey")
dat_stateset = left_join(dat_stateset, inventory, by = "X_rxn_organic.inchikey")

wrong_amine = 'UMDDLGMCNFAZDX-UHFFFAOYSA-O'
viable_amines = c('ZEVRFFCPALTVDN-UHFFFAOYSA-N',
                  'KFQARYBEAKAXIC-UHFFFAOYSA-N',
                  'NLJDBTZLVTWXRG-UHFFFAOYSA-N',
                  'LCTUISCIGMWMAT-UHFFFAOYSA-N',
                  'JERSPYRKVMAEJY-UHFFFAOYSA-N',
                  'VAWHFUNJDMQUSB-UHFFFAOYSA-N',
                  'WGYRINYTHSORGH-UHFFFAOYSA-N',
                  'FCTHQYIDLRRROX-UHFFFAOYSA-N',
                  'VNAAUNTYIONOHR-UHFFFAOYSA-N',
                  'KOAGKPNEVYEZDU-UHFFFAOYSA-N',
                  'FJFIJIDZQADKEE-UHFFFAOYSA-N',
                  'XFYICZOIWSBQSK-UHFFFAOYSA-N',
                  'KFXBDBPOGBBVMC-UHFFFAOYSA-N',
                  'HBPSMMXRESDUSG-UHFFFAOYSA-N',
                  'NXRUEVJQMBGVAT-UHFFFAOYSA-N',
                  'CALQKRVFTWDYDG-UHFFFAOYSA-N',
                  'LLWRXQXPJMPHLR-UHFFFAOYSA-N',
                  'BAMDIFIROXTEEM-UHFFFAOYSA-N',
                  'XZUCBFLUEBDNSJ-UHFFFAOYSA-N')

hold_out_amines = c('CALQKRVFTWDYDG-UHFFFAOYSA-N',
                    'KOAGKPNEVYEZDU-UHFFFAOYSA-N',
                    'FCTHQYIDLRRROX-UHFFFAOYSA-N')

# keep uniformly sampled historical data
keep = contains("Uniform", vars = dat_train$X_raw_modelname)
dat_train = dat_train[keep, ]

# binarize crystal scores
dat_train$X_out_crystalscore[dat_train$X_out_crystalscore %in% c(1:3)] = 0
dat_train$X_out_crystalscore[dat_train$X_out_crystalscore %in% c(4)] = 1

dat_new_amine$X_out_crystalscore[dat_new_amine$X_out_crystalscore %in% c(1:3)] = 0
dat_new_amine$X_out_crystalscore[dat_new_amine$X_out_crystalscore %in% c(4)] = 1

# training set is 16 old + 3 holdout amines
dat_train = dat_train[dat_train$X_rxn_organic.inchikey %in% viable_amines, ]
dat_train = dat_train[!(dat_train$X_rxn_organic.inchikey %in% new_amine), ]

length(unique(dat_train$X_rxn_organic.inchikey))

# ---------------------------------------
# rm column in dat_train not in dat_stateset
which_col = which(names(dat_train) == 'X_feat_vdw_per_N')
dat_train = dat_train[, -which_col]

#  change colname of dat_new_amine to match dat_train
col_names = names(dat_new_amine) 
col_names[1] = 'name'
names(dat_new_amine) = col_names
dat_new_amine = dat_new_amine[, names(dat_train)]

# read in outputs from chemprop trained on asa etc.
chemprop = read.csv(file = "data/chemprop_embeddings.csv", header = T, row.names = 1)

chemprop_train = chemprop[dat_train$X_rxn_organic.smile, ]

chemprop_new_amine = chemprop[dat_new_amine$X_rxn_organic.smile, ]

chemprop_stateset = chemprop[dat_stateset$X_rxn_organic.smile, ]
#-------------------------------------------------------------------------------

class_weights = 6

dat_train_jumpstart = dat_new_amine

dat_train_all = rbind(dat_train, dat_train_jumpstart)

if (k > 1) {
  dat_AL = read.csv(paste("data/", new_amine, "/dat_AL_draw_", model, "_k_", 1, ".csv", sep = ''), header = T, stringsAsFactors = F)

  if (k > 2) {
    for (i in 2:(k-1)) {
      new = read.csv(paste("data/", new_amine, "/dat_AL_draw_", model, "_k_", i, ".csv", sep = ''), header = T, stringsAsFactors = F)
      dat_AL = rbind(dat_AL, new)
    }
  }
  dat_AL$X_out_crystalscore[dat_AL$X_out_crystalscore %in% c(1:3)] = 0
  dat_AL$X_out_crystalscore[dat_AL$X_out_crystalscore %in% c(4)] = 1
  
  dat_stateset = dat_stateset[!(dat_stateset$name %in% dat_AL$name), ]
  dat_AL = dat_AL[, names(dat_train_all)]
  dat_train_all = rbind(dat_train_all, dat_AL)
}

X_tr = dat_train_all[, !(names(dat_train_all) %in% c("name", "X_raw_modelname", "X_rxn_organic.inchikey", "X_rxn_organic.smile", "X_out_crystalscore"))]
chemprop_tr = chemprop[dat_train_all$X_rxn_organic.smile, ]
X_train = cbind(X_tr, chemprop_tr)
X_train = as.matrix(X_train)
  
X_ts = dat_stateset[, !(names(dat_stateset) %in% c("name", "X_raw_modelname", "X_rxn_organic.inchikey", "X_rxn_organic.smile", "X_out_crystalscore"))]
chemprop_ts = chemprop[dat_stateset$X_rxn_organic.smile, ]
X_ts = X_ts[, names(X_tr)]

X_test = cbind(X_ts, chemprop_ts)
X_test = as.matrix(X_test)
  
Y_train = dat_train_all$X_out_crystalscore
  
class_weight = numeric(length(Y_train))
class_weight[Y_train == 0] = 1
class_weight[Y_train == 1] = class_weights
  
R = 1000
bart_out = bart(X_train, Y_train, X_test, weights = class_weight,
              nskip = 1000, ndpost = R, ntree = 200)
  
bart_yhat_mean = apply(pnorm(bart_out$yhat.test), 2, mean)
  
# max uncertainty rule
uncertainty = bart_yhat_mean
uncertainty[bart_yhat_mean > 0.5] = 1 - bart_yhat_mean[bart_yhat_mean > 0.5]
uncertainty[bart_yhat_mean <= 0.5] = bart_yhat_mean[bart_yhat_mean <= 0.5]
  
AL_add = dat_stateset$name[which.max(uncertainty)]

write.csv(dat_stateset[dat_stateset$name == AL_add,], file = paste("data/", new_amine, "/dat_AL_draw_", model, "_k_", k, ".csv", sep = ''), row.names = F)

reagents = read.csv(volume_dir, header = T)
col_names = names(reagents)
col_names[1] = 'name'
names(reagents) = col_names

write.csv(reagents[reagents$name == AL_add,], file = paste("data/", new_amine, "/dat_AL_draw_", model, "_k_", k, "_reagents.csv", sep = ''), row.names = F)





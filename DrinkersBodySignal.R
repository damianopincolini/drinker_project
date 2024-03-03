# 1. PROJECT INTRODUCTION
  
# 1.1. PROJECT GOAL
# The aim of this project is to investigate the potential relation between a target value
# (drinker status: "N", "Y") and some predictors based on body signals such as qualitative
# variables (i.e. smoker status) and quantitative features (i.e. cholesterol).


# 1.2. LOADING LIBRARIES

pacman::p_load(readr,
               tidyverse, 
               tidymodels,
               DataExplorer,
               SmartEDA,
               ggcorrplot,
               corrplot,
               moments,
               skimr,
               knitr,
               DescTools,
               stats, 
               vip,
               datawizard,
               GGally,
               caret,
               MASS,
               kknn,
               rpart.plot,
               healthyR.ai,
               embed,
               cowplot,
               factoextra,
               LiblineaR,
               shapviz)

# 1.3. LOADING DATA
# The dataset is taken from this Kaggle's page:
# <https://www.kaggle.com/datasets/sooyoungher/smoking-drinking-dataset.>
# After downloading it on the laptop hard disk, it has been imported into R as a data frame
# called "DataOrigin".

DataOrigin <- read_csv("DataKaggle.csv")


# 1.4. DATA CHECK

# 1.4.1. Dataset structure and datatype analysis

DataExp1 <- ExpData(DataOrigin, type=1) 
kable(DataExp1)

DataExp2 <- ExpData(DataOrigin, type=2)
kable(DataExp2)


# At a first sight, we can observe what follows:
# "sex" and "DRK_YN" columns which are character data type.
# All the other variables are numeric.
# "hear_left", "hear_right" are dicotomic numeric features (1=normal, 2=abnormal).
#   They should be represented by a factor.
# "urine_protein" is a numeric ordinal variable to be transformed into a factor
#   (1=-, 2=+/-, 3=+1, 4=+2, 5=+3, 6=+4).
# "SMK_stat_type" is a numeric ordinal variable to be transformed into a factor
#   (1=never, 2=used to smoke but quit, 3=still smoke).
# "DRK_YN" is a dicotomic character features (Y, N) that should be turned into a factor.
#  This is supposed to be the target variable.
# There are visible differences in features' scales.
# There are no missing value. No imputation is thus needed. 


# 1.4.2. Conversion to factor
# I want to force to factor every feature that come with a numerical or categorical class.

Data <- DataOrigin|>
  mutate(sex=as_factor((sex)),
         drkStatus=as_factor((DRK_YN)),
         hear_left=as_factor((hear_left)),
         hear_right=as_factor(hear_right),
         urine_protein=as_factor(urine_protein),
         smk_status=as_factor(SMK_stat_type_cd),
         .keep="unused")


# 1.5. SAMPLING AND DATA PARTITIONING

set.seed(666, sample.kind = "Rounding")

Index<- sample(1:nrow(Data),
                     size=100000,
                     replace=FALSE)

DataReduced <- Data[Index, ]

DataSplit <- initial_split(DataReduced,
                            prop=0.8,
                            strata=drkStatus)

trainData <- training(DataSplit)

testData <- testing(DataSplit)



# 2. EDA

# 2.1. Univariate analysis

# 2.1.1. Numerical features
# I keep on using smartEDA package. The ExpNumStat command gives a pretty clear bird's
# eye view of how every single variable behaves.

DataSmartEDA <- ExpNumStat(trainData,
                           by="A",
                           gp=NULL,
                           Qnt=c(0.25,0.75),
                           MesofShape=2,
                           Outlier=TRUE,
                           round=2)
                           
kable(DataSmartEDA)

# SGOT_ALT and SGOT_AST have a CV around 100% (102% and 90% respectively), gamma_GTP' CV
#   measures 1.36.
# Several variables have a very high skewness and kurtosis and quite a high number of
#   outliers.
# Generally speaking, based on these statistics, it seems there's quite a clear need for
#   feature transformation.

# I definitely want to properly detect outliers.

DataOutlier <- ExpOutliers(trainData,
                           varlist = c("gamma_GTP", "SGOT_AST", "SGOT_ALT"),
                           method="boxplot")$outlier_summary

kable(DataOutlier)


# Let's move on and look for some pattern related to numerical features distribution

ExpNumViz(data=trainData,
          target="drkStatus",
          type=1,
          Page=c(3,2))

# The numerical features visualization allows us to see that some predictors affect
# significantly the target value, while others don't.
# Age, height, weight look like the most relevant to this extent, even if the
# different scale of each box plot could lead to some kind of misinterpretation.


# 2.1.2. Categorical features
# Let's summarize categorical features with ExpCatStat() command from smartEDA package.

DataSmartEda_cat <- ExpCatStat(trainData,
                               Target="drkStatus",
                               result="stat",
                               clim=10,
                               nlim=5,
                               bins=10,
                               Pclass="Y",
                               plot=FALSE,
                               top=20,
                               Round=2)|>
  rename(PredPower='Predictive Power')|>
  dplyr::filter(PredPower=="Highly Predictive")

kable(DataSmartEda_cat)



# We can see six predictors that seem to have a highly predictive power. 
# Compared to decision tree's most important variable, five out of six are the same.
# Serum_creatinine is considered amongst the top 6 by decision tree, while in Information
#   Value's ranking it gets a "somewhat predictive"/"moderate" predictive power.
# On the other hand, gamma_GTP is more considered by Information value while reaches the
#   eight place only in decision tree's important variables ranking.
# Amongst these six feature only height and weight seems pretty much correlated (0.7).
#   Very, very curiously, height is more correelated than weight to drinking habit (the
#   target value).

# Is there something we can observe about categorical features distribution?
#  I can quickly find it out with (again: from smartEDa) ExpCatViz().

ExpCatViz(data=trainData,
          target="drkStatus",
          Page=c(3,2),
          Flip=TRUE,
          col=c("blue", "violet"))

# Sex and smoking habits (smk_status) features look correlated to the target variable.
# Specifically, males tend to smoke more frequently and non smokerks (status 1) are
# tipically non drinkers, while ex smokers or actual smokers more often use alchool.
# Other categorical variables don't seem to be significantly associated to drinking habits.


# 2.2. Multivariate analysis

# 2.2.1. Correlation between numerical predictors

corMatr <- round(cor(trainData[,-c(1,8,9,18,23,24)], use="complete.obs"), 1)

ggcorrplot(corMatr,
           hc.order = TRUE,
           type = "lower",
           lab = TRUE)

# Only a few numerical features seem to be significantly mutually correlated:
# tot_chole and LDL_chole (0.9),
# height and weight (0.7),
# SBP and DBP (0.7).
# According to this, it's logical to take into account the idea of eliminating one for
#   each couple of correlated features above.


# 2.2.2. Principal Component Analysis (factoextra package)

# With correlation plot, I have taken a first glance to see if some variable may be
# considered "useless" because its potential predictive power is still provided by
# another variable(s) and thus semplify the dataset and reducing the "curse of
# dimensionality" risk. Another analysis I can do toward this goal may be Principal
# Compentent Analysis (PCA); if I can reduce the dataset features in a smaller number of
# brand new variable (or principal components) and the very first principal components
# (let's say from two to four) are able to explain the most target variable variance,
# I could feed machine learning models with quite a light and simple predictors' dataset
# which, in some cases could be very useful. 

dsPca2 <- trainData|>
  dplyr::select(where(is.numeric))|>
  prcomp(scale=TRUE)

summary(dsPca2)

# It takes 9 principal component to explain the 80% of target variance.
# Of course, it is necessary to keep in mind that PCA only affects numerical predictors.

                                                                                                                                                                                                                                      
                                                                                                                                                                                                                                         
                                                                                                                                                                                                                                         
                                                                                                                                                                                                                                         
                                                                                                                                                                                                                                         
                                                                                                                                                                                                                                         
                                                                                                                                                                                                                                         
                                                                                                                                                                                                                                         
                                                                                                                                                                                                                                         
                                                                                                                                                                                                                                         
                                                                                                                                                                                                                                         
                                                                                                                                                                                                                                         
                                                                                                                                                                                                                                         
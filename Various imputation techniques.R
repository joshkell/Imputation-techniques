#https://www.analyticsvidhya.com/blog/2016/03/tutorial-powerful-packages-imputing-missing-values/  common packages MICE, missForest, amelia, Hmisc, mi
#in this code I've used MICE and Amelia

######################
#no imputation
no_missing_data <- na.omit(clean_debt_discharge)

hist(clean_debt_discharge$DEBT_TERM_UG_GPA)
summary(clean_debt_discharge$DEBT_TERM_UG_GPA)


#########################
#model mean imputation (could also do median imputation) - this does introduce bias and strength relationships present in the data
clean_debt_discharge$GPA_imp_mean <- clean_debt_discharge$DEBT_TERM_UG_GPA
clean_debt_discharge$GPA_imp_mean[is.na(clean_debt_discharge$GPA_imp_mean)] = mean(clean_debt_discharge$DEBT_TERM_UG_GPA, na.rm=TRUE)
summary(clean_debt_discharge$GPA_imp_mean) #no missing values now
hist(clean_debt_discharge$GPA_imp_mean)

#imputed as mean + missing indicator method
#advantage is that it treats those with a missing GPA as systematically different
clean_debt_discharge <- dplyr::mutate(clean_debt_discharge,
                                GPA_missing_ind = dplyr::case_when(
                                    DEBT_TERM_UG_GPA >= 0 ~ 1,
                                    TRUE ~ 0
                                ))
#more complex strategies
# 1) Predicting based on other data
# 2) Random sampling based on the distribution of the variable

#HOT Deck - Requires data to be MCAR (takes the last observed value) - Don't meet that assumption with GPA
#can stratify the imputation by a variable (i.e. gender); can also sort data by a variable


#regression imputation
#MICE https://www.rdocumentation.org/packages/mice/versions/3.14.0/topics/mice
library(mice)
set.seed(9999)
#deterministic - replaces values with the exact prediction.
#Random variation is not considered. Overestimates the relationship between X and Y
clean_debt_discharge2 = subset(clean_debt_discharge, select = -c(GPA_scaled, GPA_CAT, GPA_CAT_ord,
                                                                 age_cat, age_cat_ord, GPA_imp,
                                                                 continue_SumF))

imp <- mice(clean_debt_discharge2, method= "norm.predict", m=5)
data2 <- complete(imp)
summary(data2$DEBT_TERM_UG_GPA) #does create a negative GPA...
hist(data2$DEBT_TERM_UG_GPA)
summary(data2$DEBT_TERM_UG_GPA)

modelRETURN_imp_det <- glm(RETURNED1X ~ HISPANIC_IND + GENDER + relevel(factor(age_cat2), ref=2) + GPA_ind + DEBT_TERM_UG_GPA + FIRST_GENERATION_IND + relevel(factor(RACE), ref =7) + balance_100 + EVER_PELL_ELIGIBLE_IND, data=data2, family=binomial())
summary(modelRETURN_imp_det)

oddsRe_imp_det <- exp(cbind(OR=coef(modelRETURN_imp_det), confint(modelRETURN_imp_det)))
oddsRe_imp_det


modelRETURN_imp_det5 <- with(imp, glm(RETURNED1X ~ HISPANIC_IND + GENDER + relevel(factor(age_cat2), ref=2) + GPA_ind + DEBT_TERM_UG_GPA + FIRST_GENERATION_IND + relevel(factor(RACE), ref =7) + balance_100 + EVER_PELL_ELIGIBLE_IND, family=binomial(link = "logit")))
summary(pool(modelRETURN_imp_det5), conf.int = TRUE, exponentiate = TRUE)


#stochastic - adds random error term to the predicted value - multiple imputations change m=2+ (tried 5)
imp2 <- mice(clean_debt_discharge2, method= "norm.nob", m=1)
data3 <- complete(imp2)
summary(data3$DEBT_TERM_UG_GPA) #does create a negative GPA...
hist(data3$DEBT_TERM_UG_GPA)
summary(data3$DEBT_TERM_UG_GPA)


#Random forest (Reference: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3939843/)
imp_rf <- mice(clean_debt_discharge2, method= "rf", m=5)
data_rf <- complete(imp_rf)
summary(data_rf$DEBT_TERM_UG_GPA) #does NOT create a negative GPA
hist(data_rf$DEBT_TERM_UG_GPA)
summary(data_rf$DEBT_TERM_UG_GPA)




#Amelia - multiple imputation using Expectation-Maximization - finds the most likely value and
   #expectation-maximization with bootstrapping
   #https://pop.princeton.edu/document/5151

clean_debt_discharge3 = subset(clean_debt_discharge2, select = -c(enrolled, LAST_GRAD_TERM_NAME, continue_SumSpr, continue_FSpr, BALANCE_SPRING_SUMMER_2020, debt_in_spring, debt_in_summer, debt_term, debt_both))
View(clean_debt_discharge3)
library(Amelia)
missmap(clean_debt_discharge3)
completed_data <- amelia(clean_debt_discharge3, m=5, p2s=0, noms = c("RACE", "IPEDS_ETHNICITY", "FIRST_GENERATION_IND"),
                         idvars = c("BANNER_ID", "LAST_TERM_SLCC", "GPA_ind", "HISPANIC_IND", "age_cat2"))
View(completed_data$imputations)
data_EM <- completed_data$imputations$imp1
hist(data_EM$DEBT_TERM_UG_GPA)
summary(data_EM$DEBT_TERM_UG_GPA)


#CART - Classification and regression tree
library(simputation) #this package can do hotdeck, RF, CART, lasso regression, linear regression, knn, EM etc. https://cran.r-project.org/web/packages/simputation/vignettes/intro.html
data_CART <- impute_cart(clean_debt_discharge3, DEBT_TERM_UG_GPA ~ GENDER + RACE + HISPANIC_IND + FIRST_GENERATION_IND + balance_100 + DEBT_TERM_AGE_AT_THIRD_WEEK + EVER_PELL_ELIGIBLE_IND)
View(data_CART)
summary(data_CART$DEBT_TERM_UG_GPA)
hist(data_CART$DEBT_TERM_UG_GPA)



#Recursive partitioning trees (RPT) - decision tree classifier for missing data
#library(dlookr) #https://choonghyunryu.github.io/dlookr/reference/imputate_na.html or https://rdrr.io/cran/dlookr/man/imputate_na.html
#data_RPT <- imputate_na(clean_debt_discharge3, DEBT_TERM_UG_GPA, method="rpart") #rpart is Recursive partitioning tree; other options knn = k-nearest neightbor; mice; mode; median; mean
#other packages that are shown here - simputation;
#also tried dlookr (RPT), missMethods (EM), and mvdalab (EM) but couldn't quite get them to work with my dataset

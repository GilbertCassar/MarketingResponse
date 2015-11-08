# We want to find out which of the 1932 feature variables have a significant degree of association with the target variable.
# We will compare two approaches for feature selection: 
# The first one uses mathematical tests that measure the degree of association between variables. 
# The second fits a Random Forest to the whole dataset and uses the fitted model to determine which variables had the highest predictive significance.

require(randomForest)
require(readr)
require(heplots)

train <- read_csv("C:/Workspace/Kaggle/MarketingResponse/train.csv")

#Get the names of the feature variables
featureVariables <- names(train)[2:(ncol(train)-1)]

#Replace null values with -1
train[is.na(train)] <- -1

#A place holder for the measured degree of association
degreeOfAssociation <- data.frame( "featureName" = character(ncol(train)-2), "Value" = numeric(ncol(train)-2), stringsAsFactors=FALSE)

#Change any character features to factor
for (f in featureVariables) {
  if (class(train[[f]])=="character") {
    train[[f]] <- factor(train[[f]])
  }
}

for (i in 2:(ncol(train)-1)) {

  # The Eta Squared test to measure the degree of association between continuous variables and the categorical target variable. 
  if (class(train[,i])=="numeric" || class(train[,i])=="integer") {
	#Catch "residual sum of squares is 0" caused by feature variables that had too many null values
	tryCatch({
		model.aov <- aov(train[,i] ~ target, data = train)
		n = etasq(model.aov, partial = TRUE)
		cat(sprintf("%s has association with target of %s \n", featureVariables[i-1], sqrt(n[1,1])))
		degreeOfAssociation$featureName[i-1] <- featureVariables[i-1]
		degreeOfAssociation$Value[i-1]=sqrt(n[1,1])},
		error = function(e) 
		{
			cat(sprintf("%s caused an error because of too many (previously) null values and association was set to %s \n", featureVariables[i-1], 0))
			degreeOfAssociation$featureName[i-1] <- featureVariables[i-1]
			degreeOfAssociation$Value[i-1]=0
		}
	)
	
  # The Chi Squared test is used to measure the degree of association between categorical variables and the categorical target variable.
  } else if (class(train[,i])=="factor") {
		countDf <- subset(train, select=c(i,ncol(train)))
		cooccurenceTable = table(countDf)
		chisq <- chisq.test(cooccurenceTable)
		if(chisq$p.value < 0.05){
			cat(sprintf("%s has association with target of %s \n", featureVariables[i-1],1))
			degreeOfAssociation$featureName[i-1] <- featureVariables[i-1]
			degreeOfAssociation$Value[i-1]=1
		} else {
			cat(sprintf("%s has association with target of %s \n", featureVariables[i-1],0))
			degreeOfAssociation$featureName[i-1] <- featureVariables[i-1]
			degreeOfAssociation$Value[i-1]=0
		}	
  }
}

# Keep the features with degree of association > 0.1
usefulFeaturesT <- subset(degreeOfAssociation, Value > 0.1)

# Random Forest implementation cannot handle factors with more than 53 levels so these these variables are converted to Integer type
for (f in featureVariables) {
  if (class(train[[f]])=="factor") {
    if(nlevels(train[[f]]) > 53){
		train[[f]] <- as.integer(train[[f]])
	}
  }
}

trainSamp <- train[sample(nrow(train), 40000),]

# Fit the random forest model
rf <- randomForest(train[,featureVariables], factor(train$target), ntree=40, sampsize=5000, nodesize=2)

# In case of limited memory, the following sampling can be used
#trainSamp <- train[sample(nrow(train), 40000),]
#rf <- randomForest(trainSamp[,featureVariables], factor(trainSamp$target), ntree=40, sampsize=5000, nodesize=2)

# Get the feature importance values from the Random Forest model and keep only the features with importance > 1.5
impRF <- data.frame(featureName=rownames(rf$importance),Value=rf$importance)
colnames(impRF) <- c("featureName", "Value")
impRF$featureName <- as.character(impRF$featureName)
usefulFeaturesRF <- subset(impRF, Value > 1.5)
mostImportantRF <- subset(impRF, Value > 3)

# Inner join with the useful features found by the Approach1
commonUsefulFeatures <- merge(usefulFeaturesT,usefulFeaturesRF, by="featureName")
commonMostUseful <- merge(commonUsefulFeatures,mostImportantRF, by="featureName")

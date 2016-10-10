# ========================================================================
# ouvertures du fichier initial, nettoyage des données, initialisations des tableaux
# ========================================================================

setwd("~/Documents/Travail_Mines_de_Paris/2A/Apprentissage_Artificiel/Projet")
mydata <- read.csv("ad5.txt", header=FALSE)
mydata$V1[mydata$V1==56743] <- NA
mydata$V2[mydata$V2==56743] <- NA
mydata$V3[mydata$V3==56743] <- NA
mydatax <- na.omit(mydata)
mydatax$V1559 <- as.factor(mydatax$V1559)

# libraires utilisées
library(randomForest)
library(caret)
library(plyr)

# liste de C parcourus
Depth <- list()
Depth[1] = as.integer(200)
Depth[2] = as.integer(300)
Depth[3] = as.integer(400)
Depth[4] = as.integer(500)
Depth[5] = as.integer(600)
Depth[6] = as.integer(700)
Depth[7] = as.integer(800)
Depth[8] = as.integer(900)
Depth[9] = as.integer(1000)

# tableau des scores
Score_accuracy <- array(0.0, c(9, 9))

#Randomly shuffle the data
mydatax<-mydatax[sample(nrow(mydatax)),]

# Splittage des données : on garde un test absolu (1/10) et un ensemble de validation (9/10)
test1 <- rep(F, 2369)
test1[1:2132] = T
DataValidation <- mydatax[test1,]
test2 <- rep(F, 2369)
test2[2133:2368] = T
DataTest_absolu <- mydatax[test2,]

#Create 9 equally size folds
folds <- cut(seq(1,nrow(DataValidation)),breaks=9,labels=FALSE)

# ========================================================================
# 10-fold cross-validation sur la taille de la forêt
# ========================================================================

# On va faire de la cross-validaion
for (i in 1:9) {  # Boucle sur les valeurs de Depth (table Depth)
  
  for(j in 1:9){  #Perform 10 fold cross validation
    #Segement your data by fold using the which() function 
    testIndexes <- which(folds==j,arr.ind=TRUE)
    testData <- DataValidation[testIndexes, ]
    trainData <- DataValidation[-testIndexes, ]
    
    #Use the test and train data partitions
    modelForest <- randomForest(V1559~., 
                          data = trainData,
                          importance=TRUE,
                          keep.forest=TRUE,
                          ntree = as.integer(Depth[i])
    )
    predicted <- predict( modelForest, testData )
    actual <- testData$V1559

    # analyser les resultats
    cm <- confusionMatrix(as.factor(predicted), as.factor(actual))
    # stocker l'accuray
    overall.accuracy <- cm$overall['Accuracy']
    Score_accuracy[i,j] = overall.accuracy
    print(cat(i,j))
    print(Score_accuracy[i,j])
  } # fin boucle sur les 9 segments de cross-validation
  
} # fin Boucle sur Depth

# tableau des scores moyennés pour 
Score_accuracy_moy <- array(0.0, c(9))

# remplissage du tableau moyenné
for (i in 1:9) {  
  for (j in 1:9) {  
    Score_accuracy_moy[i] = Score_accuracy_moy[i] + Score_accuracy[i,j]
  }
  Score_accuracy_moy[i] = Score_accuracy_moy[i]/9
}

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# on lit dans la console que le gagnant est ntree = 400 ou 500
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# ========================================================================
# Utilisation du modèle gagnant
# ========================================================================

#Use the test and train data partitions
modelForest <- randomForest(V1559~., 
                            data = DataValidation,
                            importance=TRUE,
                            keep.forest=TRUE,
                            ntree = 400
)
predicted <- predict( modelForest, DataTest_absolu )
actual <- DataTest_absolu$V1559

# analyser les resultats
confusionMatrix(as.factor(predicted), as.factor(actual))
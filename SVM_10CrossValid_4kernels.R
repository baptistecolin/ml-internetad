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
library(kernlab)
library(caret)
library(plyr)

# liste de C parcourus
Cv <- list()
Cv[1] = 0.01
Cv[2] = 0.05
Cv[3] = 0.1
Cv[4] = 0.5
Cv[5] = 1
Cv[6] = 5
Cv[7] = 10
Cv[8] = 50
Cv[9] = 100

# tableau des scores
Score_accuracy <- array(0.0, c(4, 9, 9))

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
# kernel vanilladot
# ========================================================================

# On va faire de la cross-validaion
for (i in 1:9) {  # Boucle sur les valeurs de C (table Cv)
  
  for(j in 1:9){  #Perform 10 fold cross validation
    #Segement your data by fold using the which() function 
    testIndexes <- which(folds==j,arr.ind=TRUE)
    testData <- DataValidation[testIndexes, ]
    trainData <- DataValidation[-testIndexes, ]
    
    #Use the test and train data partitions
    modelSVM <- ksvm(V1559~., data=trainData, type='C-svc',
                     kernel='vanilladot',
                     C=Cv[i], scale=c() )
    predicted <- predict( modelSVM, testData )
    actual <- testData$V1559
    
    # analyser les resultats
    cm <- confusionMatrix(as.factor(predicted), as.factor(actual))
    # stocker l'accuray
    overall.accuracy <- cm$overall['Accuracy']
    Score_accuracy[1,i,j] = overall.accuracy
    print(cat(1,i,j))
    print(Score_accuracy[1,i,j])
  } # fin boucle sur les 9 segments de cross-validation
  
} # fin Boucle sur C

# tableau des scores moyennés pour 
Score_accuracy_moy <- array(0.0, c(4, 9))

# remplissage du tableau moyenné
for (i in 1:9) {  
  for (j in 1:9) {  
    Score_accuracy_moy[1,i] = Score_accuracy_moy[1,i] + Score_accuracy[1,i,j]
  }
  Score_accuracy_moy[1,i] = Score_accuracy_moy[1,i]/9
} 

# ========================================================================
# kernel rbfdot
# ========================================================================

#créer le kernel - valeurs d'hyper-parametres de kernel de l'exemple
rbf <- rbfdot(sigma=0.1)

# On va faire de la cross-validaion
for (i in 1:9) {  # Boucle sur les valeurs de C (table Cv)
  
  for(j in 1:9){  #Perform 10 fold cross validation
    #Segement your data by fold using the which() function 
    testIndexes <- which(folds==j,arr.ind=TRUE)
    testData <- DataValidation[testIndexes, ]
    trainData <- DataValidation[-testIndexes, ]
    
    #Use the test and train data partitions
    modelSVM <- ksvm(V1559~., data=trainData, type='C-svc',
                     kernel=rbf,
                     C=Cv[i], scale=c() )
    predicted <- predict( modelSVM, testData )
    actual <- testData$V1559
    
    # analyser les resultats
    cm <- confusionMatrix(as.factor(predicted), as.factor(actual))
    # stocker l'accuray
    overall.accuracy <- cm$overall['Accuracy']
    Score_accuracy[2,i,j] = overall.accuracy
    print(cat(2,i,j))
    print(Score_accuracy[2,i,j])
  } # fin boucle sur les 9 segments de cross-validation
  
} # fin Boucle sur C

# tableau des scores moyennés pour 
for (i in 1:9) {  
  for (j in 1:9) {  
    Score_accuracy_moy[2,i] = Score_accuracy_moy[2,i] + Score_accuracy[2,i,j]
  }
  Score_accuracy_moy[2,i] = Score_accuracy_moy[2,i]/9
} 

# ========================================================================
# kernel polydot
# ========================================================================

#créer le kernel - valeurs d'hyper-parametres de kernel de l'exemple
pld <- polydot(degree = 1, scale = 1, offset = 1)

# On va faire de la cross-validaion
for (i in 1:9) {  # Boucle sur les valeurs de C (table Cv)
  
  for(j in 1:9){  #Perform 10 fold cross validation
    #Segement your data by fold using the which() function 
    testIndexes <- which(folds==j,arr.ind=TRUE)
    testData <- DataValidation[testIndexes, ]
    trainData <- DataValidation[-testIndexes, ]
    
    #Use the test and train data partitions
    modelSVM <- ksvm(V1559~., data=trainData, type='C-svc',
                     kernel=pld,
                     C=Cv[i], scale=c() )
    predicted <- predict( modelSVM, testData )
    actual <- testData$V1559
    
    # analyser les resultats
    cm <- confusionMatrix(as.factor(predicted), as.factor(actual))
    # stocker l'accuray
    overall.accuracy <- cm$overall['Accuracy']
    Score_accuracy[3,i,j] = overall.accuracy
    print(cat(3,i,j)) #pour vérifier avancement du calcul
    print(Score_accuracy[3,i,j])
  } # fin boucle sur les 9 segments de cross-validation
  
} # fin Boucle sur C

# tableau des scores moyennés pour 
for (i in 1:9) {  
  for (j in 1:9) {  
    Score_accuracy_moy[3,i] = Score_accuracy_moy[3,i] + Score_accuracy[3,i,j]
  }
  Score_accuracy_moy[3,i] = Score_accuracy_moy[3,i]/9
} 

# ========================================================================
# kernel tanhdot
# ========================================================================

#créer le kernel - valeur de sigma = valeur de l'exemple
bsld <- besseldot(sigma = 1, order = 1, degree = 1)

# On va faire de la cross-validaion
for (i in 1:9) {  # Boucle sur les valeurs de C (table Cv)
  
  for(j in 1:9){  #Perform 10 fold cross validation
    #Segement your data by fold using the which() function 
    testIndexes <- which(folds==j,arr.ind=TRUE)
    testData <- DataValidation[testIndexes, ]
    trainData <- DataValidation[-testIndexes, ]
    
    #Use the test and train data partitions
    modelSVM <- ksvm(V1559~., data=trainData, type='C-svc',
                     kernel=bsld,
                     C=Cv[i], scale=c() )
    predicted <- predict( modelSVM, testData )
    actual <- testData$V1559
    
    # analyser les resultats
    cm <- confusionMatrix(as.factor(predicted), as.factor(actual))
    # stocker l'accuray
    overall.accuracy <- cm$overall['Accuracy']
    Score_accuracy[4,i,j] = overall.accuracy
    print(cat(4,i,j)) #pour vérifier avancement du calcul
    print(Score_accuracy[4,i,j])
  } # fin boucle sur les 9 segments de cross-validation
  
} # fin Boucle sur C

# tableau des scores moyennés pour 
for (i in 1:9) {  
  for (j in 1:9) {  
    Score_accuracy_moy[4,i] = Score_accuracy_moy[4,i] + Score_accuracy[4,i,j]
  }
  Score_accuracy_moy[4,i] = Score_accuracy_moy[4,i]/9
} 

# ========================================================================
# finitions
# ========================================================================

#Dilaltion des moyennes
Score_accuracy_moy = 100*Score_accuracy_moy #passage en pourcents
Score_accuracy_moy

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# on lit dans la console que le gagnant est polydot avec C = 0.5
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# ========================================================================
# Utilisation du modèle gagnant
# ========================================================================

#Use the test and train data partitions
modelSVM <- ksvm(V1559~., data=DataValidation, type='C-svc',
                 kernel=pld,
                 C=0.5, scale=c() )
predicted <- predict( modelSVM, DataTest_absolu )
actual <- DataTest_absolu$V1559
confusionMatrix(as.factor(predicted), as.factor(actual))

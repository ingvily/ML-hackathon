
Komme igang med AWS
====================

1. Last opp treningsdata og testsett til S3
Last opp filen `traindata/train.csv` til en S3-bøtte. 
Dette kan du gjøre via consolet eller med `AddDataToS3.py`.
	

2. Lag en Datasource av datasettet 
Opprett en datasources for trening som peker til `train_scaled.csv`.
Dette må vi gjøre i consolet. 
	
3. Lag en modell med treningsdataen
Lag en modell som bruker trenignsdataen. 
Dette kan du gjøre i consolet eller med `CreateModel.py`. Dersom du bruker python-skriptet trenger du id-en til treningssettet, den finner du i consolet.


4. Evaluate modellen
For å se hvor god modellen er på ukjent data skal vi evaluere den på valideringssettet som AWS satt til side for deg fra datasourcesn i punkt 2) da du trente modellen.
Gjør dette enten i consolet eller med `CreateModel.py`. Også her trenger du id-en til valideringssettet, som du finner i consolet. 


5. Prediker ukjent data
Nå skal vi bruke modellen predikere strømpris på ny data som ikke inneholder `target`! 
Da må vi gjenta steg 1. og 2. med datasettet vi ønsker å predikere (på formatet som `predict.csv`). 

Bruk så modellen fra steg 3. til å predikere. Det kan du gjøre i consolet ved å opprette en `Batch Predictions` eller med `MakePredictions.py`. 
Resultatet havner i S3. 


Gratulerer! Du har laget en modell i AWS! Hva gjør vi nå? 
=========================================================

Her er noen tips til hvordan du kan forbedre modellen: 

- Skaler datasettet. Bruk f.eks. datasettet `traindata/train_scaled.csv` 

- Vi har nå brukt default oppsett til AWS. Du kan stille på noen parametre. Se mer her: http://docs.aws.amazon.com/machine-learning/latest/dg/training-parameters.html

- Modellen vvi har laget bruker linear regresjon. Du vil kunne tjene på å preprossesere dataen ytterligere før den lastes opp til modellen. F.eks. en ny kolonne med kvadratet av en annen kolonne vil gi deg regresjon av høyere orden. Dette kan gjøres i en `Recipe` eller manuelt. Se http://docs.aws.amazon.com/machine-learning/latest/dg/feature-transformations-with-data-recipes.html

- Som jeg var inne på over - vi har brukt regresjon. AWS tilbyr også Binary Classification og Multiclass Classification. Kan disse brukes til noe på dette datasettet? 

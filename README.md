# Sign alphabet classification
## Wybrane zagadnienia uczenia maszynowego - Projekt 2023 - Miłosz Gajewski
## Zadanie
Budowa zbioru danych gestów alfabetu migowego przy użyciu dostarczonego narzędzia, 
oraz przygotowanie modelu uwzględniającego preprocessing danych i klasyfikatora.

Program rozpoznaje znaki: ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y'].
## Przygotowanie danych
TODO
## Wybór klasyfikatora
TODO
### Przeprowadzone testy
W celu wybor odpowiedniego klasyfikatora przeprowadzone zostały testy z wykorzystaniem GridSearchCV. Parametrem wyboru była wartość 'accuracy'. Wyniki przedstawiono w tabeli:

| Klasyfikator           | Parametry                                                                                  | Uzyskany wynik <br/> dane uczące | Uzyskany wynik <br/> dane testowe |
|------------------------|--------------------------------------------------------------------------------------------|----------------------------------|-----------------------------------|
| SCV                    | {'C': 1000, 'degree': 1, 'gamma': 0.9, 'kernel': 'poly'}                                   | 93.30 %                          | 93.23%                            |
| KNeighborsClassifier   | {'algorithm': 'ball_tree', 'leaf_size': 5, 'n_neighbors': 1, 'weights': 'uniform'}         | 70.18 %                          | 73.89%                            |
| LinearSVC              | {'C': 1000, 'dual': False, 'loss': 'squared_hinge', 'multi_class': 'ovr', 'penalty': 'l2'} | 92.85 %                          | 94.06%                            |
| RandomForestClassifier | {'criterion': 'entropy', 'max_features': None, 'n_estimators': 200}                        | 80.91%                           | 81.90%                            |
| MLPClassifier          | {}                                                                                         | 90.57%                           | 93.23%                            |
| SGDClassifier          | {'loss':'log_loss'}                                                                        | 80.02%                           | 83.04%                            |
### Wybrany klasyfikator
SCV : {'C': 1000, 'degree': 1, 'gamma': 0.9, 'kernel': 'poly'}
## Uruchomienie programu
```console
TODO
```

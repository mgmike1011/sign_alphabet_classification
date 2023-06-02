# Sign alphabet classification
## Wybrane zagadnienia uczenia maszynowego - Projekt 2023 - Miłosz Gajewski
## Zadanie
Budowa zbioru danych gestów alfabetu migowego przy użyciu dostarczonego narzędzia, 
oraz przygotowanie modelu uwzględniającego preprocessing danych i klasyfikatora.

Program rozpoznaje znaki: ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y'].
## Przygotowanie danych
Stworzone na potrzeby trenowania oraz testowania klasyfikatorów zbiory danych były wynikiem współpracy całej grupy specjalizacji Roboty i Systemy Autonomiczne. Do przechowywania i wymiany danych użyty został system internetowych arkuszy kalkulacyjnych Google.
### Wykorzystane zbiory danych do wykonania zadania
Wykorzystane zbiory danych do analizy, trenowania klasyfikatorów i testowania znajdują się w katalogu [datasets](datasets).
### Analiza dostarczonych danych
Pierwszym etapem zadania była analiza otrzymanych danych [Data_analysis](analysis/Data_analysis.ipynb). Rozpoczęto od sprawdzenia dostępnych znaczników pomiarowych, zakresu danych w poszczególnych kolumnach, podzieleniu na etykiety oraz dane właściwe, jak również wizualizacji danych w celu określenia ubytków i nieprawidłowości. Otrzymane dane charakteryzują się stosunkowo równomierną rozpiętością, jednak na dalszych etapach pracy zostały one sprowadzone do zerowej wartości średniej i jednostkowego odchylenia standardowego, w celu polepszenia uzyskiwanych wyników modelu. Na potrzeby uczenia klasyfikatora użyto jedynie znaczników względnych, odrzucając dane pochodzące z „układu świata”, dodatkowo odrzucono dane z kolumn 'handedness.score' oraz 'handedness.label', gdyż wykazano, że nie wpływają one na uzyskiwane wyniki. Początkowo etykiety danych zostały zamienione na wartości liczbowe według wzoru: {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'i': 8, 'k': 9, 'l': 10, 'm': 11, 'n': 12, 'o': 13, 'p': 14, 'q': 15, 'r': 16, 's': 17, 't': 18, 'u': 19, 'v': 20, 'w': 21, 'x': 22, 'y': 23}, jednak na dalszych etapach pracy okazało się, że transformacja nie wpływa na uzyskiwane wyniki klasyfikatora a jedynie wydłuża czas działania programu, dlatego zrezygnowano z tego rozwiązania.
## Wybór klasyfikatora
W celu wybrania odpowiedniego klasyfikatora przeprowadzono szereg testów dla różnych klasyfikatorów i parametrów dla nich [Model_selection](analysis/Model_selection.ipynb), na podstawie uzyskanych wyników wybrano najlepiej rokujące klasyfikatory (SVC, LinearSVC oraz MLPClassifier) i przeprowadzono na nich testy związane z łączeniem klasyfikatorów [SVC_analysis](analysis/SVC_analysis.ipynb).
### Przeprowadzone testy
W celu wyboru odpowiedniego klasyfikatora przeprowadzone zostały testy z wykorzystaniem GridSearchCV. Parametrem wyboru była wartość 'accuracy'. Wyniki przedstawiono w tabeli:

| Klasyfikator                               | Parametry                                                                                                                                                                                                 | Uzyskany wynik <br/> dane uczące | Uzyskany wynik <br/> dane testowe |
|--------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------|-----------------------------------|
| SCV                                        | {'C': 1000, 'degree': 1, 'gamma': 0.9, 'kernel': 'poly'}                                                                                                                                                  | 93.30 %                          | 93.23%                            |
| KNeighborsClassifier                       | {'algorithm': 'ball_tree', 'leaf_size': 5, 'n_neighbors': 1, 'weights': 'uniform'}                                                                                                                        | 70.18 %                          | 73.89%                            |
| LinearSVC                                  | {'C': 1000, 'dual': False, 'loss': 'squared_hinge', 'multi_class': 'ovr', 'penalty': 'l2'}                                                                                                                | 92.85 %                          | 94.06%                            |
| RandomForestClassifier                     | {'criterion': 'entropy', 'max_features': None, 'n_estimators': 200}                                                                                                                                       | 80.91%                           | 81.90%                            |
| MLPClassifier                              | {}                                                                                                                                                                                                        | 90.57%                           | 93.23%                            |
| SGDClassifier                              | {'loss':'log_loss'}                                                                                                                                                                                       | 80.02%                           | 83.04%                            |
| Voting: SVC+LinearSVC+MLPClassifier        | SVC: {'C': 1000, 'degree': 1, 'gamma': 0.9, 'kernel': 'poly'}, LinearSVC:  {'C': 1000, 'dual': False, 'loss': 'squared_hinge', 'multi_class': 'ovr', 'penalty': 'l2'} , MLPClassifier: {'max_iter': 1000} | 99.34%                           | 94.94%                            |
| Stacking: SVC+LinearSVC+MLPClassifier->SVC | SVC: {'C': 1000, 'degree': 1, 'gamma': 0.9, 'kernel': 'poly'}, LinearSVC:  {'C': 1000, 'dual': False, 'loss': 'squared_hinge', 'multi_class': 'ovr', 'penalty': 'l2'} , MLPClassifier: {'max_iter': 1000} | 91.55%                           | 87.93%                            |

### Wybrany klasyfikator
```python
mdl_pipe = Pipeline([
    ('standard_scaler', StandardScaler()),
    ('classifier', VotingClassifier([
        ('SVC_1', SVC(C=1000, degree=1, gamma=0.9, kernel='poly')),
        ('MLP', MLPClassifier(max_iter=1000)),
        ('Lin_SVC_1', LinearSVC(C=1000, dual=False, loss='squared_hinge', multi_class='ovr', penalty='l2', max_iter=10000))
    ]))
])
```
## Uruchomienie programu
```console
python3 main.py /path/to/dataset/Dataset.csv /path/to/output.txt
```
## Struktura pliku wynikowego programu
a</br>
x</br>
e</br>
...

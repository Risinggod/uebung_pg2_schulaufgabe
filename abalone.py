import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from sklearn.preprocessing import StandardScaler

# 1.1 Daten aus CSV laden
path = "/Users/student/PycharmProjects/übung_pg2_schulaufgabe/abalone.data"
data = pd.read_csv(path, delimiter=",")

#den Spalten überschrieften geben
data.columns = ["Sex", "Length", "Diameter", "Height", "Whole_weight", "Shucked_weight", "Viscera_weight", "Shell_weight", "Rings"]

#von stings in ints umwandeln
data_unknown = data
data_unknown = data_unknown.astype("category")
data_unknown = data_unknown.apply(lambda x: x.cat.codes)

# 1.2 die ersten und letzten 5 Zeilen ausgeben
print('#'*100)
print('Aufgabe 1.2')

print(data_unknown.head())
print(data_unknown.tail())

# 1.3 Anzahl Zeilen und Spalten + Datentyp
print('#'*100)
print('Aufgabe 1.3')

print(data_unknown.info())

# 2.1 Überprüfen Sie den Datensatz auf fehlende Werte und behandeln Sie diese entsprechend (z.B. durch Entfernen oder Ersetzen).
print('#'*100)
print('Aufgabe 2.1')

print(data_unknown.isnull().any())

# 2.2 Duplikate entfernen
print('#'*100)
print('Aufgabe 2.2')

print(data_unknown.duplicated())
data = data_unknown.drop_duplicates()
print(data_unknown.duplicated())

model = KMeans()
visualizer = KElbowVisualizer(model, K=(2, 9), timings=False)
visualizer.fit(data)
visualizer.show()

KMeans = KMeans(n_clusters=4)

#ergebnis
pred = KMeans.fit_predict(data)
data_new = pd.concat([data, pd.DataFrame(pred, columns=["label"])], axis=1)
print(data_new)

# 3.1 statistische Kennzahlen
print('#'*100)
print('Aufgabe 3.1')

descriptive_statistic = data_unknown.describe()
print(descriptive_statistic)

# 3.2 Ermitteln Sie die Korrelation zwischen der Qualität des Weins und anderen chemischen Eigenschaften.
print('#'*100)
print('Aufgabe 3.2')

correlation = data_unknown.corr()
print(correlation['Rings'].sort_values())

# 4.1 Histogramm
print('#'*100)
print('Aufgabe 4.1')
print('Histogramm')

data_unknown['Diameter'].hist()
plt.title('Verteilung des Durchmessers')
plt.show()

# 4.2 ScatterPlot
print('#'*100)
print('Aufgabe 4.2')
print('ScatterPlot')

plt.scatter(data_unknown['Diameter'], data_unknown['Rings'])
plt.title('Diameter vs. Ringe von der Schneke')
plt.show()

# 4.3 BoxPlot
print('#'*100)
print('Aufgabe 4.3')
print('BoxPlot')

data_unknown.boxplot(column='Length', by='Rings')
plt.title('Länge nach ringen')
plt.show()
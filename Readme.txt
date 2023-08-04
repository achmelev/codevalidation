Das ist ein Test-Repo zum Trainieren eines kleinen neuronalen Netzwerks.
Das neuronale Netzwerk liest ein Zifferncode bestehend aus X 9-Ziffernguppen (0-9) gefolgt von X Prüfziffern für jede Ziffergruppe.
Die Prüfziffer ist die  max. Ziffer der jeweiligen Ziffergruppe. Das Netzwerk gibt aus ob der 10X-lange Code Valide ist
(Also richtige Prüfziffern hat), oder nicht.

Folgende Skripte/Dateien sind implementiert:

- train.conf - die Konfigurationsdatei für alle Skripte. Enthält zwei Abschnitte.  
Jeweils ein fürs Training mit und ohne Cuda. Speziell der Parameter width bestimmt die Anzahl (X) der Zifferngruppen.
Ansonsten siehe train.py (die Implementierung des Trainirens).


- generate_dateset - Zum Generieren eines Datensets. 
    Aufruf: python3 generate_dataset <Anzahl der Zifferngruppen in Codes> <Anzahl der Datensätze> <Dateiname>
- train.py - Zum Trainieren des Netzwerks. Das Ergebnis (das serialisierte Netzwerk) wird in der Datei abgelegt, 
die im train.conf - Parameter model_file konfiguriert ist.
    Aufruf python3 train.py nocuda|cuda
- test_model.py - zum Testen des trainierten Netzwerks. Lädt das Netzwerk aus der Datei, erzeugt 1000 zufällige Codes
und prüft, ob das Netzwerk das richtige Ergebnis liefert.
    Aufruf: python3 test_model.py nocuda|cuda
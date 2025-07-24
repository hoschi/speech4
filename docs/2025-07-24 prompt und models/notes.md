# Prompt und Modelle

Also habe ich verschiedenes ausprobiert. Ich habe herausgefunden, dass man ein YouTube-Transkript relativ schnell und einfach runterladen kann. Dafür gibt es existierende Tools. Der erste Ansatz war das komplette Skript mit den Zeitstempeln als zusätzlichen Kontext der KI mitzugeben. Das hat so überhaupt nicht funktioniert, da man hier für mein durchschnittliches Beispiel-Video schon mehr als 20.000 Tokens braucht. Es gibt auch Modelle, die das schaffen und die auch lokal laufen, sehr spät habe ich dann aber rausgefunden, dass das Modell stark verlangsamt wird. Umso mehr im System Prompt steht, umso langsamer wird alles. Deshalb habe ich dann ein Skript von der KI erstellen lassen, um nur den Text aus dem Transkript zu extrahieren. Mit diesem Text als Kontext hab meine Beispielschnipsel bestehend aus zwei Sätzen, aber immer noch ungefähr 50 Sekunden gedauert, um verbessert zu werden. Als nächsten Schritt habe ich dann der KI aufgetragen, aus dem Transkript alle wichtigen Namen und technischen Begriffe zu extrahieren und aufzulisten. Außerdem das Video in 50 Wörtern zusammenfassen. Beides funktioniert sehr gut, um spezielle Begriffe wie Obsidian als Notizsystem, Chat-GPT 4 als spezielle Version eines Modells oder auch Eigennamen wie Dailymotion besser zu erkennen. Ich habe kurz auch probiert, ohne diesen Kontext die gleichen Teile übersetzen zu lassen, was trotzdem gut funktioniert hat, aber eben bei diesen Eigennamen nicht sehr gut. Hier sind wir dann bei circa 20 Sekunden pro Schnipsel. Meine Tests habe ich mit dem Web-Interface Open Web UI gemacht, das automatisch auch alle vorherigen Nachrichten wieder dem Modell zur Verfügung stellt. Hier kann man noch das Gedächtnis entfernen, dass jede Nachricht nur den System-Prompt als Kontext bekommt. Interessanterweise waren auch die kleineren Modelle schneller wie die größeren, ich hatte das verstanden, dass es eigentlich andersrum sein sollte. Getestete Modelle:

* Llama3.1:8b
* Mistral-Nenom:12b
* Gemma3:12b
* Gemma3:4b

Temperatur war immer auf 0,3 gestellt.
Was auch besser funktioniert hat zu meinem Prompt vorher war die Eingabe in `<note>` Tags zu umschließen, damit die KI unterscheiden kann, was eine Frage an sie ist und was Text, den sie übersetzen soll. Die Extraktion vom Transkript in die wichtigsten Daten kann man gleich machen, wenn in der UI die YouTube ID eingetragen wurde und zum Server geschickt wird.

Der Prompt zur Extrahierung war einfach gehalten:
Extrahiere alle Namen und technischen Begriffe aus diesem Text, außerdem erstelle eine Zusammenfassung des Textes bestehend aus 50 Worten.

Die 50-Wörter-Zusammenfassung soll wirklich nur dazu dienen, um ein Video über KIs von einem Video über Holzwerken unterscheiden zu können. So kann die KI schon sehr gut unterscheiden, ob es sich bei "Bank" um Finanzen oder die Sitzgelegenheit handelt.

Was man natürlich auch noch machen kann ist auf Hugging Face zu sehen ob es spezielle Modelle für diesen Use Case gibt. Wichtig ist aber dann auch dass diese auf Deutsch mit englischen Begriffen trainiert wurde.
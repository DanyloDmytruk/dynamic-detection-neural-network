## Структура

- `windows/collector.py` — збирач подій Sysmon для Windows VM.
- `windows/collect_batch.py` — пакетний запуск зразків у контрольованій VM.
- `dynamic/predict.py` —  аналіз ознак і винесення рішення по шкідливості пз.

Приклад перевірки JSON логу подій моделлю:
<img width="1164" height="368" alt="image" src="https://github.com/user-attachments/assets/c18c6a3a-96ee-4cd0-9662-debe7c3d1d4a" />


## Важливо щодо безпеки

`windows/collector.py` та `windows/collect_batch.py` призначені виключно для ізольованої Windows VM, створеної для оборонного дослідження. Перед роботою з невідомими зразками зробіть snapshot, вимкніть спільний буфер/перетягування/спільні каталоги та не використовуйте робочу машину. Віртуалізація знижує ризик, але не гарантує повної ізоляції.
